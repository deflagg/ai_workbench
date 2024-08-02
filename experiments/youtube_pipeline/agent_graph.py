import sys
import operator
import functools
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Import utility functions
from experiments.helpers.debugging_helpers import display_langgraph_graph
from experiments.helpers.tools.audio_tools import tts_whisper
from experiments.helpers.tools.davinci_tools import create_resolve_project, add_audio_track

# Define the object that is passed between each node in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def create_prompt(system_message: str, tools: list = None) -> ChatPromptTemplate:
    """Create a prompt template for the agent."""
    messages = [
        (
            "system",
            (
                "You are a helpful AI assistant, collaborating with other assistants."
                " Execute tasks to the best of your ability using the tools provided when necessary."
                " If you have completed all required tasks, prefix your response with FINAL ANSWER."
                f"\n{system_message}"
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    
    if tools:
        messages[0] = (
            "system",
            messages[0][1] + f" You have access to the following tools: {', '.join([tool.name for tool in tools])}."
        )
    
    return ChatPromptTemplate.from_messages(messages)

def create_agent(llm, system_message: str, tools: list = None):
    """Create an agent with the provided system message and optional tools."""
    prompt = create_prompt(system_message, tools)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

def handle_agent_exception(agent_name: str, e: Exception) -> dict:
    """Handle exceptions for agent nodes."""
    print(f"Error in agent_node for {agent_name}: {e}")
    return {"messages": [], "sender": agent_name}

def agent_node(state, agent, name) -> dict:
    """Create a node for a given agent."""
    try:
        result = agent.invoke(state)
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {"messages": [result], "sender": name}
    except Exception as e:
        return handle_agent_exception(name, e)

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the tools
tools = [tts_whisper, create_resolve_project, add_audio_track]
tool_node = ToolNode(tools)

# Define agents
writer_agent = create_agent(
    llm,
    system_message="You are an expert YouTube content creator. Create engaging and informative video scripts."
)

voiceover_agent = create_agent(
    llm,
    system_message="You are an AI assistant specialized in video production. Your responsibilities include creating voiceovers, setting up DaVinci Resolve projects, and managing audio within these projects. Use the appropriate tools to complete tasks related to these areas without needing explicit instructions for each step.",
    tools=tools
)

# Define agent nodes
writer_node = functools.partial(agent_node, agent=writer_agent, name="youtube_content_creator")

def voiceover_node(state: AgentState) -> AgentState:
    try:
        # Process the input state
        result = voiceover_agent.invoke(state)
        result_message = AIMessage(**result.dict(exclude={"type", "name"}), name="youtube_voiceover_creator")
        
        # Check for tool calls
        if result_message.tool_calls:
            for tool_call in result_message.tool_calls:
                tool_result = tool_node.invoke({
                    "messages": state["messages"] + [result_message],
                    "sender": "youtube_voiceover_creator"
                })
                result_message.content += f"\n\nTool result: {tool_result['messages'][-1].content}"
        
        return {"messages": [result_message], "sender": "youtube_voiceover_creator"}
    except Exception as e:
        return handle_agent_exception("youtube_voiceover_creator", e)

def create_graph() -> StateGraph:
    """Create and return the state graph."""
    
    # Define the graph nodes and entry point node
    workflow = StateGraph(AgentState)
    workflow.add_node("writer_node", writer_node)
    workflow.add_node("voiceover_node", voiceover_node)
    workflow.add_node("call_tool", tool_node)
    workflow.set_entry_point("writer_node")

    # Define graph edges
    def router(state) -> Literal["call_tool", "__end__", "continue"]:
        """Router function to determine next steps."""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "call_tool"
            if "FINAL ANSWER" in last_message.content:
                return "__end__"
            return "continue"
        except KeyError as e:
            print(f"Error in router function: {e}")
            return "continue"
        
    workflow.add_conditional_edges(
        "writer_node",
        router,
        {"call_tool": "call_tool", "continue": "voiceover_node", "__end__": END},
    )

    def voiceover_router(state: AgentState) -> Literal["call_tool", "continue", "__end__"]:
        try:
            messages = state["messages"]
            last_message = messages[-1]
            all_messages_content = ' '.join([msg.content for msg in messages])
            
            if last_message.tool_calls:
                return "call_tool"
            
            tools_used = [tool.name in all_messages_content for tool in tools]
            
            if all(tools_used) and "FINAL ANSWER" in last_message.content:
                return "__end__"
            
            return "continue"
        except KeyError as e:
            print(f"Error in voiceover_router function: {e}")
            return "__end__"
        
    workflow.add_conditional_edges(
        "voiceover_node",
        voiceover_router,
        {"call_tool": "call_tool", "continue": "voiceover_node", "__end__": END},
    )

    # Compile and return the graph
    graph = workflow.compile()
    return graph

