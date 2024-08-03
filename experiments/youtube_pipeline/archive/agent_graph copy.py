import sys
import operator
import functools
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Import utility functions
from experiments.helpers.debugging_helpers import display_langgraph_graph
from experiments.helpers.tools.generic_tools import get_current_datetime
from experiments.helpers.tools.audio_tools import tts_whisper
from experiments.helpers.tools.web_tools import google_search
from experiments.helpers.tools.davinci_tools import create_resolve_project, add_audio_track, add_video_clips

# Define the object that is passed between each node in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def create_prompt(system_message: str, tools: list) -> ChatPromptTemplate:
    """Create a prompt template for the agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt.partial(system_message=system_message).partial(
        tool_names=", ".join([tool.name for tool in tools])
    )


def create_agent(llm, tools: list, system_message: str) -> ChatPromptTemplate:
    """Create an agent with the provided tools and system message."""
    prompt = create_prompt(system_message, tools)
    return prompt | llm.bind_tools(tools)


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
tools = [get_current_datetime, tts_whisper, create_resolve_project, add_audio_track]
tool_node = ToolNode(tools)

# Define agents
writer_agent = create_agent(
    llm,
    [get_current_datetime],
    system_message="You are an expert YouTube content creator."
)

voiceover_agent = create_agent(
    llm,
    [tts_whisper],
    system_message="You are the editor and voiceover creator for a YouTube channel."
)


# Define agent nodes
writer_node = functools.partial(agent_node, agent=writer_agent, name="youtube_content_creator")
voiceover_node = functools.partial(agent_node, agent=voiceover_agent, name="youtube_voiceover_creator")

def create_graph() -> StateGraph:
    """Create and return the state graph."""
    
    # Define the graph nodes and entry point node
    workflow = StateGraph(AgentState)
    workflow.add_node("writer_node", writer_node)
    workflow.add_node("voiceover_node", voiceover_node)
    workflow.add_node("call_tool", tool_node)
    workflow.set_entry_point("writer_node")

    # Defice graph edges
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

    def voiceover_router(state) -> Literal["call_tool", "__end__"]:
        """Router function for voiceover_node to end the process."""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return "call_tool"
            if "FINAL ANSWER" in last_message.content:
                return "__end__"
            return "__end__"
        except KeyError as e:
            print(f"Error in voiceover_router function: {e}")
            return "__end__"
        
    workflow.add_conditional_edges(
        "voiceover_node",
        voiceover_router,
        {"call_tool": "call_tool", "__end__": END},
    )

    # Compile and return the graph
    graph = workflow.compile()
    return graph

