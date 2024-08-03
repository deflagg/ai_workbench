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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    script_path: str
    voiceover_path: str
    project_path: str

def create_agent(llm, system_message: str, tools):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4")

# Define the tools
tools = [tts_whisper, create_resolve_project, add_audio_track]
tool_node = ToolNode(tools)


def handle_agent_exception(agent_name: str, e: Exception) -> dict:
    print(f"Error in agent_node for {agent_name}: {e}")
    return {"messages": [], "sender": agent_name}

def agent_node(state: AgentState, agent, name) -> AgentState:
    """Create a node for a given agent."""
    try:
        result = agent.invoke(state)
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        content = result.content
        return {
            "messages": [result],
            "sender": name,
            "script_path": "",
            "voiceover_path": "",
            "project_path": "",
        }
    except Exception as e:
        return handle_agent_exception(name, e)

# Define agent nodes
writer_node = functools.partial(
    agent_node,
    agent = create_agent(
        llm,
        system_message="You are an expert YouTube content creator. Create engaging and informative video scripts based on the given topic. Output only the script text, no additional explanations.",
        tools=[]
    ),
    name = "writer_node"
)

voiceover_node = functools.partial(
    agent_node,
    agent = create_agent(
        llm,
        system_message="You are an AI assistant specialized in creating voiceovers. Use the tts_whisper tool to generate a voiceover for the given script. Return the path to the generated audio file.",
        tools=[tts_whisper]
    ),
    name = "voiceover_node"
)

project_setup_node = functools.partial(
    agent_node,
    agent = create_agent(
        llm,
        system_message="You are an AI assistant specialized in setting up DaVinci Resolve projects. Use the create_resolve_project tool to create a new project for the YouTube video. Return the path to the created project.",
        tools=[create_resolve_project]
    ),
    name = "project_setup_node"
)

audio_node = functools.partial(
    agent_node,
    agent = create_agent(
        llm,
        system_message="You are an AI assistant specialized in managing audio within DaVinci Resolve projects. Use the add_audio_track tool to add the voiceover audio to the project timeline. Confirm when the task is completed.",
        tools=[add_audio_track]
    ),
    name = "audio_node"
)


# Define the edge logic
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
        
def create_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("writer_node", writer_node)
    workflow.add_node("voiceover_node", voiceover_node)
    workflow.add_node("project_setup_node", project_setup_node)
    workflow.add_node("audio_node", audio_node)
    workflow.add_node("call_tool", tool_node)
    workflow.set_entry_point("writer_node")

    workflow.add_conditional_edges(
        "writer_node",
        router,
        {"call_tool": "call_tool", "continue": "voiceover_node", "__end__": END},
    )
    
    workflow.add_conditional_edges(
        "voiceover_node",
        router,
        {"call_tool": "call_tool", "continue": "project_setup_node", "__end__": END},
    )
    
    workflow.add_conditional_edges(
        "project_setup_node",
        router,
        {"call_tool": "call_tool", "continue": "audio_node", "__end__": END},
    )
    
    
    workflow.add_conditional_edges(
        "audio_node",
        router,
        {"call_tool": "call_tool", "continue": END, "__end__": END},
    )
    

    return workflow.compile()