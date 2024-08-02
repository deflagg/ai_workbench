import os
import sys
import operator
import functools
import langchain_openai

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

sys.path.append('./')
from experiments.helpers.debugging_helpers import display_langgraph_graph
from experiments.helpers.tools.generic_tools import get_current_datetime
from experiments.helpers.tools.web_tools import google_search
from experiments.helpers.tools.audio_tools import tts_whisper
#from experiments.helpers.tools.davinci_tools import create_project

# Load environment variables from .env file
load_dotenv("/.env")

# Helper function to create an agent
def create_agent(llm, tools, system_message: str):
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
    return prompt | llm.bind_tools(tools)

# Define the object that is passed between each node in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    """Create a node for a given agent."""
    try:
        result = agent.invoke(state)
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            "sender": name,
        }
    except Exception as e:
        print(f"Error in agent_node for {name}: {e}")
        return {
            "messages": [],
            "sender": name,
        }

# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4o-mini")


# YouTube content creator agent and node
writer_agent = create_agent(
    llm,
    [get_current_datetime, tts_whisper],
    system_message="You are an expert YouTube content creator. You are tasked with creating video scripts and then converting it to audio."
)
writer_node = functools.partial(agent_node, agent=writer_agent, name="youtube content creator")

# Voiceover artist agent and node
voiceover_agent = create_agent(
    llm,
    [tts_whisper],
    system_message="You are the editor and voiceover creator. You are tasked with editing the script first and then recording voiceovers for the video scripts."
)
voiceover_node = functools.partial(agent_node, agent=voiceover_agent, name="youtube voiceover creator")

# Define the tools
tools = [get_current_datetime, tts_whisper]
tool_node = ToolNode(tools)

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("writer_node", writer_node)
#workflow.add_node("voiceover_node", voiceover_node)
workflow.add_node("call_tool", tool_node)

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

workflow.add_conditional_edges(
    "writer_node",
    router,
    {"call_tool": "call_tool", "continue": END, "__end__": END},
)

# Define the edge logic for the voiceover_node
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

# workflow.add_conditional_edges(
#     "voiceover_node",
#     voiceover_router,
#     {"call_tool": "call_tool", "__end__": END},
# )

workflow.set_entry_point("writer_node")
graph = workflow.compile()

# Optionally display the graph
showGraph = False
if showGraph:
    try:
        display_langgraph_graph(graph, "", figure_size=(10, 6))
    except Exception as e:
        print(f"Error displaying graph: {e}")

# Main loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    try:
        events = graph.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ],
            },
            # Maximum number of steps to take in the graph
            {"recursion_limit": 150},
        )
        for s in events:
            print(s)
            print("----")
    except Exception as e:
        print(f"Error processing user input: {e}")