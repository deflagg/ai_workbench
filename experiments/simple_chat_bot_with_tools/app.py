import sys
sys.path.append('d:/source/vscode/ai_workbench')

import os
import operator
import functools

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from experiments.helpers.debugging_helpers import display_langgraph_graph
from experiments.helpers.tools.generic_tools import get_current_datetime
from experiments.helpers.tools.web_tools import google_search
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)


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


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)    
    # We convert the agent output into a format that is suitable to append to the global state
    # if isinstance(result, ToolMessage):
    #     pass
    # else:
    #     result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

llm = ChatOpenAI(model="gpt-4-1106-preview")

# Research agent and node
smart_guy_agent = create_agent(
    llm,
    [google_search],
    system_message=("You are a smart guy.")
)
smart_guy_node = functools.partial(agent_node, agent=smart_guy_agent, name="smart guy")


# Define the tools
tools = [google_search]
tool_node = ToolNode(tools)


# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("smart_guy_node", smart_guy_node)
workflow.add_node("call_tool", tool_node)

# Define the edge logic
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "__end__"
    return "continue"

workflow.add_conditional_edges(
    "smart_guy_node",
    router,
    {"call_tool": "call_tool", "continue": "smart_guy_node", "__end__": END},
)


workflow.set_entry_point("smart_guy_node")
graph = workflow.compile()

showGraph = False
if showGraph == True:
    display_langgraph_graph(graph, "", figure_size=(10, 6))



while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content = user_input
                )
            ],
        },
        # Maximum number of steps to take in the graph
        {"recursion_limit": 150},
    )
    for s in events:
        print(s)
        print("----")
    