import sys
sys.path.append('d:/source/vscode/ai_workbench')

import os
import operator
import functools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from experiments.helpers.debugging_helpers import display_langgraph_graph
from experiments.helpers.tools.github_tools import create_file, get_repo, python_repl
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
                " will help where you left off. Execute what you can to make progress. Be concise."
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


tavily_tool = TavilySearchResults(max_results=5)


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

llm = ChatOpenAI(model="gpt-4-1106-preview")

# Research agent and node
# software_engineer_agent = create_agent(
#     llm,
#     [get_repo, create_file, tavily_tool],
#     system_message=("You are a software engineer working on a project.",
#                     "Use the provided tools to progress towards answering the question.",
#                     "If you are unable to fully answer, that's OK, another assistant with",
#                     "different tools will help where you left off.",
#                     "Execute what you can to make progress.",
#                     "If you or any of the other assistants have the final answer or deliverable,",
#                     "prefix your response with FINAL ANSWER so the team knows to stop.")
# )
software_engineer_agent = create_agent(
    llm,
    [get_repo, create_file, tavily_tool],
    system_message=("You are a good assistant.")
)
software_engineer_node = functools.partial(agent_node, agent=software_engineer_agent, name="Software Engineer")

# chart_generator
chart_agent = create_agent(
    llm,
    [python_repl],
    system_message="Run the python code to display the chart."
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="chart_generator")

# Define the tools
tools = [get_repo, create_file, tavily_tool, python_repl]
tool_node = ToolNode(tools)


# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("Software Engineer", software_engineer_node)
#workflow.add_node("chart_generator", chart_node)
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
    "Software Engineer",
    router,
    {"call_tool": "call_tool", "__end__": END},
)

# workflow.add_conditional_edges(
#     "chart_generator",
#     router,
#     {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
# )

# workflow.add_conditional_edges(
#     "call_tool",
#     # Each agent node updates the 'sender' field
#     # the tool calling node does not, meaning
#     # this edge will route back to the original agent
#     # who invoked the tool
#     lambda x: x["sender"],
#     {
#         "Researcher": "Researcher",
#         "chart_generator": "chart_generator",
#     },
# )
workflow.set_entry_point("Software Engineer")
graph = workflow.compile()

# Display the graph for debugging purposes
#graph.get_graph(xray=True).draw_mermaid_png(output_file_path="./pile_of_shit/bug_killer/graph_visualization.png")

# Load the generated PNG image
# img = mpimg.imread('./pile_of_shit/bug_killer/graph_visualization.png')

# # Display the image
# plt.imshow(img)
# plt.axis("off")  # Turn off axes for a cleaner look
# plt.show()

# Stream the graph to see the conversation
#content="write a python script to generate numbers from 1 to 100"
events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="what tools are available for the software engineer"
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 150},
)
for s in events:
    print(s)
    print("----")
    
