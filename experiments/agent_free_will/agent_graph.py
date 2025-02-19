import sys
import json
import operator
import functools
from typing_extensions import TypedDict, Annotated, Union, Literal, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

# Example of a "tool" that just returns structured conversation output
class BasicResponse(BaseModel):
    """
    Respond in a conversational manner. Be kind and helpful.
    Provide your next_agent to indicate who should respond next.
    """
    response: str = Field(description="The response or idea from this agent.")
    next_agent: str = Field(
        description="The next agent who should receive control. If no further collaboration is needed, use FINISH."
    )

# --- Business Requirements (for reference in system prompts) ---
BUSINESS_REQUIREMENTS = (
    "The business must not be government or military related, must be scalable, "
    "automated, fully agentic, and profitable."
)

# If an agent sets next_agent to this stop_word, the chain ends.
stop_word = "FINISH"

# The five agents
members = [
    "expert_entrepreneur",
    "expert_computer_scientist_and_software_engineer",
    "expert_researcher",
    "ceo_with_mba",
    "expert_system_designer",
]

# We provide a helper to create a "collaboration agent" prompt with the ability to pass control
def create_collab_agent(llm, system_message: str, tools, other_members):
    """
    Create an agent with awareness of other members it can pass control to.
    It can also produce a response in the shape of BasicResponse to indicate
    the next agent or FINISH if no further collaboration is needed.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI agent. Your role: {system_message}\n\n"
                "Business Requirements:\n"
                f"{BUSINESS_REQUIREMENTS}\n"
                "You can pass the conversation control to these collaborators: {members}\n\n"
                "{tool_prompt}"
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "If you need another collaborator's expertise, set `next_agent` to them. "
                "If no further input is needed, set `next_agent` to FINISH."
            ),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(members=", ".join(other_members))

    # Show which tools are available to each agent
    if tools:
        tools_prompt = "You have access to the following tools: " + ", ".join(
            getattr(tool, "name", tool.__name__) if isinstance(tool, type)
            else getattr(tool, "name", tool.__class__.__name__)
            for tool in tools
        )
    else:
        tools_prompt = "No special tools are available."
    prompt = prompt.partial(tool_prompt=tools_prompt)

    # Bind the LLM with the tools for function calling
    if tools:
        return prompt | llm.bind_tools(tools, strict=True)
    return prompt | llm


# The "state" your graph will pass around
class AgentState(TypedDict):
    messages: Annotated[Union[BaseMessage], operator.add]  # conversation so far
    next: str   # the agent set to speak next
    sender: str # the agent who just spoke


# We’ll use an OpenAI-like LLM stub for demonstration. 
# Customize model/temperature as needed.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Here is our single "tool" example. You can add more as needed.
tools = [BasicResponse]
tool_node = ToolNode(tools)


def agent_node(state: AgentState, agent, name) -> AgentState:
    """
    Generic node logic for a single agent that calls the LLM chain.
    If the LLM uses BasicResponse (the 'tool'), we parse the JSON and set next accordingly.
    """
    try:
        # Invoke the chain with the current conversation
        result = agent.invoke(state)

        # If the chain used the BasicResponse tool:
        if isinstance(result, AIMessage) and len(result.tool_calls) > 0:
            tool_call = result.tool_calls[0]
            if tool_call["name"] == "BasicResponse":
                structured_response = tool_call["args"]
                # Attach the agent's name so we know who spoke
                structured_response["agent_name"] = name
                return {
                    "messages": [AIMessage(content=str(structured_response), agent_name=name)],
                    "next": structured_response.get("next_agent", None),
                    "sender": name,
                }
        
        # If no tool calls, treat it as a direct AIMessage
        if isinstance(result, AIMessage):
            return {
                "messages": [AIMessage(**result.dict(exclude={"type", "name"}), agent_name=name)],
                "next": getattr(result, "next", None),
                "sender": name,
            }
        
        raise ValueError(f"Unexpected result type: {type(result)}")
    except Exception as e:
        # If there's an error, append that to messages, try the same agent again
        return {
            "messages": [AIMessage(content=f"Error occurred: {e}", agent_name=name)],
            "next": name,
            "sender": name,
        }


# ---- Define each specialized agent node ----
expert_entrepreneur_node = functools.partial(
    agent_node,
    agent=create_collab_agent(
        llm=llm,
        system_message=(
            "You are an Expert Entrepreneur with deep knowledge of "
            "starting and scaling businesses, focusing on profitable models. "
            "Brainstorm under the given constraints."
        ),
        tools=tools,
        other_members=[m for m in members if m != "expert_entrepreneur"]
    ),
    name="expert_entrepreneur_node"
)

expert_computer_scientist_and_software_engineer_node = functools.partial(
    agent_node,
    agent=create_collab_agent(
        llm=llm,
        system_message=(
            "You are an Expert Computer Scientist & Software Engineer. "
            "Focus on technical feasibility, software architecture, and "
            "automation aspects."
        ),
        tools=tools,
        other_members=[m for m in members if m != "expert_computer_scientist_and_software_engineer"]
    ),
    name="expert_computer_scientist_and_software_engineer_node"
)

expert_researcher_node = functools.partial(
    agent_node,
    agent=create_collab_agent(
        llm=llm,
        system_message=(
            "You are an Expert Researcher adept at market analysis, "
            "data gathering, and identifying gaps/opportunities."
        ),
        tools=tools,
        other_members=[m for m in members if m != "expert_researcher"]
    ),
    name="expert_researcher_node"
)

ceo_with_mba_node = functools.partial(
    agent_node,
    agent=create_collab_agent(
        llm=llm,
        system_message=(
            "You are a CEO with an MBA, skilled at strategic planning, "
            "financial modeling, and scaling organizations."
        ),
        tools=tools,
        other_members=[m for m in members if m != "ceo_with_mba"]
    ),
    name="ceo_with_mba_node"
)

expert_system_designer_node = functools.partial(
    agent_node,
    agent=create_collab_agent(
        llm=llm,
        system_message=(
            "You are an Expert System Designer with experience in designing "
            "large-scale, robust, and efficient automated systems."
        ),
        tools=tools,
        other_members=[m for m in members if m != "expert_system_designer"]
    ),
    name="expert_system_designer_node"
)


# ---- Decide who speaks next by analyzing the 'next' field in state ----
def router(state: AgentState) -> Literal[
    "__end__",
    "expert_entrepreneur_node",
    "expert_computer_scientist_and_software_engineer_node",
    "expert_researcher_node",
    "ceo_with_mba_node",
    "expert_system_designer_node",
    "tool_node"
]:
    """
    If an AIMessage calls a tool, we route to 'tool_node'.
    Else we check 'next' in the state. If it's FINISH, we end.
    Otherwise, route to the appropriate agent node.
    """
    try:
        last_message = state["messages"][-1]

        # If a tool was called, go handle the tool
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"

        # Next agent logic
        nxt = state["next"]
        if nxt == stop_word:
            return "__end__"
        elif nxt == "expert_entrepreneur":
            return "expert_entrepreneur_node"
        elif nxt == "expert_computer_scientist_and_software_engineer":
            return "expert_computer_scientist_and_software_engineer_node"
        elif nxt == "expert_researcher":
            return "expert_researcher_node"
        elif nxt == "ceo_with_mba":
            return "ceo_with_mba_node"
        elif nxt == "expert_system_designer":
            return "expert_system_designer_node"
        else:
            # If unrecognized, or empty, just end
            return "__end__"
    except KeyError as e:
        print(f"Router error: {e}")
        return "__end__"


def create_graph() -> StateGraph:
    """
    Build and compile the StateGraph for these five agents and the tool node.
    """
    graph = StateGraph(AgentState)

    # ---- Register agent nodes ----
    graph.add_node("expert_entrepreneur_node", expert_entrepreneur_node)
    graph.add_node("expert_computer_scientist_and_software_engineer_node", expert_computer_scientist_and_software_engineer_node)
    graph.add_node("expert_researcher_node", expert_researcher_node)
    graph.add_node("ceo_with_mba_node", ceo_with_mba_node)
    graph.add_node("expert_system_designer_node", expert_system_designer_node)

    # Tool node (no chain of tool -> tool)
    graph.add_node("tool_node", tool_node)

    # ---- Create a common conditional map for each agent -> router -> next node ----
    conditional_map = {
        "expert_entrepreneur_node": "expert_entrepreneur_node",
        "expert_computer_scientist_and_software_engineer_node": "expert_computer_scientist_and_software_engineer_node",
        "expert_researcher_node": "expert_researcher_node",
        "ceo_with_mba_node": "ceo_with_mba_node",
        "expert_system_designer_node": "expert_system_designer_node",
        "tool_node": "tool_node",
        "__end__": END,
    }

    # Each agent’s node uses the router to figure out where to go next
    graph.add_conditional_edges("expert_entrepreneur_node", router, conditional_map)
    graph.add_conditional_edges("expert_computer_scientist_and_software_engineer_node", router, conditional_map)
    graph.add_conditional_edges("expert_researcher_node", router, conditional_map)
    graph.add_conditional_edges("ceo_with_mba_node", router, conditional_map)
    graph.add_conditional_edges("expert_system_designer_node", router, conditional_map)

    # The tool node always goes back to whoever invoked the tool (i.e., state["sender"] + "_node")
    # (no direct tool_node -> tool_node transition)
    def tool_return(state: AgentState) -> str:
        return state["sender"] + "_node"

    graph.add_conditional_edges("tool_node", tool_return, conditional_map)

    # START the conversation at the first agent you wish. For example:
    graph.add_edge(START, "expert_entrepreneur_node")

    # Compile the graph so it’s ready to be used
    return graph.compile()
