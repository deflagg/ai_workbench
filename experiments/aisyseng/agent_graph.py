import sys
import json
import operator
import functools
from typing_extensions import TypedDict, Annotated, List, Literal, Union, Optional, Dict, Any
import prompts as prompts

from pydantic import BaseModel, Field
#from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
)
from langchain_openai import ChatOpenAI

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
#from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import utility functions
# from experiments.helpers.debugging_helpers import display_langgraph_graph
# from experiments.helpers.tools.audio_tools import tts_whisper
# from experiments.helpers.tools.code_automation import python_repl
# from experiments.helpers.tools.davinci_tools import create_resolve_project, add_audio_track
from experiments.helpers.tools.generic_tools import get_current_datetime, do_nothing

stop_word = "FINISH"
# members = ["architect", "alice", "mark", "susan"]
members = ["analyst", "architect"]
options = [stop_word] + members

class BasicResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""
    response: str = Field(description="The response of the agent.", example="Hello, how are you?")
    next_agent: str = Field(description="The agent this response is meant for.", example="david")

class Task(BaseModel):
    """ Task in the plan."""
    description: Optional[str] = Field(default=None, description="Description of the task to be completed.")
    role: Optional[str] = Field(default=None, description="Agent responsible for this task.")
    

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    #messages: Annotated[List[BaseMessage], operator.add]
    next: str
    sender: str

def create_agent(llm, system_message: str, tools):
    """Create a generic agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_message} "
                "{tool_prompt} "
                ,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    
    tools_prompt = "You have access to the following tools: "

    # Get the names of the tools/function and the name of the response types/Class
    tools_prompt = tools_prompt + ", ".join([
    getattr(tool, 'name', tool.__name__) if isinstance(tool, type) else getattr(tool, 'name', tool.__class__.__name__)
    for tool in tools])
    
    prompt = prompt.partial(tool_prompt=tools_prompt)
    
    if tools:
        return prompt | llm.bind_tools(tools, strict=True, response_format=BasicResponse)
    
    return prompt | llm.bind_tools([do_nothing], strict=True, response_format=BasicResponse)

def create_collab_agent(llm, system_message: str, tools, members):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI agent, collaborating with other agents: {members} "
                "\n{system_message} "
                "{tool_prompt} "
                ,
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "If you are unable to complete the objective, another agent with different resources can help you. "
                "If further action is required to achieve the objective, set the next agent to the appropriate agent: {members} "
                "If you are not the analyst and no further action is needed to achieve the objective, set the next agent to analyst. "
                "If you are the analyst and no further action is needed to achieve the objective, deliver the deliverables to the end user and set the next agent to {stop_word}. "
                ,
            )
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(stop_word=stop_word)
    prompt = prompt.partial(options=str(options), members=", ".join([member for member in members]))
    
    tools_prompt = "Use the provided tools to progress towards completing the task, " \
                   "if the tools will help with the task. You have access to the following tools: "

    # Get the names of the tools/function and the name of the response types/Class
    tools_prompt = tools_prompt + ", ".join([
    getattr(tool, 'name', tool.__name__) if isinstance(tool, type) else getattr(tool, 'name', tool.__class__.__name__)
    for tool in tools])
    
    prompt = prompt.partial(tool_prompt=tools_prompt)
    
    if tools:
        return prompt | llm.bind_tools(tools, strict=True)
    
    return prompt | llm
   
class GenericResponse(BaseModel):
    """Generic Reponse."""
    responseadf23423: str = Field(description="Response")
    custom_field: str = Field(description="Custom field")
 
# Initialize the LLM model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatOpenAI(model="gpt-4o", temperature=0, )
# llm = llm.bind_tools([get_current_datetime], strict=True, response_format=GenericResponse)
# llm = llm.with_structured_output(GenericResponse, strict=True, method="json_schema")
# test = llm.invoke("what is the color of an a white rabbit?")
# test = llm.invoke("what time is it? You have access to the following tools: get_current_datetime.")


# Define the tools
tools = [BasicResponse, get_current_datetime]
tool_node = ToolNode(tools)

def agent_node(state: AgentState, agent, name) -> AgentState:
    """Create a node for a given agent."""
    try:
        # Include all previous messages in the agent's context
        result = agent.invoke(state)

        # if (isinstance(result, AIMessage) and 
        #     len(result.tool_calls) > 0 and 
        #     result.tool_calls[0]["name"] in ["BasicResponse"]):
            
        #     structured_response = result.tool_calls[0]["args"]
        #     structured_response["agent_name"] = name
        
        #     return {
        #         "messages": [AIMessage(content=str(structured_response), agent_name=name)],
        #         "next": structured_response.get("next_agent", None),
        #         "sender": name,
        #     }
        if isinstance(result, ToolMessage):
            pass
        elif isinstance(result, AIMessage):
            structured_response = result.additional_kwargs.get("parsed")
            
            return {
                "messages": [AIMessage(**result.model_dump(exclude={"type", "name"}), agent_name=name)],
                "next": structured_response.next_agent,
                "sender": name,
            }
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error in agent_node method (agent name: {name}): {e}", agent_name=name)],
            "next_agent": name,
            "sender": name,
        }

# Define agent nodes
analyst_node = functools.partial(
    agent_node,
    agent = create_agent(
        llm,
        #system_message=prompts.analyst_prompt,
        system_message="You are an analyst. You are part of a team of agents: [user, architect].",
        tools=[],
    ),
    name = "analyst_node"
)


architect_node = functools.partial(
    agent_node,
    agent = create_agent(
        llm,
        # system_message=prompts.architect_prompt,
        system_message="You are an architect. You are part of a team of agents: [analyst].",
        tools=[get_current_datetime],
    ),
    name = "architect_node"
)

# def human_node(state: AgentState):
#     try:
#         user_input = input("Prompt (or 'quit' to exit): ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")

#         return {
#                 "messages": [HumanMessage(content=user_input)],
#                 "next": structured_response.get("next_agent", None),
#                 "sender": name,
#             }
#     except:
#         user_input = "go to step 3!"

alice_node = functools.partial(
    agent_node,
    agent = create_collab_agent(
        llm,
        system_message="You a are a woman named Alice.",
        tools=[get_current_datetime],
        members=[member for member in members if member != "alice"]
    ),
    name = "alice_node"
)

mark_node = functools.partial(
    agent_node,
    agent = create_collab_agent(
        llm,
        system_message="You are a non-binary named Mark.",
        tools=[get_current_datetime],
        members=[member for member in members if member != "mark_node"]
    ),
    name = "mark_node"
)

susan_node = functools.partial(
    agent_node,
    agent = create_collab_agent(
        llm,
        system_message="You are a code tester. You write test scripts and execute them to ensure the code works as expected. Ensure to include the script to be tested.",
        tools=[get_current_datetime],
        members=[member for member in members if member != "susan_node"]
    ),
    name = "susan_node"
)

# Define the edge logic
def router(state):
    """Router function to determine next steps."""
    try:
        messages = state["messages"]
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"
        if state["next"] == stop_word:
            return "__end__"
        if state["next"] in members:
             return state["next"] + "_node"
        return "__end__"
    
    except KeyError as e:
        print(f"Error in router function: {e}")
        return "__end__"
    
        
def create_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    # graph.add_node("analyst_node", analyst_node)
    # # graph.add_node("human_node", human_node)
    # graph.add_node("tool_node", tool_node)
    
    # graph.add_edge(START, "analyst_node")
    # graph.add_conditional_edges(
    #     "analyst_node",
    #     router,
    # )
    # graph.add_edge("tool_node", "analyst_node")
   
    
    for member in members:
        node_name = member + "_node"
        if node_name in globals():
            graph.add_node(node_name, globals()[node_name])
        else:
            raise ValueError(f"{node_name} not found in globals")
    
    graph.add_node("tool_node", tool_node)
    
    # Add conditional edges
    for member in members:
        member_node = member + "_node"
        
        # Create conditional map
        conditional_map = {k + "_node": k + "_node" for k in members if k != member}
        # conditional_map = {k + "_node": k + "_node" for k in members}
        conditional_map["__end__"] = END
        conditional_map["tool_node"] = "tool_node"

        
        if member_node in globals():
            graph.add_conditional_edges(member_node, router, conditional_map)
        else:
            raise ValueError(f"{member_node} not found in globals")
    
    conditional_map = {k + "_node": k + "_node" for k in members}   
    graph.add_conditional_edges("tool_node", lambda x: x["sender"], conditional_map)
    
    
    # start with first member in the list
    graph.add_edge(START, members[0] + "_node")
    

    
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
