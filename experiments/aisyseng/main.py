import sys
import json
import datetime

sys.path.append('d:/source/vscode/ai_workbench')

from agent_graph import create_graph, AgentState
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
from experiments.helpers.debugging_helpers import display_langgraph_graph

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

import json
import datetime

def pretty_print_json(json_string, agent_name):
    """
    This function takes a JSON string and an agent name as input, 
    formats the JSON, prints a header with a timestamp and the 
    agent name, and then prints the formatted JSON to the console.

    Args:
      json_string: The JSON string to be formatted and printed.
      agent_name: The name of the agent to be included in the header.
    """
    try:
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print the header with the timestamp and agent name
        print(f"********************J.A.R.V.I.S ({timestamp}) - Agent: {agent_name}**********************")  
        data = json.loads(json_string)
        formatted_json = json.dumps(data, indent=4)
        print()
        print(formatted_json)
        print()
    except json.JSONDecodeError:
        print("Invalid JSON string provided.")


def main():
    graph = create_graph()
    
    # Optionally display the graph
    showGraph = False
    if showGraph:
        try:
            display_langgraph_graph(graph, "", figure_size=(10, 6))
        except Exception as e:
            print(f"Error displaying graph: {e}")
    

    try:
        while True:
            user_input = input("Prompt (or 'quit' to exit): ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
        
            initial_state: AgentState = {
                "messages": [HumanMessage(content=user_input)],
                "next": "analyst",
                "sender": "user",
            }
            config = {"configurable": {"thread_id": "2"}}
            for step in graph.stream(initial_state, config):
                node_name = list(step.keys())[0]
                state = step[node_name]

                last_message = state["messages"][-1]
                pretty_print_json(last_message.content, node_name)

    except Exception as e:
        print(f"Error processing user input: {e}")

if __name__ == "__main__":
    main()