import sys
import json
sys.path.append('d:/source/vscode/ai_workbench')

from agent_graph import create_graph, AgentState
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv, find_dotenv
from experiments.helpers.debugging_helpers import display_langgraph_graph

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

def main():
    graph = create_graph()
    
    # Optionally display the graph
    showGraph = True
    if showGraph:
        try:
            display_langgraph_graph(graph, "", figure_size=(10, 6))
        except Exception as e:
            print(f"Error displaying graph: {e}")
    

    try:
        user_input = input("Prompt (or 'quit' to exit): ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
    
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
            last_message.pretty_print()

        print("Completed!")
    except Exception as e:
        print(f"Error processing user input: {e}")

if __name__ == "__main__":
    main()