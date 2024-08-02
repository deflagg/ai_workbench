import sys
sys.path.append('d:/source/vscode/ai_workbench')

from agent_graph import create_graph
from langchain_core.messages import HumanMessage
from experiments.helpers.debugging_helpers import display_langgraph_graph
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

def main():
    graph = create_graph()
    
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

if __name__ == "__main__":
    main()