import sys
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
    
    # Main loop to interact with the user
    while True:
        user_input = input("Enter a YouTube video topic (or 'quit' to exit): ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        try:
            initial_state: AgentState = {
                "messages": [HumanMessage(content=f"Create a YouTube video about {user_input}")],
                "sender": "human",
                "script": "",
                "voiceover_path": "",
                "project_path": ""
            }
            
            for step in graph.stream(initial_state):
                node_name = list(step.keys())[0]
                state = step[node_name]
                
                print(f"----\nCompleted step: {node_name}")
                print(f"Sender: {state['sender']}")
                print(f"Latest message: {state['messages'][-1].content}")
                
                if 'script' in state:
                    print(f"Script: {state['script']}")
                if 'voiceover_path' in state:
                    print(f"Voiceover Path: {state['voiceover_path']}")
                if 'project_path' in state:
                    print(f"Project Path: {state['project_path']}")
                

            print("YouTube video creation process completed!")
        except Exception as e:
            print(f"Error processing user input: {e}")

if __name__ == "__main__":
    main()