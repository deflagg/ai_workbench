import sys
import json
import os
from dotenv import load_dotenv, find_dotenv

# Append the custom path to sys.path if not already present
custom_path = 'd:/source/vscode/ai_workbench'
if custom_path not in sys.path:
    sys.path.append(custom_path)

try:
    from agent_graph import create_graph, AgentState
    # Updated import path based on LangChain's latest structure
    from langchain.schema import HumanMessage, AIMessage, ToolMessage
    from experiments.helpers.debugging_helpers import display_langgraph_graph
except ImportError as ie:
    print(f"Import Error: {ie}")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

def main():
    try:
        graph = create_graph()
    except Exception as e:
        print(f"Error creating graph: {e}")
        sys.exit(1)
    
    # Optionally display the graph
    show_graph = True
    if show_graph:
        try:
            display_langgraph_graph(graph, "", figure_size=(10, 6))
        except Exception as e:
            print(f"Error displaying graph: {e}")
    
    # Main loop to interact with the user
    while True:
        try:
            user_input = input("Prompt (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a valid prompt.")
            continue

        try:
            # Ensure 'next' and 'sender' are actual string values
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                next_node="initial_node",  # Replace with appropriate initial node name
                sender="user"              # Replace with appropriate sender identifier
            )
        except Exception as e:
            print(f"Error initializing AgentState: {e}")
            continue
        
        try:
            for step in graph.stream(initial_state):
                if not step:
                    print("Received an empty step from the graph.")
                    continue

                node_name, state = next(iter(step.items()))
                messages = state.get("messages", [])
                
                if not messages:
                    print(f"No messages found for agent '{node_name}'.")
                    continue

                last_message = messages[-1]
                print("*" * 50)
                print(f"Agent Name: {node_name}\n")
                
                if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_name = tool_call.get('name', 'Unnamed Tool')
                        tool_args = tool_call.get('args', {})
                        print(f"Tool Call: {tool_name}")
                        print(f"Arguments: {json.dumps(tool_args, indent=2)}\n")
                elif isinstance(last_message, AIMessage):
                    print(f"Response:\n{last_message.content}")
                else:
                    print(f"Message Type Not Handled: {type(last_message)}")
                
            print("*" * 50)
        except Exception as e:
            print(f"Error processing the graph stream: {e}")

if __name__ == "__main__":
    main()
