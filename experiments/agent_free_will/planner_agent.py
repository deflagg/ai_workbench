import sys
import json
sys.path.append('d:/source/vscode/ai_workbench')

from dotenv import load_dotenv, find_dotenv
from typing import Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
import planner_prompts as prompts
from langchain import hub

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

select_prompt = hub.pull("hwchase17/self-discovery-select")
adapt_prompt = hub.pull("hwchase17/self-discovery-adapt")
structured_prompt = hub.pull("hwchase17/self-discovery-structure")
reasoning_prompt = hub.pull("hwchase17/self-discovery-reasoning")

class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]
    

model = ChatOpenAI(temperature=0, model="gpt-4o-mini")


def select(inputs):
    select_chain = select_prompt | model | StrOutputParser()
    return {"selected_modules": select_chain.invoke(inputs)}


def adapt(inputs):
    adapt_chain = adapt_prompt | model | StrOutputParser()
    return {"adapted_modules": adapt_chain.invoke(inputs)}


def structure(inputs):
    structure_chain = structured_prompt | model | StrOutputParser()
    return {"reasoning_structure": structure_chain.invoke(inputs)}


def reason(inputs):
    reasoning_chain = reasoning_prompt | model | StrOutputParser()
    return {"answer": reasoning_chain.invoke(inputs)}


graph = StateGraph(SelfDiscoverState)
graph.add_node(select)
graph.add_node(adapt)
graph.add_node(structure)
graph.add_node(reason)
graph.add_edge(START, "select")
graph.add_edge("select", "adapt")
graph.add_edge("adapt", "structure")
graph.add_edge("structure", "reason")
graph.add_edge("reason", END)
app = graph.compile()



def main():

    task_example = "Lisa has 10 apples. She gives 3 apples to her friend and then buys 5 more apples from the store. How many apples does Lisa have now?"

    # task_example = """This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
    # 45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
    # (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle"""

    reasoning_modules_str = "\n".join(prompts.reasoning_modules)

    for s in app.stream(
        {"task_description": task_example, "reasoning_modules": reasoning_modules_str}
    ):
        print(s)


if __name__ == "__main__":
    main()