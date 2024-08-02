import sys
sys.path.append('d:/source/vscode/ai_workbench')

import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

def create_translate_chain():
    model = ChatOpenAI(model="gpt-3.5-turbo")

    system_template = "Translate the following into {language}:"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    parser = StrOutputParser()
    chain = prompt_template | model | parser
    return chain

if __name__ == "__main__":
    chain = create_translate_chain()
    res = chain.invoke({"language": "italian", "text": "what's up?"})
    print(res)