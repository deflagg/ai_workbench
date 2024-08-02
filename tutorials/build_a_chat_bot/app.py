import sys
sys.path.append('d:/source/vscode/ai_workbench')

import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def create_translate_chain():
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    system_template = "You are a helpful assistant. Answer all questions to the best of your ability in {language}:"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), MessagesPlaceholder(variable_name="messages")]
    )

    parser = StrOutputParser()
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer) |
        prompt_template |
        model |
        parser
    )
    
    return chain

if __name__ == "__main__":
    runnable = create_translate_chain()
    
    runnable_with_history  = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="messages"
    )
    
    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]
    
    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm todd. tell me a joke")
    ]
    
    # Only 65 tokens are allowed in the prompt
    # It will forget the user's name is Bob but remember the math problem.  
    
    config = {"configurable": {"session_id": "2"}}
    
    for r in runnable_with_history.stream(
        {
            "messages": messages,
            "language": "English"
        },
        config=config
    ):
        print(r, end="|") # the end parameter is used to avoid new lines and show where the messages are separated in chunks
    
