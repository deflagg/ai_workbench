import sys
sys.path.append('d:/source/vscode/ai_workbench')

import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings()
)



# docs = vectorstore.similarity_search("cat")
# print(docs)

# docs = vectorstore.similarity_search_with_score("cat", 3)
# print(docs)

# embedding = OpenAIEmbeddings().embed_query("cat")
# vectorstore.similarity_search_by_vector(embedding)
# print(docs)


model = ChatOpenAI(model="gpt-3.5-turbo")

vector_store_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": vector_store_retriever, "question": RunnablePassthrough()} | prompt | model

response = rag_chain.invoke("tell me about parrots") # only takes one paramter. parsing the paraters is done in the chain

print(response.content)