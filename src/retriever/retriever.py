import os 
import json 
import qdrant_client
import warnings 
import time 
import requests
import re

from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

from qdrant_client import QdrantClient 
from langchain_core.output_parsers.string import StrOutputParser
from langchain.chains import create_retrieval_chain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Qdrant 
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Collection name
collection_name = "ipc-data-001"

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", 
                                          google_api_key=gemini_api_key)

llm = ChatGroq(groq_api_key=groq_api_key,
               model="llama3-8b-8192",
               temperature=0.2)

## Connect to client
client = QdrantClient(url="http://localhost:6333/")

def vector_store():
    # Define vector store for vector retriever
    vector_store = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    return vector_store

# Define vector_db
vector_db = vector_store()
retriever = vector_db.as_retriever(top_k=5)

# Define system-prompt
system_prompt = (
    """You are legal advisor assistant for question-answering tasks.
       Use the followinf piece of retrieved context to answer the following query.
       Don't answer without referring the context.
       Find Indian Penal Code (IPC) Section from the context 
       
       Provide information in below format:
       1. Sections: 
           Describe all sections from context
       2. Punishment: Describe punishment 
       3. Legal Advice: Suggest Legal Advice
       
       Context:
       {context}
       
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
parser = StrOutputParser()

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

query = input("Ask a query: ")

def result(query):
    # response = rag_chain.invoke({"input": query})
    for response in rag_chain.stream({"input": query}):
        print(response, end="")

if __name__ == '__main__':
    result(query)
    

# print(response['answer'])

# for token in response['answer']:
#     print(token, end="")



# retriever = vector_db.as_retriever()

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# #  System Prompt
# contexualize_q_system_prompt = """ 
#     Given a chat history and the latest user question \
#     which might reference context in the chat history, formulate a standalone question \
#     which can be understood without the chat history, Do Not answer the question, \
#     just reformulate it if needed and otherwise return it as it.
# """

# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contexualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ]
# )

# # Hisotry aware retiever
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# qa_system_prompt = """You are legal advisor chatbot for question-answering taks. \
#         Using the following pieces of retrieved context related to Indial Penal Code (IPC) section information for related question asker. \
#         Keep your answer clean and provide each IPC section:
        
#         Example:
#             1. Section A - 
#              Description: Provide description here
#              Punishment: This is the Punishment 
            
#             Legal action to take: Take help from police or elder persons in law.
# CONTEXT:
# {context}
# """

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}")
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# if __name__ == '__main__':
#     # chat_history = []
#     # i =0 
    
#     # while i < 3:
#     #     question = input("Ask a Query: ")
#     #     response = rag_chain.invoke({"input": question, "chat_history": chat_history})
#     #     chat_history.extend([HumanMessage(content=question), response["answer"]])
#     #     print(response["answer"])
#     #     print("\n\n")
#     #     i+=1
#     query = input("Ask a query: ")
#     results = vector_db.similarity_search(query, k=4)
#     print(type(results))
    
#     for i in range(len(results)):
#         print(results[i].page_content)
#         print(f"Section: {results[i].metadata}")
#         print("\n\n")
    


