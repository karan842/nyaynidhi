import os 
import json 

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.tools import tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain import hub 
from langchain_chroma import Chroma
from qdrant_client import QdrantClient
from src.retriever import vector_store

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Load config 
config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)

# Collection name
collection_name = config['COLLECTION_NAME']

## Connect to Qdrant client
client_url = QdrantClient(url="http://localhost:6333/")

# Models
llm = config['LLAMA']
embedding_model = config['EMBEDDING_MODEL']

def ipc_agent(query, chat_history, embedding_model, llm, client, collection_name):

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=gemini_api_key)

    llm = ChatGroq(groq_api_key=groq_api_key,
                    model=llm,
                    temperature=0.2)

    # Define vector retriever
    vector_db = vector_store(embeddings, client_url, collection_name)
    retriever = vector_db.as_retriever(top_k=5)

    # Define RetrievalQA
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    @tool 
    def query_legal_data(query):
        """ 
        Queries the legal laws data of IPC sections and retrieves information from its contents. 
        Returns the result and the source documents.

        Args:
            query (string):  query derived from the criminal description asked by the user.
        """
        result = qa.invoke(query)
        return result['result'], result['source_documents']

    # Template
    template = """
    You are legal assistand AI chatbot that answers questions on the retriever provided.
    Use the tools to respond accurately
    The query_legal_data tool should be used to retreiver information from IPC Data.

    Don't answer without referring the context.
    Find Indian Penal Code (IPC) Section from the context. 
    Focus on entities present in the information. Only add relevant information.

    **USE BEAUTIFUL MARKDOWN FORMAT and ANSWER like chatbot.** 

    Provide information in below format:
    A. Sections:
            List all sections in below format:
            Section (section number or name): Description \n

    B. Punishments:
            Define Punishments in detail for associate section name

    C. Legal Advice:
        Define legal advice including police and medical if emergency.

    If question is not about criminal description and general follow-up question dont include above format 
    and just answer that general question. 
    for questions that require further information, use tavily_search_tool_json tool to conduct research and respond 
    with accurate answer.

    Question: {input}
    """

    # Define components such as prompts, agent and executor
    prompt_template = PromptTemplate.from_template(template=template)

    agentprompt = hub.pull("hwchase17/react-chat")

    tools = [query_legal_data, TavilySearchResults(max_results=3)]

    agent = create_react_agent(llm=llm,
                            tools=tools,
                            prompt=agentprompt)

    agent_executor = AgentExecutor(agent=agent,
                                tools=tools,
                                handle_parsing_errors=True,
                                verbose=True)
    
    response = agent_executor.invoke({"input": prompt_template.format(input=query), "chat_history": chat_history})
    return response

# if __name__ == '__main__':
#     chat_history = []
#     i = 0
    
#     while i < 3:
#         query = input("Ask a query: ")
#         response = ipc_agent(query, chat_history, embedding_model, llm, client_url, collection_name)
#         chat_history.extend({"human":query, "ai":response["output"]})
#         print(response["output"])
#         print("\n\n")
#         i += 1


