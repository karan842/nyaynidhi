import os 
import json 

from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.tools import tool, Tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain import hub 
from qdrant_client import QdrantClient
from src.agents_utils.retriever import vector_store
from src.utils import load_prompt_template

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Define Judicial Agent
def judicial_agent(query, section_type, chat_history, rerank=False, top_k=5):
    
    # Load config 
    config_file = "./artifacts/config.json"
    with open(config_file, "r") as file:
        config = json.load(file)

    # is it to the cloud?
    to_cloud = config['IS_QDRANT_TO_CLOUD']
    # Collection name
    collection_name = config['COLLECTION_NAME']['ipc-800'] if section_type=="IPC" else config['COLLECTION_NAME']['bns-001']

    # Models
    llm = config['LLAMA']
    embedding_model = config['EMBEDDING_MODEL']
    
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=gemini_api_key)
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=llm,
        temperature=0.2
    )

    # Define vector retriever
    vector_db = vector_store(embeddings, collection_name, to_cloud=to_cloud)
    retriever = vector_db.as_retriever(top_k=top_k)
    
    if rerank:
        # ReRank compressor
        compressor = FlashrankRerank()
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

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
        Queries the legal laws data of differnt judicial sections and retrieves information from its contents. 
        Returns the result and the source documents.

        Args:
            query (string): query derived from the criminal description asked by the user.
        """
        result = qa.invoke(query)
        return result['result'], result['source_documents']

    # Template
    if section_type=="BNS":
        template = load_prompt_template("./prompt_templates/bns_prompts.txt")
    elif section_type=="IPC":
        template = load_prompt_template("./prompt_templates/ipc_section_prompt.txt")

    # Define components such as prompts, agent and executor
    prompt_template = PromptTemplate.from_template(template=template)
    agentprompt = hub.pull("hwchase17/react-chat")
    tools = [query_legal_data, TavilySearchResults(max_results=3)]
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=agentprompt
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=False
    )
    
    response = agent_executor.invoke({
        "input": prompt_template.format(input=query),
        "chat_history": chat_history
    })
    return response
