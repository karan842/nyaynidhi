import streamlit as st 
import os 
import json
from src.retriever import memory_chain
from src.agents import ipc_agent
from dotenv import load_dotenv
from qdrant_client import QdrantClient 

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')
qdrant_url = os.getenv('QDRANT_URI')
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Load configuration
config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)

# Qdrant Client
client = QdrantClient(url=qdrant_url)

# Collection name
collection_name = config['COLLECTION_NAME']

# Models
llm = config['LLAMA']
embedding_model = config['EMBEDDING_MODEL']

def chat_ui():
    st.title("Nyaynidhi-IPC Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def process_query(query):
        if query.lower() == 'exit':
            st.session_state.chat_history = []
        else:
            response = ipc_agent(query, st.session_state.chat_history, embedding_model, llm, client, collection_name)
            st.session_state.chat_history.extend({"human":query, "ai":response["output"]})
            return response["output"]

    if prompt := st.chat_input("Ask a query: "):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == '__main__':
    chat_ui()