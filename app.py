import streamlit as st 
import os 
import json
from src.retriever.retriever import memory_chain
from dotenv import load_dotenv
from qdrant_client import QdrantClient 

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')
qdrant_url = os.getenv('QDRANT_URI')
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

# Load configuraion
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


def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input('Ask a query (type "exit" to quit): ')

    # Process query
    if query:
        if query.lower() == 'exit':
            st.session_state.chat_history = []
        else:
            # Call memory_chain function (replace with your actual function)
            response = memory_chain(query, st.session_state.chat_history, embedding_model, llm, client, collection_name)
            st.session_state.chat_history.extend([query, response["answer"]])
            st.markdown(response["answer"])
            
if __name__ == '__main__':
    main()