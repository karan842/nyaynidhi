import streamlit as st 
import os 
import json
from src.generate_response import get_response
from dotenv import load_dotenv
from qdrant_client import QdrantClient 

load_dotenv()
gemini_api_key = st.secrets["GEMINI_API_KEY"]
groq_api_key = st.secrets['GROQ_API_KEY']
qdrant_url = st.secrets['QDRANT_URI']
qdrant_cloud_url = st.secrets["QDRANT_CLOUD_URL"]
qdrant_cloud_api_key = st.secrets["QDRANT_CLOUD_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def chat_ui():

    st.set_page_config(page_title='NyayNidhi', page_icon='⚖️', layout='wide')
    st.title("NyayNidhi.AI")
    st.markdown("*v0.0.1 beta*")
    st.warning("NyayNidhi can make mistakes. Verify responses before any action.")

    # Add a sidebar with radio buttons
    # section_type = st.sidebar.selectbox("Select Data", ["BNS"], placeholder="Select Criminal Code")
    
    try:
        # Print the selected data
        # st.toast(body=f"Selection Section: {section_option}", icon="✅")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        def process_query(query):
            st.session_state.chat_history = []
            
            response = get_response(query, "BNS", st.session_state.chat_history)
            
            st.session_state.chat_history.extend({"human":query, "ai":response['output']})
            return response['output']

        if prompt := st.chat_input("Ask a query: "):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = process_query(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        
    except Exception as e:
        st.error("Something went wrong! NyayNidhi can't work at this time.")
        print(e)

if __name__ == '__main__':
    chat_ui()