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

# os.environ["LANGCHAIN_API_KEY"] = os.getenv["LANGCHAIN_API_KEY"]
# os.environ["LANGCHAIN_TRACING_V2"] = "true"


def chat_ui():

    st.set_page_config(page_title='NyayNidhi', page_icon='⚖️', layout='wide')
    st.markdown(
    """
    <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
           display: none;
         }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.title("NyayNidhi.AI")
    st.markdown("*v0.0.1 beta*  [Feedback Form](https://docs.google.com/forms/d/e/1FAIpQLSfrbTdAxQ3zQvH2M5k-HD9huWZht7rp51sS6_AYBS1h1oaGIQ/viewform?usp=sf_link)")
    st.warning("NyayNidhi can make mistakes. Verify responses before any action.")
    
    # Streamlit Sidebar with Example Queries
    st.sidebar.title("Example Queries to Try")

    queries = [
        "How can a woman travel in India alone?",
        "A 22-year-old girl was raped by men in Kolkata RG Kar Medical College.",
        "What are the laws of foriegn direct investment in Stock market?",
        "A random guy was staring at me in the metro and attempted to take my photo.",
        "Which cities in India has most rapse cases?",
        "Hey! Someone's dog bite me."
    ]

    for query in queries:
        st.sidebar.write(f"• {query}")
        
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
            
            response = get_response(query, st.session_state.chat_history)
            
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
