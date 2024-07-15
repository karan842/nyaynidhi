import os 
import sys 
import re 
import json 
import time
import requests
import warnings
import os 
import re
from dotenv import load_dotenv

import qdrant_client
from qdrant_client import QdrantClient

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.document_loaders import JSONLoader 
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_chroma import Chroma

# Configuration
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)

# Collection name
collection_name = 'ipc-data-001'

# IPC data file path
ipc_data_path = './data/ipc_data.json'

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Store in vector database
def ingest_in_vectordb(text_chunk, embedding_model):
    print("\nConnecting Qdrant DB")
    # Connect to the QdrantDB local
    client = QdrantClient(url="http://localhost:6333/")
    
    print("\nSetting Vectors Configuration")
    vectors_config = qdrant_client.http.models.VectorParams(
        size=768,
        distance=qdrant_client.http.models.Distance.COSINE
    )
    
    # Creating collection
    print("\nCreating collection")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    
    # Save in QdrantDB
    print("\nSaving into QdrantDB")
    qdrant = Qdrant.from_documents(
        text_chunk,
        embedding_model,
        url="http://localhost:6333/",
        # api_key=qdrant_api_key,
        collection_name=collection_name
    )
    print("\n> Chunk of text saved in DB with Embedding format.")

    
# Ingest data from PDF format
def pdf_loader():                                     
    loader = DirectoryLoader("./data/", glob="**/*.pdf",
                        show_progress=True, loader_cls=PyPDFLoader)
    print("\nSplitting PDF documents")
    documents = loader.load_and_split(text_splitter=splitter)
    return documents


def json_loader():
    with open(ipc_data_path, 'r') as json_file:
        ipc_data = json.load(json_file)

    # Create a Langchain document for each JSON record 
    documents = []
    for record in ipc_data:
        document = Document(
            page_content=record['content'],
            metadata={'section':record['section']}
        )
        documents.append(document)
    
    docs = text_splitter.split_documents(documents)
    return docs


## Ingest data from JSON data
if __name__ == '__main__':
    json_docs = json_loader()
    ingest_in_vectordb(json_docs, embeddings)