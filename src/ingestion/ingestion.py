import os 
import sys 
import re 
import json 
import glob
import time
import requests
import warnings
import argparse
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

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=800)

# Store in vector database
def ingest_in_vectordb(text_chunk, embedding_model, collection_name):
    print("\nConnecting Qdrant DB")
    
    # Connect to the QdrantDB local
    client = QdrantClient(url="http://localhost:6333/")
    
    # Delete existing collection 
    client.delete_collection(collection_name)
    
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
        collection_name=collection_name
    )
    print("\n> Chunk of text saved in DB with Embedding format.")

    
# Ingest data from PDF format
def pdf_loader(section_type):
    if section_type == 'bns':                                     
        loader = PyPDFLoader("./data/BNS.pdf")
    print("\nSplitting PDF documents")
    documents = loader.load_and_split(text_splitter=text_splitter)
    return documents


def json_loader(section_type):
    section_data = []
    
    if section_type=='all_sections':
        file_paths = glob.glob(f"./data/*_data.json")
        
        for path in file_paths:
            with open(path, 'r') as file:
                data = json.load(file)
                section_data.extend(data)
        with open('./data/all_sections_data.json', 'w') as outputfile:
            json.dump(section_data, outputfile, indent=4)
    else:
        data_path = f'./data/{section_type}_data.json'
        with open(data_path, 'r') as json_file:
            section_data = json.load(json_file)

    # Create a Langchain document for each JSON record 
    documents = []
    for record in section_data:
        document = Document(
            page_content=record['content'],
            metadata={'section':record['section']}
        )
        documents.append(document)
    
    docs = text_splitter.split_documents(documents)
    return docs


# Main funciton to execute the script
def main(section_type, data_type, collection_name):
    # try:
    print("\n\n> Loading files..")
    if data_type == 'json':
        docs = json_loader(section_type)
    elif data_type=='pdf':
        docs = pdf_loader(section_type)
    print("\n> Saved in document loader format...")
    print("\n> Ingesting into Qdrant DB..")
    ingest_in_vectordb(docs, embeddings, collection_name)
    print("\n>Ingested in QdrantDB")
    print("\n> Data ingestion successful")
    print("\n\n")
    # except Exception as e:
    #     print(f"Something went wrong while data ingestion to vector db: {e}")


## Ingest data from JSON data
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load JSON documents and ingest them into QdrantDB.")
    
    parser.add_argument(
        '--section_type',
        type=str,
        required=True,
        help="The type of section to load JSON documents for [use `all_sections` for all sections]."
    )
    
    parser.add_argument(
        '--data_type',
        type=str,
        required=True,
        help="Define type of data pdf or json"
    )
    
    parser.add_argument(
        '--collection_name',
        type=str,
        required=True,
        help="The name of the collection"
    )
    
    args = parser.parse_args()
    
    main(args.section_type, args.data_type, args.collection_name)