import os 
import numpy as np
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient 
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, SearchParams
import json 
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from src.utils import load_prompt_template

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')
qdrant_cloud_url = os.getenv("QDRANT_CLOUD_URL")
qdrant_cloud_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
genai.configure(api_key=gemini_api_key)


# Semantic Cache for direct user query
class SemanticCache:
    def __init__(self, embedding_model:str, cache_collection_name:str, llm:str, threshold=0.7):
        self.embedding_model = embedding_model
        self.euclidean_threshold = threshold
        self.query_response = {}
        self.result = None
        self.config_file = './artifacts/config.json'
        
        # Set Qdrant
        with open(self.config_file, 'r') as file:
            config = json.load(file)
        
        # is it to cloud?
        to_cloud = config["IS_QDRANT_TO_CLOUD"]
        
        self.cache_client = QdrantClient(url="http://localhost:6333/") if not to_cloud else QdrantClient(url=qdrant_cloud_url, api_key=qdrant_cloud_api_key,)
        self.cache_collection_name = cache_collection_name
        
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=llm,
            temperature=0.2
        )

        is_exists_collection = self.cache_client.collection_exists(collection_name=self.cache_collection_name)
        if not is_exists_collection:
            self.cache_client.create_collection(
                collection_name=self.cache_collection_name,
                vectors_config = models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                )
            )

    def get_embedding(self, question):
        embeddings = genai.embed_content(model=self.embedding_model,
                             content=question)
        return list(embeddings['embedding'])  
         

    def search_cache(self, embedding):
        search_result = self.cache_client.search(
            collection_name=self.cache_collection_name,
            query_vector=embedding,
            limit=1
        )
        return search_result

    def add_to_cache(self, question, response_text):
        # Create a unique ID for the new point
        point_id = str(uuid.uuid4())
        vector = self.get_embedding(question)
        # Extract the output from the response_text dictionary
        output = response_text['output']
        # Create the point with payload
        point = PointStruct(id=point_id, vector=vector, payload={"response_text": output})
        # Upload the point to the cache
        self.cache_client.upsert(
            collection_name=self.cache_collection_name,
            points=[point]
        )
    
    def find_in_cache(self, question):
        vector = self.get_embedding(question)
        search_result = self.search_cache(vector)
        

        if search_result:
            for s in search_result:
                if s.score >= self.euclidean_threshold:
                    self.query_response['question'] = question
                    self.query_response['response'] = s.payload['response_text']
                    break 
                
        # print(self.query_response)
        if self.query_response:
            prompt_template = load_prompt_template(file_path="./prompt_templates/cache_prompts.txt")
            prompt = PromptTemplate.from_template(prompt_template)
            
            chain = prompt | self.llm | StrOutputParser()
            
            self.result = chain.invoke({"question":self.query_response['question'], "context": self.query_response['response']})
        return {"output":self.result}











