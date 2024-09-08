import os 
import json 
from src.cache_utils.semantic_cache_response import SemanticCache
from src.agents_utils.agents import judicial_agent
from dotenv import load_dotenv


# Generate response including semantic cache
async def get_response(query:str, chat_history:list, section_type="BNS"):
    
    # Load config
    config_file = "./artifacts/config.json"
    with open(config_file, "r") as file:
        config = json.load(file)
    
    # Semantic cache collection
    cache_collection = config["CACHE_COLLECTION"]["semantic-cache"]
    
    # Embedding_model name
    embedding_model = config["EMBEDDING_MODEL"]

    ## LLM 
    llm = config["LLAMA"]
    
    
    """Cache responses"""
    semantic_cache = SemanticCache(embedding_model, cache_collection,
                                   llm)
    
    # Check if the anwer present in the cache
    cached_response = semantic_cache.find_in_cache(query)
    if cached_response['output']:
        return cached_response

    # If the answer is not in the cache, generate it using the judicial agent
    response = judicial_agent(query, section_type, chat_history)
    
    # Add the response to the cache
    semantic_cache.add_to_cache(query, response)
    
    return response

if __name__ == "__main__":
    chat_history = []
    
    i = 0
    
    while i < 3:
        query = input("Ask a query: ")
        response = get_response(query,chat_history=chat_history)
        chat_history.extend({"human":query, "ai":response["output"]})
        print(response['output'])
        print("\n\n")
        i+=1