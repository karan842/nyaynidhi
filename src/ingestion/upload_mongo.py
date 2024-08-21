import os 
import json 
from pymongo import MongoClient 
from dotenv import load_dotenv

# Load dotenv 
load_dotenv()
mongo_url = os.getenv('MONGO_URL')

# Load configuration
config_file = "config.json"
with open(config_file, "r") as file:
    config = json.load(file)

# Collection name
collection_name = config['COLLECTION_NAME']['ipc-800']
mongo_db_name = config['MONGODB']

def insert_to_mongo(url, db_name, collection_name, section_type): 
    
    # Connect to MongoDB
    client = MongoClient(url)
    mongo_db = client[mongo_db_name]
    collection_name = mongo_db[collection_name]
    
    # Create collection
    collection = mongo_db.create_collection(collection_name)

    print("Loading data from JSON")
    
    # Read JSON data from a file 
    with open(f'./data/{section_type}_data.json', 'r') as ipc_data_file:
        ipc_data = json.load(ipc_data_file)

    print("Inserting the data")
    
    # Insert data into MongoDB
    if isinstance(ipc_data, list):
        # Insert multiple documents
        result = collection.insert_many(ipc_data)
    else:
        result = collection.insert_one(ipcs_data)
        
    # Output result
    print(f"Data inserted")


if __name__ == '__main__':
    insert_to_mongo(mongo_url, mongo_db_name, "crpc-data-001", "crpc")