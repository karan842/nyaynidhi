from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union
from src.generate_response import get_response
import uvicorn

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; adjust this for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (e.g., GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the data model for the request body
class ChatRequest(BaseModel):
    query: str

# Define the data model for the response
class ChatResponse(BaseModel):
    output: str
    chat_history: List[Dict[str, Union[str, str]]]

# Chat history storage
chat_history = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Process the query with the current chat history
        response = await get_response(request.query, chat_history)
        
        # Append new entries to chat history
        chat_history.extend([{"role": "human", "content": request.query}, {"role": "assistant", "content": response["output"]}])
        
        return ChatResponse(output=response["output"], chat_history=chat_history)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong with processing the request.")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)