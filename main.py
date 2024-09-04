import os 
import sys 
import json 
import uuid
import uvicorn
import asyncio
from fastapi import FastAPI, Request 
from fastapi.responses import StreamingResponse 
from typing import AsyncGenerator, List
from pydantic import BaseModel
from src.generate_response import get_response

app = FastAPI()

app = FastAPI()

class ChatMessage(BaseModel):
    human: str
    ai: str

class Query(BaseModel):
    query: str

@app.get("/")
async def home():
    return {"Welcome to NyayNidhi": "Route to `/chat` to chat with the AI app."}

@app.post("/chat/")
async def stream_response(query: Query):
    chat_history = []
    response = get_response(query.query, chat_history)
    chat_history.append({"human": query.query, "ai": response["output"]})

    # Stream the response to the client
    async def generate():
        yield json.dumps({"output": response["output"]}) + "\n"
        await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="application/json")

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)