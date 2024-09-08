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

class Query(BaseModel):
    query: str

@app.get("/")
async def home():
    return {"Welcome to NyayNidhi": "Route to `/chat` to chat with the AI app."}

@app.post("/chat")
async def chat(query: Query, request: Request):
    if not hasattr(request.app.state, 'chat_histories'):
        request.app.state.chat_histories = {}
    
    chat_id = uuid.uuid4()
    chat_history = []
    
    response = await get_response(query.query, chat_history)
    chat_history.append({"human": query.query, "ai": response["output"]})
    
    request.app.state.chat_histories[chat_id] = chat_history

    async def generate() -> AsyncGenerator[str, None]:
        yield json.dumps({
            "chat_id": str(chat_id),
            "output": response["output"]
        }) + "\n"
        await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="application/json")



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)