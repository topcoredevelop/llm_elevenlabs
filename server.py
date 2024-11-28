from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
from dotenv import load_dotenv

# Laste inn miljøvariabler
load_dotenv()

# Hente OpenAI API-nøkkel
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set in .env file")

# Opprette FastAPI-app
app = FastAPI()

# Modeller for forespørsel og svar
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Asynkron forespørsel til OpenAI
        response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=[message.dict() for message in request.messages],
            temperature=request.temperature
        )
        return response  # Returner svaret fra OpenAI
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server is running. Use POST on /v1/chat/completions."}
