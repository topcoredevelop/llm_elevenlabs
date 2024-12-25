import json
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import tiktoken
import uvicorn

# Last miljøvariabler fra .env
load_dotenv()

# Sett opp logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialiser OpenAI klient
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY mangler i miljøvariabler.")

# Initialiser FastAPI-appen
app = FastAPI()

# Legg til CORS-middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token grenser for modeller
MODEL_TOKEN_LIMITS = {
    'gpt-4-1106-preview': 128000,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384
}

# Pydantic-modeller
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user_id: Optional[str] = None

def count_tokens(text: str, model: str) -> int:
    """Beregn antall tokens i teksten"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split()) * 2  # Enkel fallback

def adjust_max_tokens(request_data: dict) -> dict:
    """Juster max_tokens basert på modell og innhold"""
    model = request_data["model"]
    model_limit = MODEL_TOKEN_LIMITS.get(model, 4096)
    
    # Beregn tokens brukt i meldinger
    total_tokens = sum(count_tokens(msg["content"], model) 
                      for msg in request_data["messages"])
    
    # Beregn tilgjengelige tokens
    available_tokens = model_limit - total_tokens - 50  # Buffer på 50 tokens
    
    # Sett max_tokens
    if "max_tokens" not in request_data or request_data["max_tokens"] is None:
        request_data["max_tokens"] = min(available_tokens, 4096)
    else:
        request_data["max_tokens"] = min(request_data["max_tokens"], available_tokens)
    
    logger.info(f"Justerte max_tokens til {request_data['max_tokens']}")
    return request_data

async def event_stream(completion):
    try:
        async for chunk in completion:
            chunk_dict = chunk.model_dump()
            yield f"data: {json.dumps(chunk_dict)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        request_data = request.dict(exclude_none=True)
        if "user_id" in request_data:
            request_data["user"] = request_data.pop("user_id")
            
        # Juster max_tokens før sending
        request_data = adjust_max_tokens(request_data)

        completion = await client.chat.completions.create(**request_data)

        if request_data.get("stream", False):
            return StreamingResponse(
                event_stream(completion),
                media_type="text/event-stream"
            )
        
        return completion.model_dump()

    except Exception as e:
        logger.error(f"Error i chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    print('')
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
