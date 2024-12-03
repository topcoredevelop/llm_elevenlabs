import json
import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from typing import List, Optional
import openai
import tiktoken
from dotenv import load_dotenv
import uvicorn

# Last miljøvariabler fra .env
load_dotenv()

# Sett opp logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sett OpenAI API-nøkkel
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY mangler i miljøvariabler.")

# Initialiser FastAPI-appen
app = FastAPI(
    title="OpenAI Proxy API",
    description="Et proxy-API for OpenAI med token-håndtering og streaming-støtte",
    version="1.0.0"
)

# Legg til CORS-middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Juster dette for produksjon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic-modeller for forespørsel
class Message(BaseModel):
    role: str
    content: str

    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role må være enten system, user, eller assistant')
        return v

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError('Temperature må være mellom 0 og 2')
        return v

    @validator('model')
    def validate_model(cls, v):
        allowed_models = ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']
        if v not in allowed_models:
            raise ValueError(f'Model må være en av følgende: {", ".join(allowed_models)}')
        return v

# Token-grenser for ulike modeller
MODEL_TOKEN_LIMITS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384
}

def get_token_count(text: str, model: str) -> int:
    """Beregn antall tokens i en tekst for en spesifikk modell."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Kunne ikke beregne tokens for {model}: {e}")
        # Fallback til en enkel estimering
        return len(text.split()) * 1.3

def adjust_max_tokens(request_data: dict) -> dict:
    """Juster max_tokens basert på modellens grense og meldingenes lengde."""
    model = request_data["model"]
    model_limit = MODEL_TOKEN_LIMITS.get(model, 4096)
    
    # Beregn total token-bruk for alle meldinger
    total_tokens = sum(
        get_token_count(msg["content"], model)
        for msg in request_data["messages"]
    )
    
    # Beregn gjenværende tokens
    remaining_tokens = model_limit - total_tokens
    safe_buffer = 50  # Buffer for å unngå å nå grensen
    
    # Juster max_tokens
    if "max_tokens" in request_data:
        request_data["max_tokens"] = min(
            request_data["max_tokens"],
            max(remaining_tokens - safe_buffer, 0)
        )
    else:
        request_data["max_tokens"] = max(remaining_tokens - safe_buffer, 0)
    
    logger.debug(f"Adjusted max_tokens: {request_data['max_tokens']}")
    return request_data

async def event_stream(openai_response):
    """Håndter streaming av OpenAI-respons."""
    try:
        async for chunk in openai_response:
            if chunk and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield f'data: {json.dumps({"choices": [{"delta": delta, "finish_reason": None}]})}\n\n'
            elif chunk and chunk.get("choices")[0].get("finish_reason") is not None:
                yield f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'
                yield "data: [DONE]\n\n"
                break
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f'data: {json.dumps({"error": str(e)})}\n\n'

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """Hovedendepunkt for chat completions."""
    try:
        # Log innkommende forespørsel
        client_host = raw_request.client.host
        logger.info(f"Innkommende forespørsel fra {client_host} for modell {request.model}")
        
        # Konverter forespørselen til en ordbok og juster tokens
        request_data = request.dict(exclude_none=True)
        request_data = adjust_max_tokens(request_data)
        
        # Send forespørselen til OpenAI
        openai_response = openai.ChatCompletion.acreate(**request_data)
        
        # Håndter streaming hvis aktivert
        if request_data.get("stream", False):
            return StreamingResponse(
                event_stream(openai_response),
                media_type="text/event-stream"
            )
        
        # Returner hele responsen for ikke-streaming
        result = await openai_response
        return result

    except Exception as e:
        logger.error(f"Error i chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Endepunkt for helse-sjekk."""
    try:
        # Test OpenAI-tilkobling
        openai.Model.list()
        return {
            "status": "healthy",
            "openai_connection": "ok",
            "api_version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Helsesjekk feilet: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )

# Legg til en enkel hjemmeside-rute
@app.get("/")
async def root():
    return {
        "message": "OpenAI Proxy API",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    # Hent port fra miljøvariabel (Railway setter denne)
    port = int(os.getenv("PORT", 8000))
    
    # Start serveren
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Sett til False i produksjon
    )
