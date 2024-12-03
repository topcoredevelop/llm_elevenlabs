from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import asyncio
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# Initialiser logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialiser FastAPI-appen
app = FastAPI()

# Sett OpenAI API-nøkkel fra miljøvariabler
openai.api_key = os.getenv("OPENAI_API_KEY")

# CORS-innstillinger (valgfritt hvis du tester fra forskjellige domener)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modell for chat-forespørsler
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False

# Funksjon for å logge forespørsel og respons
@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logger.debug(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    logger.debug(f"Body: {body.decode('utf-8')}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

# Funksjon for å håndtere streaming-respons
async def event_stream(openai_response):
    try:
        async for chunk in openai_response:
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    data = f'data: {{"choices": [{{"delta": {delta}, "finish_reason": null}}]}}\n\n'
                    logger.debug(f"Streamed chunk: {data}")
                    yield data
            elif chunk.get("choices")[0].get("finish_reason") is not None:
                yield f'data: [DONE]\n\n'
                break
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f'data: [ERROR: "{str(e)}"]\n\n'

# Endepunkt for chat-kompletteringer
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Valider forespørselen
        if not request.messages:
            raise HTTPException(status_code=400, detail="Meldinger mangler i forespørselen.")
        
        logger.info(f"Processing request for model: {request.model}")
        
        # Kall OpenAI API
        openai_response = openai.ChatCompletion.acreate(
            model=request.model,
            messages=request.messages,
            stream=request.stream
        )

        # Returner streaming-respons hvis aktivert
        if request.stream:
            logger.info("Streaming response enabled")
            return StreamingResponse(event_stream(openai_response), media_type="text/event-stream")
        
        # Returner komplett respons for non-streaming
        result = await openai_response
        logger.debug(f"Non-streaming response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helse-sjekk for serveren
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}
