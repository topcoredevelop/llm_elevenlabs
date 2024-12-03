from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import asyncio
import os
import logging

# Initialiser logger for debugging
logging.basicConfig(level=logging.DEBUG)

# Initialiser FastAPI-appen
app = FastAPI()

# Sett OpenAI API-nøkkel fra miljøvariabler
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY er ikke satt i miljøvariablene.")

# Modell for chat-forespørsler
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False

# Funksjon for å håndtere streaming-respons
async def event_stream(openai_response):
    try:
        async for chunk in openai_response:
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield f'data: {{"choices": [{{"delta": {delta}, "finish_reason": null}}]}}\n\n'
            elif chunk.get("choices")[0].get("finish_reason") is not None:
                # Siste chunk som avslutter streamen
                yield "data: [DONE]\n\n"
                break
    except Exception as e:
        logging.error(f"Feil under streaming: {str(e)}")
        yield f'data: [ERROR: "{str(e)}"]\n\n'

# Endepunkt for chat-kompletteringer
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logging.debug(f"Inkommende forespørsel: {request}")
        # Valider forespørselen
        if not request.messages:
            raise HTTPException(status_code=400, detail="Meldinger mangler i forespørselen.")

        # Kall OpenAI API
        openai_response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=request.messages,
            stream=request.stream
        )

        # Returner streaming-respons hvis aktivert
        if request.stream:
            return StreamingResponse(event_stream(openai_response), media_type="text/event-stream")

        # Returner komplett respons for non-streaming
        result = await openai_response
        logging.debug(f"Respons fra OpenAI (non-streaming): {result}")
        return result
    except Exception as e:
        logging.error(f"Feil: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helse-sjekk for serveren
@app.get("/health")
async def health_check():
    logging.debug("Helse-sjekk utført.")
    return {"status": "ok"}
