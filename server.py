from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import asyncio
import os
import logging

# Sett opp logger
logging.basicConfig(level=logging.INFO)

# Initialiser FastAPI-appen
app = FastAPI()

# Sett OpenAI API-nøkkel fra miljøvariabler
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY er ikke satt i miljøvariabler.")

# Modell for chat-forespørsler
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 256
    user_id: str = None

# Funksjon for å håndtere streaming-respons
async def event_stream(openai_response):
    try:
        async for chunk in openai_response:
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    # Bygg streaming JSON-responsen
                    yield f'data: {{"choices": [{{"delta": {delta}, "finish_reason": null}}]}}\n\n'
            if chunk.get("choices")[0].get("finish_reason") is not None:
                # Siste chunk som avslutter streamen
                yield "data: [DONE]\n\n"
                break
    except Exception as e:
        logging.error(f"Feil under streaming: {e}")
        yield f'data: {{"error": "En feil oppsto under streaming."}}\n\n'

# Endepunkt for chat-kompletteringer
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Valider forespørselen
        if not request.messages:
            raise HTTPException(status_code=400, detail="Meldinger mangler i forespørselen.")

        # Håndter bruker-ID (valgfritt felt fra ElevenLabs)
        openai_request = request.dict(exclude_none=True)
        if "user_id" in openai_request:
            openai_request["user"] = openai_request.pop("user_id")

        # Kall OpenAI API
        openai_response = openai.ChatCompletion.acreate(
            model=openai_request["model"],
            messages=openai_request["messages"],
            stream=openai_request["stream"],
            temperature=openai_request["temperature"],
            max_tokens=openai_request["max_tokens"]
        )

        # Returner streaming-respons hvis aktivert
        if request.stream:
            return StreamingResponse(event_stream(openai_response), media_type="text/event-stream")
        
        # Returner full respons hvis ikke-streaming
        result = await openai_response
        return result
    except Exception as e:
        logging.error(f"Feil: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helse-sjekk for serveren
@app.get("/health")
async def health_check():
    return {"status": "ok"}

