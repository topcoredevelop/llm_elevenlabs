from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import asyncio
import os

# Initialiser FastAPI-appen
app = FastAPI()

# Sett OpenAI API-nøkkel fra miljøvariabler
openai.api_key = os.getenv("OPENAI_API_KEY")

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
                yield f'data: [DONE]\n\n'
                break
    except Exception as e:
        print(f"Feil under streaming: {e}")
        yield f'data: [ERROR: "{str(e)}"]\n\n'

# Endepunkt for chat-kompletteringer
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Valider forespørselen
        if not request.messages:
            raise HTTPException(status_code=400, detail="Meldinger mangler i forespørselen.")
        
        # Kall OpenAI API
        openai_response = openai.ChatCompletion.acreate(
            model=request.model,
            messages=request.messages,
            stream=request.stream
        )

        # Returner streaming-respons hvis aktivert
        if request.stream:
            return StreamingResponse(event_stream(openai_response), media_type="text/event-stream")
        
        # Returner komplett respons for non-streaming
        result = await openai_response
        return result
    except Exception as e:
        print(f"Feil: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helse-sjekk for serveren
@app.get("/health")
async def health_check():
    return {"status": "ok"}

