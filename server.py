import json
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import openai
from dotenv import load_dotenv

# Last miljøvariabler fra .env
load_dotenv()

# Sett opp logging
logging.basicConfig(level=logging.DEBUG)

# Sett OpenAI API-nøkkel
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY mangler i miljøvariabler.")

# Initialiser FastAPI-appen
app = FastAPI()

# Pydantic-modeller for forespørsel
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]  # Liste over Message-objekter
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# Funksjon for å justere max_tokens
def adjust_max_tokens(request_data):
    # Beregn tokenforbruk for meldinger
    message_tokens = sum(len(msg["content"].split()) for msg in request_data["messages"])
    remaining_tokens = 8192 - message_tokens

    # Juster max_tokens hvis nødvendig
    if "max_tokens" in request_data and request_data["max_tokens"] > remaining_tokens:
        request_data["max_tokens"] = max(remaining_tokens - 10, 0)  # Reserver buffer
    elif "max_tokens" not in request_data:
        request_data["max_tokens"] = max(remaining_tokens - 10, 0)
    
    logging.debug(f"Adjusted max_tokens: {request_data['max_tokens']}")
    return request_data

# Funksjon for streaming-respons
async def event_stream(openai_response):
    try:
        async for chunk in openai_response:
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield f'data: {{"choices": [{{"delta": {delta}, "finish_reason": null}}]}}\n\n'
            elif chunk.get("choices")[0].get("finish_reason") is not None:
                yield "data: [DONE]\n\n"
                break
    except Exception as e:
        logging.error(f"Feil under streaming: {e}")
        yield f'data: [ERROR: "{str(e)}"]\n\n'

# Endepunkt for chat completions
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Konverter forespørselen til en ordbok
        request_data = request.dict(exclude_none=True)

        # Valider meldingsstrukturen (konverter fra dict til Message-objekter)
        request_data["messages"] = [Message(**msg).dict() for msg in request_data["messages"]]

        # Juster max_tokens
        request_data = adjust_max_tokens(request_data)

        # Send forespørselen til OpenAI
        openai_response = openai.ChatCompletion.acreate(**request_data)

        # Returner streaming-respons hvis aktivert
        if request_data.get("stream", False):
            return StreamingResponse(event_stream(openai_response), media_type="text/event-stream")
        
        # Returner hele responsen for ikke-streaming
        result = await openai_response
        return result
    except Exception as e:
        logging.error(f"Feil under behandling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endepunkt for helse-sjekk
@app.get("/health")
async def health_check():
    return {"status": "ok"}
