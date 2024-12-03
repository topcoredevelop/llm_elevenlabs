import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

# Sett OpenAI API-nøkkel
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ikke funnet i miljøvariabler")
openai.api_key = OPENAI_API_KEY

# Konfigurer logger
logging.basicConfig(level=logging.DEBUG)

# Initialiser FastAPI
app = FastAPI()

# Modell for forespørsler
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# Funksjon for streaming
async def event_stream(openai_response):
    try:
        async for chunk in openai_response:
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if delta:
                    yield f"data: {json.dumps({'choices': [{'delta': delta, 'finish_reason': None}]})}\n\n"
            if chunk.get("choices")[0].get("finish_reason") is not None:
                yield "data: [DONE]\n\n"
                break
    except Exception as e:
        logging.error(f"Feil under streaming: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# Endepunkt for chat-komplettering
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logging.debug(f"Inngående forespørsel: {request.json()}")

        # Send forespørsel til OpenAI API
        openai_response = await openai.ChatCompletion.acreate(
            model=request.model,
            messages=[msg.dict() for msg in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
        )

        # Returner streaming-respons
        if request.stream:
            return StreamingResponse(event_stream(openai_response), media_type="text/event-stream")

        # Returner full respons for non-streaming
        return openai_response

    except Exception as e:
        logging.error(f"Feil under behandling: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endepunkt for helse-sjekk
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
