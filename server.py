import json
import os
import fastapi
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import uvicorn
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional

# Last inn miljøvariabler fra .env-filen
load_dotenv()

# Hent OpenAI API-nøkkelen fra miljøvariabler
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY ikke funnet i miljøvariabler")

app = fastapi.FastAPI()
oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Logger for debugging
logging.basicConfig(level=logging.DEBUG)

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
    user_id: Optional[str] = None

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> StreamingResponse:
    try:
        # Konverter forespørselen til et dictionary-format
        oai_request = request.dict(exclude_none=True)
        if "user_id" in oai_request:
            oai_request["user"] = oai_request.pop("user_id")

        logging.debug(f"Inngående forespørsel: {oai_request}")

        # Håndter streaming-respons
        if request.stream:
            chat_completion_coroutine = oai_client.chat.completions.create(**oai_request)

            async def event_stream():
                try:
                    async for chunk in chat_completion_coroutine:
                        # Konverter til JSON
                        chunk_dict = chunk.model_dump()
                        yield f"data: {json.dumps(chunk_dict)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logging.error(f"En feil oppsto: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # Håndter non-streaming respons
        chat_completion = await oai_client.chat.completions.create(**oai_request)
        return chat_completion

    except Exception as e:
        logging.error(f"Feil i behandling: {str(e)}")
        raise fastapi.HTTPException(status_code=500, detail=str(e))

# Endepunkt for helse-sjekk
@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
