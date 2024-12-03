import json
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
app = FastAPI()

# Legg til CORS-middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

async def process_streaming_response(openai_response):
    try:
        async for chunk in openai_response:
            # Konverter chunk til dict og send
            chunk_dict = {
                "id": chunk.id,
                "object": chunk.object,
                "created": chunk.created,
                "model": chunk.model,
                "choices": [
                    {
                        "index": choice.index,
                        "delta": {
                            "role": choice.delta.role if choice.delta.role else "assistant",
                            "content": choice.delta.content if choice.delta.content else ""
                        },
                        "finish_reason": choice.finish_reason
                    } for choice in chunk.choices
                ]
            }
            yield f"data: {json.dumps(chunk_dict)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': 'Internal error occurred!'})}\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Konverter request til dict og håndter user_id
        request_data = request.dict(exclude_none=True)
        if "user_id" in request_data:
            request_data["user"] = request_data.pop("user_id")

        # Send forespørsel til OpenAI
        response = await openai.ChatCompletion.acreate(**request_data)

        # Håndter streaming-respons
        if request_data.get("stream", False):
            return StreamingResponse(
                process_streaming_response(response),
                media_type="text/event-stream"
            )

        # Returner vanlig respons
        return response

    except Exception as e:
        logger.error(f"Error i chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
