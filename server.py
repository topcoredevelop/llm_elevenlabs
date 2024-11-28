from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
from dotenv import load_dotenv
import logging
import json
from fastapi.responses import StreamingResponse

# Sett opp logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Laste inn miljøvariabler
load_dotenv()

# Hente OpenAI API-nøkkel
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY not set in environment variables")
    raise ValueError("OPENAI_API_KEY not set in environment variables")

# Opprette FastAPI-app
app = FastAPI()

# Modeller for forespørsel og svar
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.middleware("http")
async def log_request(request: Request, call_next):
    """
    Logger alle forespørsler som kommer inn til serveren.
    """
    try:
        body = await request.body()
        logger.info(f"Innkalling til sti: {request.url.path}")
        logger.info(f"Forespørselens headers: {dict(request.headers)}")
        logger.info(f"Forespørselens body: {body.decode('utf-8')}")
    except Exception as e:
        logger.error(f"Feil under logging av forespørsel: {str(e)}")
    response = await call_next(request)
    return response

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Overskrive systemmeldingen
        for message in request.messages:
            if message.role == "system":
                message.content = "Du er en hjelpsom assistent som kommuniserer på norsk."

        # Hvis strømming er aktivert
        if request.stream:
            async def stream_openai():
                response = await openai.ChatCompletion.acreate(
                    model=request.model,
                    messages=[message.dict() for message in request.messages],
                    temperature=request.temperature,
                    stream=True  # Aktiver strømming
                )
                async for chunk in response:
                    yield json.dumps(chunk) + "\n"

            # Returner strømmet respons
            return StreamingResponse(stream_openai(), media_type="application/json")
        else:
            # Vanlig respons (ikke strømmet)
            response = await openai.ChatCompletion.acreate(
                model=request.model,
                messages=[message.dict() for message in request.messages],
                temperature=request.temperature
            )

            # Logg OpenAI-responsen
            logger.info(f"OpenAI-respons mottatt: {response}")

            return response

    except Exception as e:
        # Logg feilen
        logger.error(f"Feil oppstod under behandling: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server is running. Use POST on /v1/chat/completions."}
