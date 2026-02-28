import os
from typing import Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_WORDS_DEFAULT = int(os.getenv("MAX_WORDS_DEFAULT", "8"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="EchoMind OpenAI Relay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}


class WhisperReq(BaseModel):
    system: str = "Respond professionally."
    user: Optional[str] = None
    text: Optional[str] = None
    maxWords: Optional[int] = None


class WhisperResp(BaseModel):
    text: str


def build_instructions(system: str, max_words: int) -> str:
    return f"""You are EchoMind, an AI conversation coach.
Answer in ENGLISH with at most {max_words} words.
No explanations. Only the whisper reply.

CHARACTER:
{system}
"""


@app.post("/whisper", response_model=WhisperResp)
def whisper(body: WhisperReq):

    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    transcript = (
    getattr(body, "user", None)
    or getattr(body, "text", None)
    or getattr(body, "message", None)
    or ""
).strip()

    if not transcript:
        raise HTTPException(status_code=400, detail="Missing transcript")

    max_words = int(body.maxWords or MAX_WORDS_DEFAULT)
    max_words = max(1, min(max_words, 20))

    instructions = build_instructions(body.system, max_words)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=transcript,
    )

    text = resp.output[0].content[0].text.strip()

    return {
    "text": text,
    "reply": text
}
