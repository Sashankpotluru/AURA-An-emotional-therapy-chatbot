# api_server.py

import os
import tempfile
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llm_client import generate_reply
from asr_client import transcribe_audio

# ---------- FastAPI setup ----------

app = FastAPI(title="AURA API")

# Allow same-origin and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Serve /static/* and index.html
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    """Serve the chat UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


# ---------- Schemas ----------

class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    reply: str
    mode: str
    emotions: Dict[str, float]
    style: Dict[str, str]
    crisis_flag: bool
    crisis_reason: Optional[str] = None
    effective_text: str


# ---------- Helpers ----------

def _serialize_response(
    effective_text,
    emo_res,
    style,
    reply,
) -> ChatResponse:
    emotions = {
        label: float(p) for label, p in zip(emo_res.labels, emo_res.probs)
    }
    style_dict = {
        "tone": style.tone,
        "verbosity": style.verbosity,
        "directness": style.directness,
    }

    return ChatResponse(
        reply=reply,
        mode=emo_res.mode,
        emotions=emotions,
        style=style_dict,
        crisis_flag=getattr(emo_res, "crisis_flag", False),
        crisis_reason=getattr(emo_res, "crisis_reason", None),
        effective_text=effective_text,
    )


# ---------- /api/chat  (text) ----------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Text-only chat endpoint.
    """
    effective_text, emo_res, style, reply = generate_reply(
        text=req.text,
        wav_path=None,
        alpha_text=0.6,
    )
    return _serialize_response(effective_text, emo_res, style, reply)


# ---------- /api/chat_audio  (voice via mic) ----------

@app.post("/api/chat_audio", response_model=ChatResponse)
async def chat_audio(file: UploadFile = File(...)):
    """
    Audio chat endpoint.

    Frontend records from the mic (MediaRecorder) and sends a blob here as "file".
    We:
      1) save it to a temp file,
      2) run ASR -> transcript with Whisper (OpenAI),
      3) run emotion + LLM on the transcript (text-only for now),
      4) return the same JSON shape as /api/chat.

    NOTE: For browser mic input (webm/ogg), we *do not* run speech_erc,
    because the model currently expects WAV (CREMA-D). You still have
    full text_erc + crisis detection based on the transcript.
    """
    # Save uploaded audio to a temp file
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1) Transcribe audio to text via Whisper/OpenAI
        transcript = transcribe_audio(tmp_path)

        # 2) Use transcript as text input to AURA (no speech_erc here)
        effective_text, emo_res, style, reply = generate_reply(
            text=transcript,
            wav_path=None,
            alpha_text=0.6,
        )

        # Override effective_text to be explicit
        effective_text = transcript or "[empty transcript]"

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return _serialize_response(effective_text, emo_res, style, reply)
