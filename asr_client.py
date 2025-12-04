# asr_client.py
"""
Simple wrapper around OpenAI's audio transcription API (Whisper).

Usage:
    from asr_client import transcribe_audio
    text = transcribe_audio("path/to/audio.wav")
"""

import os
from typing import Optional

from openai import OpenAI

# Reads your key from the OPENAI_API_KEY env var.
# Do NOT hard-code your key in this file.
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment. "
                "Run `export OPENAI_API_KEY='sk-...'` (Mac/Linux) first."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe_audio(
    path: str,
    model: str = "whisper-1",
    language: str = "en",
) -> str:
    """
    Transcribe a local audio file to text using Whisper via OpenAI API.

    Returns a plain text string transcript.
    """
    client = _get_client()

    with open(path, "rb") as f:
        # response_format="text" often returns a plain string; if not, we read `.text`.
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            language=language,
            response_format="text",
        )

    # GitHub issues show that with response_format="text" you sometimes
    # get a bare string, sometimes an object with `.text`.:contentReference[oaicite:0]{index=0}
    if isinstance(resp, str):
        return resp.strip()

    text = getattr(resp, "text", "")
    return text.strip()
