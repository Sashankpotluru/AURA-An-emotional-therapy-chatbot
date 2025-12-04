# # llm_client.py

# import os
# from typing import Optional, Tuple

# from openai import OpenAI

# from emotion_router import (
#     analyze_with_style,
#     analyze_audio_with_asr,
#     EmotionResult,
#     StyleCode,
# )
# from response_policy import build_safe_prompt


# # You can swap this later (gpt-4.1, gpt-4.1-mini, etc.)
# DEFAULT_MODEL = "gpt-4.1-mini"


# def _get_client() -> OpenAI:
#     """
#     Returns an OpenAI client instance.
#     Make sure OPENAI_API_KEY is set in your environment.
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError(
#             "OPENAI_API_KEY is not set. "
#             "Export it in your shell or environment before running."
#         )
#     return OpenAI(api_key=api_key)


# def run_emotion_analysis(
#     text: Optional[str] = None,
#     wav_path: Optional[str] = None,
#     alpha_text: float = 0.6,
# ) -> Tuple[str, EmotionResult, StyleCode]:
#     """
#     Wrapper that:
#       - For audio-only: runs ASR, then fusion via analyze_audio_with_asr.
#       - For text-only: runs analyze_with_style(text=...).
#       - For both: runs analyze_with_style(text=..., wav_path=...).

#     Returns:
#         (effective_user_text, emo_res, style)
#     where effective_user_text is either the raw text or the ASR transcript.
#     """
#     if wav_path and not text:
#         # audio-only: ASR + fusion
#         emo_res, style, transcript = analyze_audio_with_asr(
#             wav_path,
#             alpha_text=alpha_text,
#         )
#         user_text_effective = transcript or ""
#     elif text and not wav_path:
#         emo_res, style = analyze_with_style(
#             text=text,
#             wav_path=None,
#             alpha_text=alpha_text,
#         )
#         user_text_effective = text
#     elif text and wav_path:
#         emo_res, style = analyze_with_style(
#             text=text,
#             wav_path=wav_path,
#             alpha_text=alpha_text,
#         )
#         user_text_effective = text
#     else:
#         raise ValueError("You must provide at least text or wav_path.")

#     return user_text_effective, emo_res, style


# def generate_reply(
#     text: Optional[str] = None,
#     wav_path: Optional[str] = None,
#     alpha_text: float = 0.6,
#     model: str = DEFAULT_MODEL,
# ) -> Tuple[str, EmotionResult, StyleCode, str]:
#     """
#     High-level entry point:

#       1. Runs emotion analysis (text / speech / fusion).
#       2. Builds a safe, style-aware prompt.
#       3. Calls the OpenAI model and gets a reply.

#     Returns:
#         (effective_user_text, emo_res, style, llm_reply)
#     """
#     user_text_effective, emo_res, style = run_emotion_analysis(
#         text=text,
#         wav_path=wav_path,
#         alpha_text=alpha_text,
#     )

#     prompt = build_safe_prompt(
#         user_text=user_text_effective,
#         emo_res=emo_res,
#         style=style,
#     )

#     client = _get_client()

#     # We send the whole constructed prompt as the user message.
#     # The prompt already contains instructions + user text + style.
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         temperature=0.7,
#         max_tokens=400,
#     )

#     reply_text = response.choices[0].message.content

#     return user_text_effective, emo_res, style, reply_text

# llm_client.py

import os
from typing import Optional, Tuple

from openai import OpenAI

from emotion_router import (
    analyze_with_style,
    analyze_audio_with_asr,   # uses ASR + fusion under the hood
)
from response_policy import build_safe_prompt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_reply(
    text: Optional[str] = None,
    wav_path: Optional[str] = None,
    alpha_text: float = 0.6,
    model: str = "gpt-4o-mini",
) -> Tuple[str, object, object, str]:
    """
    Core helper used by the CLI (and later UI).

    Returns:
        effective_text: the text that went into the LLM (user text or ASR transcript)
        emo_res: EmotionResult
        style: StyleCode
        reply: LLM-generated string
    """

    # --- Decide which analysis path to use ---

    if wav_path is not None and text is None:
        # AUDIO-ONLY MODE:
        # 1) ASR transcript
        # 2) Fusion of text (transcript) + speech
        emo_res, style, transcript = analyze_audio_with_asr(
            wav_path, alpha_text=alpha_text
        )
        # use transcript as the user's message for the LLM
        user_text = transcript or "[unintelligible audio]"
        effective_text = user_text

    else:
        # TEXT (or text+speech fusion if you ever pass both)
        emo_res, style = analyze_with_style(
            text=text,
            wav_path=wav_path,
            alpha_text=alpha_text,
        )
        user_text = text or "[no text provided]"
        effective_text = user_text

    # --- Build emotionally-aware, safety-aware prompt ---
    prompt = build_safe_prompt(
        user_text=user_text,
        emo_res=emo_res,
        style=style,
    )

    # --- Call OpenAI LLM ---
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are AURA, an empathetic multimodal assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.7,
    )

    reply = response.choices[0].message.content.strip()

    return effective_text, emo_res, style, reply

