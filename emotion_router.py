# emotion_router.py
"""
High-level router that decides:
- text-only emotion analysis
- speech-only emotion analysis
- text+speech fusion

Returns a unified 7-way emotion distribution:
    ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Dict

import numpy as np

from text_erc.infer import predict_proba as text_predict_proba
from speech_erc.infer import predict as speech_predict
from fusion.infer import fuse_text_speech
from text_erc.infer import predict_proba as text_predict_proba
from asr_client import transcribe_audio





BASE_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# Map GoEmotions labels -> our 7 base emotions.
# Anything not explicitly listed will fall back to 'neutral'.
GOE_TO_BASE = {
    # positive
    "admiration": "happy",
    "amusement": "happy",
    "approval": "happy",
    "caring": "happy",
    "curiosity": "happy",
    "desire": "happy",
    "excitement": "happy",
    "gratitude": "happy",
    "joy": "happy",
    "love": "happy",
    "optimism": "happy",
    "pride": "happy",
    "relief": "happy",
    # negative
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "disgust",
    "embarrassment": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "remorse": "sad",
    "sadness": "sad",
    "disappointment": "sad",
    "grief": "sad",
    # cognitive / mixed
    "confusion": "neutral",
    "realization": "neutral",
    "surprise": "surprise",
    "neutral": "neutral",
}

# For basic crisis detection – this is NOT a diagnosis, just a safety heuristic.
CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end it all",
    "end my life",
    "want to die",
    "want to disappear",
    "hurt myself",
    "self harm",
    "self-harm",
    "can't go on",
    "cant go on",
]

NEGATIVE_EMOS = {"sad", "angry", "fear", "disgust"}

def detect_crisis(
    probs_7: np.ndarray,
    labels_7: List[str],
    text: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Very simple crisis heuristic:
    - Look for explicit self-harm / suicidal phrases in the text.
    - Combine with strong negative emotions.
    Returns (flag, reason_string_or_None).
    """
    # Map probs to dict for convenience
    prob_dict = {lab: float(p) for lab, p in zip(labels_7, probs_7)}

    # 1) Keyword hit?
    text_lower = (text or "").lower()
    keyword_hit = any(kw in text_lower for kw in CRISIS_KEYWORDS)

    # 2) Strong negative emotion?
    neg_max = max(prob_dict.get(e, 0.0) for e in NEGATIVE_EMOS)

    # Heuristic:
    # - If user mentions clear crisis keywords AND strong negative emotion -> flag.
    if keyword_hit and neg_max >= 0.4:
        reason = (
            f"High negative emotion (max={neg_max:.2f}) "
            f"and crisis-related language detected."
        )
        return True, reason

    # Optional: purely emotion-based flag (very conservative).
    # For now, we leave this off to avoid over-flagging:
    # if neg_max >= 0.9:
    #     return True, f"Extremely high negative emotion (max={neg_max:.2f})."

    return False, None


@dataclass
class EmotionResult:
    mode: Literal["text", "speech", "fusion"]
    probs: np.ndarray          # shape (7,)
    labels: List[str]          # BASE_EMOTIONS in order
    raw_extra: Dict            # optional: raw 28d probs, etc.
    crisis_flag: bool = False
    crisis_reason: Optional[str] = None


def _project_goe_to_base(probs_28: np.ndarray, labels_28: List[str]) -> np.ndarray:
    """
    Sum GoEmotions probabilities into our 7 base emotions,
    then renormalize to sum to 1.
    """
    base = np.zeros(len(BASE_EMOTIONS), dtype=np.float32)

    for p, lab in zip(probs_28, labels_28):
        base_lab = GOE_TO_BASE.get(lab, "neutral")
        idx = BASE_EMOTIONS.index(base_lab)
        base[idx] += float(p)

    s = base.sum()
    if s > 0:
        base /= s
    return base


def analyze_text(text: str) -> EmotionResult:
    """
    Text-only emotion analysis.
    Returns 7-way EmotionResult.
    """
    # Just call predict_proba(text) – it already returns (probs_28, labels_28)
    probs_28, labels_28 = text_predict_proba(text, return_labels=True)
    base_probs = _project_goe_to_base(probs_28, labels_28)

    crisis_flag, crisis_reason = detect_crisis(
        base_probs,
        BASE_EMOTIONS,
        text=text,
    )

    return EmotionResult(
        mode="text",
        probs=base_probs,
        labels=BASE_EMOTIONS,
        raw_extra={"probs_28": probs_28, "labels_28": labels_28},
        crisis_flag=crisis_flag,
        crisis_reason=crisis_reason,
    )




def analyze_speech(wav_path: str) -> EmotionResult:
    """
    Speech-only emotion analysis.
    Assumes speech_erc.infer.predict(..., return_probs=True)
    returns (probs, label_names) in BASE_EMOTIONS order.
    """
    probs_7, labels_7 = speech_predict(
        wav_path, top_k=len(BASE_EMOTIONS), return_probs=True
    )

    # For safety, reorder to BASE_EMOTIONS if needed
    label_to_idx = {lab: i for i, lab in enumerate(labels_7)}
    ordered = np.array([probs_7[label_to_idx[lab]] for lab in BASE_EMOTIONS], dtype=np.float32)

    # Normalize
    s = ordered.sum()
    if s > 0:
        ordered /= s

    # For now, no text → we don't run keyword-based crisis detection.
    crisis_flag, crisis_reason = detect_crisis(
        ordered,
        BASE_EMOTIONS,
        text=None,
    )

    return EmotionResult(
        mode="speech",
        probs=ordered,
        labels=BASE_EMOTIONS,
        raw_extra={"probs_raw": probs_7, "labels_raw": labels_7},
        crisis_flag=crisis_flag,
        crisis_reason=crisis_reason,
    )



def analyze_fusion(text: str, wav_path: str, alpha_text: float = 0.6) -> EmotionResult:
    """
    Late fusion using fusion.infer.fuse_text_speech.
    Assumes fusion already outputs BASE_EMOTIONS order.
    """
    text_probs_7, speech_probs_7, fused_probs_7, labels_7 = fuse_text_speech(
        text, wav_path, alpha_text=alpha_text, return_all=True
    )

    # labels_7 should already match BASE_EMOTIONS,
    label_to_idx = {lab: i for i, lab in enumerate(labels_7)}
    fused_ordered = np.array(
        [fused_probs_7[label_to_idx[lab]] for lab in BASE_EMOTIONS],
        dtype=np.float32,
    )

    s = fused_ordered.sum()
    if s > 0:
        fused_ordered /= s

    crisis_flag, crisis_reason = detect_crisis(
        fused_ordered,
        BASE_EMOTIONS,
        text=text,
    )

    return EmotionResult(
        mode="fusion",
        probs=fused_ordered,
        labels=BASE_EMOTIONS,
        raw_extra={
            "text_probs_7": text_probs_7,
            "speech_probs_7": speech_probs_7,
            "labels_7": labels_7,
            "alpha_text": alpha_text,
        },
        crisis_flag=crisis_flag,
        crisis_reason=crisis_reason,
    )



def analyze(
    text: Optional[str] = None,
    wav_path: Optional[str] = None,
    alpha_text: float = 0.6,
) -> EmotionResult:
    """
    Unified entry point.

    - If only text is given → text-only.
    - If only wav_path is given → speech-only.
    - If both → fusion.
    """
    if text and wav_path:
        return analyze_fusion(text, wav_path, alpha_text=alpha_text)
    elif text:
        return analyze_text(text)
    elif wav_path:
        return analyze_speech(wav_path)
    else:
        raise ValueError("Must provide at least text or wav_path to analyze().")



# ================== POLICY / STYLE MAPPING ==================

@dataclass
class StyleCode:
    tone: str         # e.g. "warm", "very_warm", "calm", "firm", "neutral", "cheerful"
    verbosity: str    # "short", "medium", "long"
    directness: str   # "reflective", "balanced", "directive"


def _style_from_emotions(prob_dict: Dict[str, float]) -> StyleCode:
    """
    Map aggregated emotion probs (over BASE_EMOTIONS) to a simple style code.

    prob_dict keys:
        "neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"
    """

    # ---- read probs with safe defaults ----
    neutral  = float(prob_dict.get("neutral", 0.0))
    happy    = float(prob_dict.get("happy", 0.0))
    sad      = float(prob_dict.get("sad", 0.0))
    angry    = float(prob_dict.get("angry", 0.0))
    fear     = float(prob_dict.get("fear", 0.0))
    disgust  = float(prob_dict.get("disgust", 0.0))
    surprise = float(prob_dict.get("surprise", 0.0))

    # ---- sensible defaults ----
    tone = "neutral"
    verbosity = "medium"
    directness = "balanced"

    # Determine primary emotion
    emo_pairs = [
        ("neutral", neutral),
        ("happy", happy),
        ("sad", sad),
        ("angry", angry),
        ("fear", fear),
        ("disgust", disgust),
        ("surprise", surprise),
    ]
    primary, primary_val = max(emo_pairs, key=lambda x: x[1])

    # 1) SAD-dominant → very warm + reflective support
    if primary == "sad":
        if sad >= 0.75:
            tone = "very_warm"
            verbosity = "long"
            directness = "reflective"
        elif sad >= 0.4:
            tone = "warm"
            verbosity = "medium"
            directness = "reflective"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)

    # 2) ANGRY-dominant → calm/firm de-escalation (unless sadness is very high)
    if primary == "angry":
        if angry >= 0.7 and sad < 0.4:
            tone = "calm"
            verbosity = "medium"
            directness = "balanced"
        elif angry >= 0.4 and sad < 0.4:
            tone = "firm"
            verbosity = "medium"
            directness = "balanced"
        else:
            # mixed anger + sadness → lean more supportive
            tone = "warm"
            verbosity = "medium"
            directness = "reflective"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)

    # 3) FEAR-dominant → reassuring & somewhat long
    if primary == "fear":
        if fear >= 0.6:
            tone = "very_warm"
            verbosity = "long"
            directness = "balanced"
        else:
            tone = "warm"
            verbosity = "medium"
            directness = "balanced"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)

    # 4) DISGUST-dominant → calm but directive (often boundaries / “ew” reactions)
    if primary == "disgust":
        tone = "calm"
        verbosity = "medium"
        directness = "directive"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)

    # 5) HAPPY-dominant and negatives small → lighter style
    if primary == "happy" and max(sad, angry, fear, disgust) < 0.3:
        tone = "cheerful"
        verbosity = "medium"
        directness = "balanced"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)

    # 6) NEUTRAL-dominant and negatives tiny → short neutral response
    if primary == "neutral" and max(sad, angry, fear, disgust) < 0.2:
        tone = "neutral"
        verbosity = "short"
        directness = "balanced"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)

    # 7) SURPRISE-dominant or mixed/unclear → fallback
    # keep defaults but maybe slightly warmer if happy is also notable
    if primary == "surprise" or True:
        if happy > max(sad, angry, fear, disgust):
            tone = "warm"
        else:
            tone = "neutral"
        verbosity = "medium"
        directness = "balanced"
        return StyleCode(tone=tone, verbosity=verbosity, directness=directness)



def analyze_with_style(
    text: Optional[str] = None,
    wav_path: Optional[str] = None,
    alpha_text: float = 0.6,
) -> Tuple[EmotionResult, StyleCode]:
    """
    Wrapper: run analyze(...) to get EmotionResult,
    then compute a StyleCode from its 7-way probs.
    """
    emo_res = analyze(text=text, wav_path=wav_path, alpha_text=alpha_text)

    prob_dict = {
        label: float(p) for label, p in zip(emo_res.labels, emo_res.probs)
    }
    style = _style_from_emotions(prob_dict)
    if emo_res.crisis_flag:
        style = StyleCode(
            tone="very_warm",
            verbosity="long",
            directness="reflective",
        )
    return emo_res, style


def analyze_audio_with_asr(
    wav_path: str,
    alpha_text: float = 0.6,
) -> Tuple[EmotionResult, StyleCode, str]:
    """
    Audio-only entry point with real ASR:
    1) Transcribe `wav_path` using OpenAI Whisper (via asr_client.transcribe_audio).
    2) Compute text-only and speech-only emotions.
    3) Choose a dynamic alpha_text for fusion based on negativity in text vs speech.
    4) Run fusion + style.
    Falls back to speech-only if ASR fails or returns an empty transcript.
    """

    # 1) Run ASR on the audio file
    try:
        transcript = transcribe_audio(wav_path)
        transcript = (transcript or "").strip()
    except Exception as e:
        print(f"[ASR] Error transcribing {wav_path}: {e}")
        transcript = ""

    # If ASR failed or produced nothing, fall back to speech-only analysis
    if not transcript:
        print("[ASR] Empty transcript, falling back to speech-only emotion analysis.")
        speech_res = analyze_speech(wav_path)

        prob_dict = {
            label: float(p) for label, p in zip(speech_res.labels, speech_res.probs)
        }
        style = _style_from_emotions(prob_dict)
        if speech_res.crisis_flag:
            style = StyleCode(
                tone="very_warm",
                verbosity="long",
                directness="reflective",
            )

        # Return speech-only result + style + empty transcript
        return speech_res, style, transcript

    # 2) Text-only emotions from the *actual* transcript
    text_res = analyze_text(transcript)   # 7-way over BASE_EMOTIONS
    text_probs = text_res.probs

    # 3) Speech-only emotions
    speech_res = analyze_speech(wav_path)
    speech_probs = speech_res.probs

    # ---- measure "negativity" in text vs speech ----
    neg_labels = ["sad", "angry", "fear", "disgust"]

    text_neg = max(
        (text_probs[text_res.labels.index(lab)] for lab in neg_labels),
        default=0.0,
    )
    speech_neg = max(
        (speech_probs[speech_res.labels.index(lab)] for lab in neg_labels),
        default=0.0,
    )

    # ---- dynamic alpha heuristic ----
    # Case 1: text is mostly neutral, speech clearly negative → emphasize speech.
    if text_neg < 0.2 and speech_neg > 0.5:
        dynamic_alpha = 0.3   # 30% text, 70% speech
    # Case 2: both clearly emotional → balanced fusion.
    elif text_neg > 0.4 and speech_neg > 0.4:
        dynamic_alpha = 0.5
    # Default: rely a bit more on text (original behaviour).
    else:
        dynamic_alpha = alpha_text

    # 4) Now run full fusion+style with this alpha
    emo_res, style = analyze_with_style(
        text=transcript,
        wav_path=wav_path,
        alpha_text=dynamic_alpha,
    )

    # Stash debug info
    if emo_res.raw_extra is None:
        emo_res.raw_extra = {}
    emo_res.raw_extra["dynamic_alpha"] = float(dynamic_alpha)
    emo_res.raw_extra["text_neg"] = float(text_neg)
    emo_res.raw_extra["speech_neg"] = float(speech_neg)
    emo_res.raw_extra["transcript"] = transcript

    return emo_res, style, transcript



