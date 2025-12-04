# fusion/infer.py
"""
Late fusion of text + speech emotions into a 7-way distribution:
    ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
"""

from typing import List, Tuple

import numpy as np

from text_erc.infer import predict_proba as text_predict_proba
from speech_erc.infer import predict as speech_predict

# Our unified 7-way space
EMO7: List[str] = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# Map GoEmotions labels -> these 7 buckets.
# (Same semantics as GOE_TO_BASE in emotion_router.py.)
GOE_TO_EMO7 = {
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


def _project_goe_to_emo7(probs_28: np.ndarray, labels_28: List[str]) -> np.ndarray:
    """
    Sum GoEmotions 28-way probabilities into our 7 EMO7 buckets,
    then renormalize to sum to 1.
    """
    base = np.zeros(len(EMO7), dtype=np.float32)

    for p, lab in zip(probs_28, labels_28):
        base_lab = GOE_TO_EMO7.get(lab, "neutral")
        idx = EMO7.index(base_lab)
        base[idx] += float(p)

    s = base.sum()
    if s > 0:
        base /= s
    return base


def fuse_text_speech(
    text: str,
    wav_path: str,
    alpha_text: float = 0.6,
    return_all: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | np.ndarray:
    """
    Late fusion:
        fused = alpha_text * text_probs_7 + (1 - alpha_text) * speech_probs_7

    If return_all=False (default):
        - pretty-print a table, return only fused_probs_7.

    If return_all=True:
        - return (text_probs_7, speech_probs_7, fused_probs_7, EMO7)
          with NO printing.
    """

    # ------ TEXT SIDE (GoEmotions → EMO7) ------
    probs_28, labels_28 = text_predict_proba(text, return_labels=True)
    text_probs_7 = _project_goe_to_emo7(probs_28, labels_28)  # shape (7,)

    # ------ SPEECH SIDE (already in EMO7 label space) ------
    # speech_predict returns (probs, label_names)
    speech_probs_raw, speech_labels = speech_predict(
        wav_path, top_k=len(EMO7), return_probs=True
    )

    # Reorder into EMO7 order just in case
    label_to_idx = {lab: i for i, lab in enumerate(speech_labels)}
    speech_probs_7 = np.array(
        [speech_probs_raw[label_to_idx[lab]] for lab in EMO7],
        dtype=np.float32,
    )

    # Normalize speech distribution (should already sum to ~1)
    s = speech_probs_7.sum()
    if s > 0:
        speech_probs_7 /= s

    # ------ FUSION ------
    alpha_speech = 1.0 - alpha_text
    fused_probs_7 = alpha_text * text_probs_7 + alpha_speech * speech_probs_7

    if return_all:
        # Used by emotion_router & programmatic callers
        return text_probs_7, speech_probs_7, fused_probs_7, EMO7

    # ------ Pretty-print table (CLI demo) ------
    print("\n================= FUSION RESULT =================")
    print(f'Text: "{text}"')
    print(f"Audio: {wav_path}")
    print(f"alpha_text = {alpha_text:.2f}, alpha_speech = {alpha_speech:.2f}")
    print("-------------------------------------------------")
    print("Emotion    |    Text |  Speech |   Fused")
    print("----------------------------------------------")
    for i, lab in enumerate(EMO7):
        print(
            f"{lab:<10} | "
            f"{text_probs_7[i]:7.3f} | "
            f"{speech_probs_7[i]:7.3f} | "
            f"{fused_probs_7[i]:7.3f}"
        )
    print("-------------------------------------------------")
    best_idx = int(fused_probs_7.argmax())
    print(f"FUSED PREDICTION: {EMO7[best_idx]} (p={fused_probs_7[best_idx]:.3f})")
    print("=================================================\n")

    return fused_probs_7


def pretty_print_fusion(text: str, wav_path: str, alpha_text: float = 0.6) -> None:
    """
    Simple wrapper for your existing run_fusion_infer.py script.
    """
    fuse_text_speech(text, wav_path, alpha_text=alpha_text, return_all=False)
