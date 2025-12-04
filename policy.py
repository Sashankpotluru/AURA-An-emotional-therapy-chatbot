# policy.py
"""
Policy & Safety controller.

Takes unified 7-way emotion distribution (neutral, happy, sad, angry, fear,
disgust, surprise) and returns:
- StyleCode (tone, verbosity, directness)
- crisis flag (bool)
- top emotions (for logging / prompting)
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

from emotion_router import BASE_EMOTIONS, EmotionResult


@dataclass
class StyleCode:
    tone: str          # "warm" | "neutral" | "firm"
    verbosity: str     # "short" | "medium" | "long"
    directness: str    # "reflective" | "balanced" | "directive"


@dataclass
class PolicyResult:
    style: StyleCode
    crisis: bool
    top_emotions: List[Tuple[str, float]]   # sorted list
    debug_info: Dict


def _simple_crisis_detector(emo: Dict[str, float]) -> bool:
    """
    VERY conservative heuristic.
    We will refine this later, but for now:
    - High sadness + fear/anger → possible distress
    - Very high disgust or fear → also flag
    """
    sad = emo.get("sad", 0.0)
    fear = emo.get("fear", 0.0)
    angry = emo.get("angry", 0.0)
    disgust = emo.get("disgust", 0.0)

    if sad > 0.6 and (fear > 0.4 or angry > 0.4):
        return True
    if disgust > 0.7 or fear > 0.7:
        return True
    return False


def _style_from_emotions(emo: Dict[str, float]) -> StyleCode:
    """
    Map base emotions → conversational style.
    This is heuristic and you'll describe it clearly in the paper.
    """
    # find top emotion
    top_label = max(emo.items(), key=lambda x: x[1])[0]

    if top_label in {"sad", "fear"}:
        # Supportive & gentle
        return StyleCode(
            tone="warm",
            verbosity="long",
            directness="reflective",
        )
    elif top_label in {"angry", "disgust"}:
        # Calm but firm, avoid escalation
        return StyleCode(
            tone="firm",
            verbosity="medium",
            directness="balanced",
        )
    elif top_label in {"happy", "surprise"}:
        # Positive, energetic but not over the top
        return StyleCode(
            tone="warm",
            verbosity="medium",
            directness="balanced",
        )
    else:  # neutral or mixed
        return StyleCode(
            tone="neutral",
            verbosity="short",
            directness="balanced",
        )


def apply_policy(result: EmotionResult) -> PolicyResult:
    """
    Take an EmotionResult from emotion_router.analyze(...)
    and produce style + crisis flag.
    """
    probs = result.probs
    labels = result.labels

    emo_dict = {lab: float(p) for lab, p in zip(labels, probs)}
    top_sorted = sorted(emo_dict.items(), key=lambda x: x[1], reverse=True)

    crisis = _simple_crisis_detector(emo_dict)
    style = _style_from_emotions(emo_dict)

    return PolicyResult(
        style=style,
        crisis=crisis,
        top_emotions=top_sorted[:3],
        debug_info={"mode": result.mode, "emo_dict": emo_dict},
    )
