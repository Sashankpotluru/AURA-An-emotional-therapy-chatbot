# fusion/label_mapping.py

import numpy as np
from typing import List, Dict

# Map from *speech label name* → list of GoEmotions labels that belong there.
# We include all speech labels we saw: angry, disgust, fear, happy, neutral, sad, surprise.
GOEMO_GROUPS: Dict[str, List[str]] = {
    "angry":   ["anger", "annoyance"],
    "disgust": ["disgust", "disapproval"],
    "fear":    ["fear", "nervousness"],
    "happy":   [
        "joy", "amusement", "excitement", "pride", "love",
        "admiration", "relief", "gratitude", "optimism", "caring",
    ],
    "neutral": [
        "neutral", "realization", "curiosity", "desire",
    ],
    "sad":     [
        "sadness", "disappointment", "embarrassment", "grief", "remorse",
    ],
    # Keep surprise as its own bucket
    "surprise": ["surprise"],
}


def goemotions_to_speech_space(
    probs_28: np.ndarray,
    label_list_28: List[str],
    speech_labels: List[str],
) -> np.ndarray:
    """
    Project 28-d GoEmotions probabilities into the space of `speech_labels`.

    For each speech label (e.g., 'angry', 'happy', 'surprise'), we average
    the probabilities of its associated GoEmotions labels.

    Returns: np.ndarray of shape (len(speech_labels),)
    """
    scores = []

    for s_label in speech_labels:
        names = GOEMO_GROUPS.get(s_label, [])
        # Find indices in GoEmotions label list that correspond to this speech group
        idxs = [i for i, name in enumerate(label_list_28) if name in names]

        if not idxs:
            # No matching labels found → give 0 (we'll renormalize later)
            scores.append(0.0)
        else:
            vals = probs_28[idxs]
            scores.append(float(np.mean(vals)))

    scores = np.array(scores, dtype=float)
    s = scores.sum()
    if s > 0:
        scores /= s
    else:
        # Fallback: uniform distribution
        scores = np.ones_like(scores) / len(scores)

    return scores
