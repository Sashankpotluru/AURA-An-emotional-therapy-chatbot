# speech_erc/infer.py

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.nn.functional import softmax
import torchaudio
import soundfile as sf

from .config import SpeechERCConfig
from .model import SpeechERCModel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "speech_erc", "best_speech.pt")


# ---- device selection ----
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[speech_infer] Using device: cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[speech_infer] Using device: mps")
else:
    DEVICE = torch.device("cpu")
    print("[speech_infer] Using device: cpu")


_model: SpeechERCModel = None
_config: SpeechERCConfig = None
_label_names: List[str] = []


def _load_checkpoint_if_needed():
    global _model, _config, _label_names

    if _model is not None:
        return

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Speech checkpoint not found at {CKPT_PATH}. "
            f"Train the model first (python3 -m speech_erc.train)."
        )

    print(f"[speech_infer] Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    _config = SpeechERCConfig()
    for k, v in ckpt.get("config", {}).items():
        if hasattr(_config, k):
            setattr(_config, k, v)

    _label_names = ckpt["label_names"]

    _model = SpeechERCModel(_config).to(DEVICE)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    print("[speech_infer] Model loaded.")


def _load_and_preprocess_audio(path: str) -> torch.Tensor:
    """
    Load audio from file, convert to mono, resample, truncate to max_duration.
    Returns 1D float32 tensor (T,)
    """
    data, sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)

    wav = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)

    if sr != _config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, _config.sample_rate)
        wav = resampler(wav)

    max_len = int(_config.max_duration_secs * _config.sample_rate)
    wav = wav[:, :max_len]

    return wav.squeeze(0)


def predict(path: str, top_k: int = 3, return_probs: bool = False):
    """
    Run speech emotion prediction on a single audio file.

    If return_probs=False (default): prints top-k emotions (like now).
    If return_probs=True: returns (probs, label_names) for fusion.
    """
    _load_checkpoint_if_needed()

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

    waveform = _load_and_preprocess_audio(str(audio_path))  # (T,)
    input_values = waveform.unsqueeze(0).to(DEVICE)         # (1, T)

    attn_mask = torch.ones_like(input_values, dtype=torch.long)

    with torch.no_grad():
        outputs = _model(
            input_values=input_values,
            attention_mask=attn_mask,
            labels=None,
        )
        logits = outputs["logits"][0]  # (num_labels,)
        probs = softmax(logits, dim=-1).cpu().numpy()  # (num_labels,)

    if return_probs:
        # For fusion: return full posterior + label order
        return probs, _label_names

    # For CLI / debugging: pretty-print top-k
    top_k = min(top_k, len(_label_names))
    top_idx = probs.argsort()[::-1][:top_k]

    print(f"\nAudio file: {audio_path}\n")
    print("Top predicted emotions:")
    for idx in top_idx:
        label = _label_names[idx]
        score = probs[idx]
        print(f"  {label:<10} {score:.3f}")
    print("")

