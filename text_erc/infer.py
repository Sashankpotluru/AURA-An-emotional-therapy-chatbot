# text_erc/infer.py

import os
import torch
from torch.nn.functional import sigmoid
from transformers import AutoTokenizer

from .config import TextERCConfig
from .model import TextERCModel


# Figure out project root: .../aura/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "text_erc", "best.pt")


# ---- device selection ----
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[infer] Using device: cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[infer] Using device: mps")
else:
    DEVICE = torch.device("cpu")
    print("[infer] Using device: cpu")


# Global objects loaded once
_model = None
_tokenizer = None
_label_list = None
_config = None


def _load_model_if_needed():
    global _model, _tokenizer, _label_list, _config

    if _model is not None:
        return

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CKPT_PATH}. "
            f"Make sure you trained the model and best.pt exists."
        )

    print(f"[infer] Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # Rebuild config
    _config = TextERCConfig()
    # overwrite with saved values (optional but keeps things consistent)
    for k, v in ckpt.get("config", {}).items():
        if hasattr(_config, k):
            setattr(_config, k, v)

    _label_list = ckpt["label_list"]

    # Build model & load weights
    _model = TextERCModel(_config).to(DEVICE)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    # Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(_config.model_name)
    print("[infer] Model and tokenizer loaded.")


# text_erc/infer.py (update this function)

# text_erc/infer.py  (only this part needs to be updated)

def predict(text: str, top_k: int = 5):
    """
    Pretty-print top_k emotions for a single text.
    This is the CLI/debug version used by run_infer.py.
    """
    _load_model_if_needed()

    enc = _tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=_config.max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][0]          # (num_labels,)
        probs = sigmoid(logits).cpu().numpy()  # (num_labels,)

    print(f"\nInput text: {text}\n")
    top_k = min(top_k, len(_label_list))
    top_idx = probs.argsort()[::-1][:top_k]

    print("Top predicted emotions:")
    for idx in top_idx:
        label = _label_list[idx]
        score = probs[idx]
        print(f"  {label:<15} {score:.3f}")
    print("")


def predict_proba(text: str, return_labels: bool = False):
    """
    Return full 28-dim probabilities for a single text.
    - If return_labels=False  -> returns probs (np.ndarray, shape [28])
    - If return_labels=True   -> returns (probs, label_list)
    """
    _load_model_if_needed()

    enc = _tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=_config.max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][0]          # (num_labels,)
        probs = sigmoid(logits).cpu().numpy()  # (num_labels,)

    if return_labels:
        return probs, _label_list

    return probs




