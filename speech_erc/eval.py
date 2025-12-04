# speech_erc/eval.py

import os
import numpy as np
import torch

from sklearn.metrics import f1_score, confusion_matrix

from .config import SpeechERCConfig
from .data import create_dataloaders
from .model import SpeechERCModel

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(ROOT_DIR, "checkpoints", "speech_erc", "best_speech.pt")


def _select_device():
    if torch.cuda.is_available():
        print("[speech_eval] Using device: cuda")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[speech_eval] Using device: mps")
        return torch.device("mps")
    else:
        print("[speech_eval] Using device: cpu")
        return torch.device("cpu")


def evaluate_split(split: str = "test"):
    device = _select_device()

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)

    config = SpeechERCConfig()
    for k, v in ckpt.get("config", {}).items():
        if hasattr(config, k):
            setattr(config, k, v)

    # Build dataloaders (uses manifests you already created)
    loaders = create_dataloaders(config)
    loader = loaders[split]

    # Label list (or fallback)
    default_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    label_list = ckpt.get("label_list", default_labels)
    num_labels = len(label_list)

    model = SpeechERCModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # (B,)

            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs["logits"]  # (B, num_labels)
            preds = torch.argmax(logits, dim=-1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Weighted accuracy (overall)
    wa = (all_labels == all_preds).mean()

    # Confusion matrix for UA, per-class recall
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))
    per_class_recalls = []
    for i in range(num_labels):
        denom = cm[i].sum()
        if denom == 0:
            per_class_recalls.append(0.0)
        else:
            per_class_recalls.append(cm[i, i] / denom)
    ua = float(np.mean(per_class_recalls))

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0.0)

    print(f"\n=== Speech ERC evaluation on split: {split} ===")
    print("Labels:", label_list)
    print(f"Weighted Accuracy (WA): {wa:.4f}")
    print(f"Unweighted Accuracy (UA): {ua:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("Per-class recalls:")
    for lbl, r in zip(label_list, per_class_recalls):
        print(f"  {lbl:<10} {r:.4f}")
    print("=======================================\n")


if __name__ == "__main__":
    evaluate_split("test")
