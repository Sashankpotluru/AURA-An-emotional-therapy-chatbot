# speech_erc/train.py

import os
import math
import random
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.functional import softmax
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score, recall_score

from .config import SpeechERCConfig
from .data import create_dataloaders
from .model import SpeechERCModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_speech(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    logits: (N, num_labels)
    labels: (N,)
    """
    probs = softmax(logits, dim=-1).cpu().numpy()
    preds = probs.argmax(axis=-1)
    labels_np = labels.cpu().numpy()

    wa = accuracy_score(labels_np, preds)                    # weighted accuracy
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    ua = recall_score(labels_np, preds, average="macro", zero_division=0)  # unweighted acc

    return {
        "wa": wa,
        "macro_f1": macro_f1,
        "ua": ua,
    }


def train_speech_erc():
    config = SpeechERCConfig()
    set_seed(config.seed)

    # ---- device selection ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # ---- data ----
    loaders = create_dataloaders(config)

    # ---- model / optimizer / scheduler ----
    model = SpeechERCModel(config).to(device)

    total_steps = math.ceil(
        len(loaders["train"]) / config.gradient_accumulation_steps
    ) * config.num_epochs

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # ---- checkpoint dir ----
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(root_dir, "checkpoints", "speech_erc")
    os.makedirs(save_dir, exist_ok=True)
    print("Checkpoints will be saved to:", save_dir)

    best_macro_f1 = float("-inf")
    global_step = 0

    # ---- training loop ----
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}, step {step+1}, loss={avg_loss:.4f}")

        # ---- validation ----
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loaders["validation"]:
                input_values = batch["input_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    labels=None,
                )
                logits = outputs["logits"]

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_metrics_speech(all_logits, all_labels)
        print(
            f"Epoch {epoch+1} validation: "
            f"WA={metrics['wa']:.4f}, "
            f"UA={metrics['ua']:.4f}, "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )

        # ---- save best checkpoint ----
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            ckpt_path = os.path.join(save_dir, "best_speech.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "label_names": config.label_names,
                },
                ckpt_path,
            )
            print(f"New best speech model saved at {ckpt_path} with Macro-F1={best_macro_f1:.4f}")

    print("Training done. Best Macro-F1:", best_macro_f1)


if __name__ == "__main__":
    train_speech_erc()
