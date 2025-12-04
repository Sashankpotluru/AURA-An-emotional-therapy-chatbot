# text_erc/train.py

import os
import math
import random
import numpy as np
from typing import Dict

import torch
from torch.optim import AdamW
from torch.nn.functional import sigmoid
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score

from .config import TextERCConfig
from .data import (
    load_goemotions_splits,
    prepare_multi_label,
    tokenize_dataset,
    create_dataloaders,
)
from .model import TextERCModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(logits, labels, threshold: float = 0.3) -> Dict[str, float]:
    """
    logits: (N, num_labels)
    labels: (N, num_labels) multi-hot
    """
    probs = sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= threshold).astype(int)
    labels = labels.numpy()

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    samples_acc = (preds == labels).all(axis=1).mean()

    return {
        "macro_f1": macro_f1,
        "sample_accuracy": samples_acc,
    }


def train_text_erc():
    config = TextERCConfig()
    set_seed(config.seed)

    # ----- device selection (update this part) -----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    # 1) Load dataset
    raw_dataset, label_list = load_goemotions_splits(config)
    print("Labels:", label_list)

    # 2) Add multi-label vectors
    processed = {}
    for split in ["train", "validation", "test"]:
        processed[split] = prepare_multi_label(raw_dataset[split], config.num_labels)

    # 3) Tokenize
    tokenized_dataset, tokenizer = tokenize_dataset(processed, config, config.model_name)

    # 4) Dataloaders
    loaders = create_dataloaders(tokenized_dataset, config)

    # 5) Model & optimizer
    model = TextERCModel(config).to(device)

    total_steps = math.ceil(
        len(loaders["train"]) / config.gradient_accumulation_steps
    ) * config.num_epochs

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = float("-inf")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(root_dir, "checkpoints", "text_erc")
    os.makedirs(save_dir, exist_ok=True)
    print("Checkpoints will be saved to:", save_dir)

    global_step = 0

    # 6) Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(loaders["train"]):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["multi_labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch+1}, step {step+1}, loss={avg_loss:.4f}")

        # 7) Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in loaders["validation"]:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["multi_labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"].cpu()

                all_logits.append(logits)
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_metrics(all_logits, all_labels)
        print(f"Epoch {epoch+1} validation: Macro-F1={metrics['macro_f1']:.4f}, "
              f"Sample-Acc={metrics['sample_accuracy']:.4f}")

        # Save best checkpoint
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            ckpt_path = os.path.join(save_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "label_list": label_list,
                },
                ckpt_path,
            )
            print(f"New best model saved at {ckpt_path} with Macro-F1={best_f1:.4f}")

    print("Training done. Best Macro-F1:", best_f1)


if __name__ == "__main__":
    train_text_erc()
