# text_erc/data.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from .config import TextERCConfig

def load_goemotions_splits(config: TextERCConfig):
    """
    Loads GoEmotions from HF and returns (dataset, label_list).
    dataset has 'train', 'validation', 'test'.
    """
    dataset = load_dataset("go_emotions")
    label_list = dataset["train"].features["labels"].feature.names  # 28 labels

    assert len(label_list) == config.num_labels, \
        f"Config num_labels={config.num_labels} but dataset has {len(label_list)}"

    return dataset, label_list

def prepare_multi_label(dataset_split, num_labels: int):
    """
    Convert 'labels' (list of indices) to multi-hot vectors.
    """
    def _to_multilabel(example):
        ml = [0] * num_labels
        for idx in example["labels"]:
            ml[idx] = 1
        example["multi_labels"] = ml
        return example

    return dataset_split.map(_to_multilabel)

def tokenize_dataset(dataset_splits, config: TextERCConfig, model_name: str):
    """
    dataset_splits: dict with keys 'train', 'validation', 'test',
    each value is a HF Dataset that already has 'multi_labels'.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
        )

    tokenized = {}
    for split_name, split_dataset in dataset_splits.items():
        ds_tok = split_dataset.map(_tokenize, batched=True)
        ds_tok.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "multi_labels"],
        )
        tokenized[split_name] = ds_tok

    return tokenized, tokenizer

def create_dataloaders(tokenized_dataset, config: TextERCConfig) -> Dict[str, DataLoader]:
    def _make_loader(split_name):
        return DataLoader(
            tokenized_dataset[split_name],
            batch_size=config.batch_size,
            shuffle=True if split_name == "train" else False,
        )

    return {
        "train": _make_loader("train"),
        "validation": _make_loader("validation"),
        "test": _make_loader("test"),
    }