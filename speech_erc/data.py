# speech_erc/data.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import soundfile as sf

from .config import SpeechERCConfig


def load_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class SERDataset(Dataset):
    """
    Simple dataset for CREMA-D (and later RAVDESS/IEMOCAP) using JSONL manifests.

    Each example:
      {
        "path": "/abs/path/to/audio.wav",
        "label": "angry",
        "speaker_id": "1001",
        "dataset": "crema_d"
      }
    """

    def __init__(self, manifest_path: Path, config: SpeechERCConfig):
        self.config = config
        self.items = load_jsonl(manifest_path)

        # Map label -> id based on config.label_names
        self.label2id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.config.label_names)
        }

    def __len__(self) -> int:
        return len(self.items)

    def _load_waveform(self, wav_path: str) -> torch.Tensor:
        # Load audio with soundfile: data shape (T,) or (T, C), sr = sample rate
        data, sr = sf.read(wav_path)  # data is a NumPy array

        # Convert to mono: average channels if multi-channel
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Convert to float32 tensor, shape (1, T)
        wav = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.sample_rate,
            )
            wav = resampler(wav)

        # Truncate to max_duration_secs
        max_len = int(self.config.max_duration_secs * self.config.sample_rate)
        wav = wav[:, :max_len]  # (1, T)

        return wav.squeeze(0)  # (T,)


    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        path = item["path"]
        label_name = item["label"]

        if label_name not in self.label2id:
            raise KeyError(f"Unknown label '{label_name}' not in config.label_names")

        label_id = self.label2id[label_name]
        waveform = self._load_waveform(path)  # 1D tensor (T,)

        return {
            "input_values": waveform,
            "label": label_id,
        }


def lengths_to_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Convert a vector of sequence lengths into a boolean mask of shape (B, T).
    True (or 1) where there is real data, False (or 0) where it's padding.
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(lengths.max().item())

    # (B, T): [0, 1, ..., max_len-1] < length
    idxs = torch.arange(max_len).expand(batch_size, -1)
    mask = idxs < lengths.unsqueeze(1)
    return mask


def ser_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to:
      - pad variable-length waveforms
      - create attention_mask for Wav2Vec2
    """
    waveforms = [b["input_values"] for b in batch]  # list of 1D tensors
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True)  # (B, T)

    attn_mask = lengths_to_mask(lengths, max_len=padded.shape[1])

    return {
        "input_values": padded,          # (B, T)
        "attention_mask": attn_mask,     # (B, T)
        "labels": labels,                # (B,)
    }


def create_dataloaders(config: SpeechERCConfig) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for CREMA-D using the manifests we wrote.
    """
    project_root = Path(__file__).resolve().parents[1]
    manifest_dir = project_root / "speech_erc" / "manifests"

    train_manifest = manifest_dir / "crema_d_train.jsonl"
    val_manifest = manifest_dir / "crema_d_val.jsonl"
    test_manifest = manifest_dir / "crema_d_test.jsonl"

    train_ds = SERDataset(train_manifest, config)
    val_ds = SERDataset(val_manifest, config)
    test_ds = SERDataset(test_manifest, config)

    def _make_loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=ser_collate_fn,
        )

    return {
        "train": _make_loader(train_ds, shuffle=True),
        "validation": _make_loader(val_ds, shuffle=False),
        "test": _make_loader(test_ds, shuffle=False),
    }
