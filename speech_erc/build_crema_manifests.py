# speech_erc/build_crema_manifests.py

import json
import random
from pathlib import Path
from typing import Dict, List


# Path to the AudioWAV folder (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "CREMA-D" / "AudioWAV"
OUT_DIR = PROJECT_ROOT / "speech_erc" / "manifests"


# Map from CREMA-D 3-letter emotion code to our unified label
EMOTION_MAP: Dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    # CREMA-D doesn't have surprise; we keep the label space ready anyway
}


def parse_crema_filename(path: Path) -> Dict:
    """
    Parse filenames like:
        1001_DFA_ANG_XX.wav
        1001_IEO_ANG_HI.wav
    Returns a dict with speaker_id and label.
    """
    stem = path.stem  # e.g. "1001_DFA_ANG_XX"
    parts = stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected CREMA-D filename format: {path.name}")

    speaker_id = parts[0]        # "1001"
    emotion_code = parts[2]      # "ANG", "DIS", etc.

    if emotion_code not in EMOTION_MAP:
        raise ValueError(f"Unknown emotion code {emotion_code} in {path.name}")

    label = EMOTION_MAP[emotion_code]

    return {
        "path": str(path.resolve()),
        "label": label,
        "speaker_id": speaker_id,
        "dataset": "crema_d",
    }


def build_examples() -> List[Dict]:
    wav_paths = sorted(AUDIO_DIR.glob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"No .wav files found in {AUDIO_DIR}")

    examples = []
    for p in wav_paths:
        try:
            ex = parse_crema_filename(p)
            examples.append(ex)
        except ValueError as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    print(f"Collected {len(examples)} examples from CREMA-D.")
    return examples


def split_by_speaker(examples: List[Dict], seed: int = 42):
    """
    Speaker-disjoint splits: train / val / test = 80 / 10 / 10 (by speakers).
    """
    speaker_ids = sorted({ex["speaker_id"] for ex in examples})
    random.Random(seed).shuffle(speaker_ids)

    n = len(speaker_ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_speakers = set(speaker_ids[:n_train])
    val_speakers = set(speaker_ids[n_train:n_train + n_val])
    test_speakers = set(speaker_ids[n_train + n_val:])

    splits = {"train": [], "validation": [], "test": []}

    for ex in examples:
        sid = ex["speaker_id"]
        if sid in train_speakers:
            splits["train"].append(ex)
        elif sid in val_speakers:
            splits["validation"].append(ex)
        else:
            splits["test"].append(ex)

    for name in ["train", "validation", "test"]:
        print(f"{name}: {len(splits[name])} examples")

    return splits


def write_jsonl(examples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} lines to {path}")


def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"AUDIO_DIR    = {AUDIO_DIR}")
    print(f"OUT_DIR      = {OUT_DIR}")

    examples = build_examples()
    splits = split_by_speaker(examples, seed=42)

    write_jsonl(splits["train"], OUT_DIR / "crema_d_train.jsonl")
    write_jsonl(splits["validation"], OUT_DIR / "crema_d_val.jsonl")
    write_jsonl(splits["test"], OUT_DIR / "crema_d_test.jsonl")


if __name__ == "__main__":
    main()
