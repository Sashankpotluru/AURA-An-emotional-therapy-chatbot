# speech_erc/config.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpeechERCConfig:
    # Audio model (you can later switch to a fine-tuned SER checkpoint if you like)
    model_name: str = "facebook/wav2vec2-base"

    # Audio settings
    sample_rate: int = 16000
    max_duration_secs: float = 5.0  # clips longer than this will be truncated

    # Label space (7-way; CREMA-D uses 6 of these, no 'surprise')
    label_names: List[str] = field(
        default_factory=lambda: [
            "neutral",
            "happy",
            "sad",
            "angry",
            "fear",
            "disgust",
            "surprise",
        ]
    )

    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1

    seed: int = 42

    @property
    def num_labels(self) -> int:
        return len(self.label_names)
