# text_erc/config.py

from dataclasses import dataclass

@dataclass
class TextERCConfig:
    model_name: str =  "roberta-large"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    num_labels: int = 28   # 27 emotions + neutral for GoEmotions
    seed: int = 42
