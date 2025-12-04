# speech_erc/model.py

from typing import Optional, Dict

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

from .config import SpeechERCConfig


class SpeechERCModel(nn.Module):
    """
    Wav2Vec2 backbone + pooled embedding + linear classifier.
    """

    def __init__(self, config: SpeechERCConfig):
        super().__init__()
        self.config = config

        hf_config = Wav2Vec2Config.from_pretrained(config.model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(config.model_name, config=hf_config)

        hidden_size = hf_config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def forward(
        self,
        input_values: torch.Tensor,      # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        labels: Optional[torch.Tensor] = None,          # (B,)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
          - loss (optional)
          - logits: (B, num_labels)
        """
        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        # outputs.last_hidden_state: (B, T, H)
        hidden_states = outputs.last_hidden_state

        # Simple mean-pooling over time dimension
        pooled = hidden_states.mean(dim=1)  # (B, H)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
