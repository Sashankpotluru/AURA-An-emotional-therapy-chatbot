# text_erc/model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .config import TextERCConfig

class TextERCModel(nn.Module):
    def __init__(self, config: TextERCConfig):
        super().__init__()
        self.config = config

        hf_config = AutoConfig.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(config.model_name, config=hf_config)

        hidden_size = hf_config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Take [CLS] / pooled representation
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Use first token's hidden state
            pooled = outputs.last_hidden_state[:, 0]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # Multi-label BCE with logits
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return {"loss": loss, "logits": logits}
