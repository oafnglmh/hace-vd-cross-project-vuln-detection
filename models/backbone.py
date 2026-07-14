"""Frozen CodePTM backbone.

We keep the backbone (CodeBERT or VulBERTa) frozen during training, following
the CEVD paper. The backbone is used only to produce a stable [CLS] embedding
that subsequent components consume.

Note. We deliberately do not fine-tune the backbone for two reasons:
    1. Reproducibility w.r.t. the CEVD setup.
    2. Efficiency. Freezing the heavyweight encoder makes the ensemble of
       lightweight MLP experts the only set of parameters that needs to be
       optimized, which keeps training cost roughly constant in the number
       of experts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CodePTMBackbone(nn.Module):
    """Thin wrapper around a Hugging Face encoder that returns [CLS] embeddings."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self.embedding_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode_text(self, texts: list[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Tokenize a batch of code strings and return [CLS] embeddings.

        Returns
        -------
        embeddings : (B, D) float tensor
        """
        device = device or next(self.encoder.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = self.encoder(**encoded)
        # The [CLS] embedding is the first token of last_hidden_state.
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-tokenized inputs (used inside the train loop)."""
        if self.encoder.training is False and any(p.requires_grad for p in self.encoder.parameters()):
            # In case the user un-froze the encoder, switch it back to train mode.
            self.encoder.train()
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
