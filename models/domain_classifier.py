"""Domain classifier.

In CEVD the domain classifier is an MLP head that outputs a softmax over the
k pseudo-domains. The softmax probabilities are then used as fusion weights.

For HACE-VD we also expose an attention-based fusion mode in which the
classifier still produces a soft membership distribution (so that the
domain-classification loss can be applied unchanged), but in addition we
provide a separate attention block in ``hacevd_model.py`` that uses the
embedding directly to compute per-expert weights via dot-product attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainClassifier(nn.Module):
    """MLP head that outputs a softmax distribution over k pseudo-domains."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        num_domains: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_domains)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, embedding_dim) -> probability vector (B, num_domains)."""
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return torch.softmax(logits, dim=-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized scores (B, num_domains)."""
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)
