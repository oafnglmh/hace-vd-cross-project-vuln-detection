"""Domain expert head.

Each expert is a small MLP that takes a frozen code embedding and produces a
vulnerability logit (2-way: non-vulnerable / vulnerable). The architecture
mirrors the classification head used by RoBERTa-style models so that we keep
the CEVD setup as the controlled baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DomainExpert(nn.Module):
    """MLP head used as a domain expert."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        activation: str = "tanh",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if activation == "tanh":
            act_module: nn.Module = nn.Tanh()
        elif activation == "relu":
            act_module = nn.ReLU()
        elif activation == "gelu":
            act_module = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dense = nn.Linear(embedding_dim, hidden_dim)
        self.activation = act_module
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, embedding_dim) -> logits (B, num_classes)."""
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.out_proj(x)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-output hidden representation (B, hidden_dim).

        This is used by the contrastive loss so that the projection head
        operates on the *expert*'s internal representation rather than the
        raw backbone embedding.
        """
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        return x
