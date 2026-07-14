"""Projection head for contrastive learning.

Following the SimCLR / SupCon recipe we use a two-layer MLP projection head
to map expert features into a normalized embedding space where the
contrastive loss is applied. The projection head is *only* used during
training; at inference time we discard it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP that produces L2-normalized vectors of size ``out_dim``."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        z = self.fc2(h)
        return F.normalize(z, dim=-1)
