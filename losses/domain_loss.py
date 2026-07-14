"""Soft cross-entropy loss for the domain classifier.

Equation (9) of the CEVD paper: the soft-label distribution s_i is matched
against the domain-classifier softmax w_i via cross-entropy. We re-implement
it explicitly to avoid surprises from PyTorch's various semantics around
soft targets (which depend on the version).
"""

from __future__ import annotations

import torch


def soft_cross_entropy(
    pred_probs: torch.Tensor,    # (B, K) -- already softmax-normalized
    soft_targets: torch.Tensor,  # (B, K) -- sum to 1 along the K axis
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean cross-entropy with soft targets."""
    loss = -(soft_targets * torch.log(pred_probs.clamp_min(eps))).sum(dim=1)
    return loss.mean()
