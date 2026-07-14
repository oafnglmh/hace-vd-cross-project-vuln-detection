"""Vulnerability-detection loss used by the fused ensemble output."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def weighted_vulnerability_loss(
    fused_logits: torch.Tensor,         # (B, 2)
    labels:       torch.Tensor,         # (B,) long in {0, 1}
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard cross-entropy on the fused logits (Equation (8) of CEVD).

    If ``class_weights`` is provided it is forwarded to ``F.cross_entropy``.
    The default is no weighting (we already oversample the minority class).
    """
    return F.cross_entropy(fused_logits, labels, weight=class_weights)
