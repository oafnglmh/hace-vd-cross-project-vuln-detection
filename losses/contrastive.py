"""Supervised contrastive loss for expert-aware representation learning.

This is the SupCon loss of Khosla et al. (NeurIPS 2020), adapted to the
HACE-VD setting. The positives of an anchor sample are the other samples in
the same mini-batch that (a) share the same hard pseudo-domain assignment,
**and** (b) share the same vulnerability label. This double constraint pushes
the projection space to encode both domain identity and the binary task
label, which is exactly the kind of structure we want each expert to
specialize on.

Mathematically:
    L_sc = - 1/|P(i)|  sum_{p in P(i)}  log(  exp(z_i . z_p / T)
                                              / sum_{a != i} exp(z_i . z_a / T)  )

where P(i) is the positive set of anchor i.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    projections: torch.Tensor,        # (B, P) -- already L2 normalized
    labels:      torch.Tensor,        # (B,)   -- vulnerability label
    domains:     torch.Tensor,        # (B,)   -- hard pseudo-domain index
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute the supervised contrastive loss.

    Returns
    -------
    loss : scalar tensor.

    Notes
    -----
    * Projections are assumed to be L2-normalized.
    * If a sample has no positives in the batch (this can happen with small
      batches), it is silently dropped from the loss so that we never divide
      by zero.
    """
    device = projections.device
    B = projections.size(0)

    if B < 2:
        return torch.zeros((), device=device)

    # Pairwise similarity matrix (B, B)
    logits = projections @ projections.T / max(temperature, 1e-8)

    # Numerical stability trick from SupCon.
    logits_max, _ = logits.detach().max(dim=1, keepdim=True)
    logits = logits - logits_max

    # Mask for self (always excluded from numerator and denominator).
    eye = torch.eye(B, dtype=torch.bool, device=device)

    # Positive mask: same label AND same domain, excluding self.
    same_label  = labels.unsqueeze(0)  == labels.unsqueeze(1)
    same_domain = domains.unsqueeze(0) == domains.unsqueeze(1)
    positive_mask = same_label & same_domain & ~eye      # (B, B) bool

    # Denominator: all non-self pairs.
    denom_mask = ~eye

    exp_logits = torch.exp(logits) * denom_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Mean of log-prob over positives, per anchor.
    pos_counts = positive_mask.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1) / pos_counts

    # Only count anchors that actually have positives.
    has_positives = positive_mask.any(dim=1)
    if not has_positives.any():
        return torch.zeros((), device=device)
    loss = -mean_log_prob_pos[has_positives].mean()
    return loss
