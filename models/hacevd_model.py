"""End-to-end HACE-VD model.

This module assembles the building blocks (experts, domain classifier,
optional projection head, and either linear-softmax or multi-head attention
fusion) into a single ``nn.Module``. The backbone is expected to have been
applied beforehand and only embeddings are passed in.

Toggling configuration flags reproduces either the **CEVD baseline** or the
**HACE-VD** variant.

Author : implementation accompanying the HACE-VD manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .domain_classifier import DomainClassifier
from .expert import DomainExpert
from .projection_head import ProjectionHead


@dataclass
class HACEVDOutput:
    """Container for the model outputs needed by the training loop."""

    vuln_logits:     torch.Tensor              # (B, 2) fused vulnerability logits
    expert_logits:   torch.Tensor              # (B, k, 2) per-expert logits
    domain_probs:    torch.Tensor              # (B, k)   classifier softmax
    fusion_weights:  torch.Tensor              # (B, k)   weights actually used in fusion
    projections:     Optional[torch.Tensor]    # (B, k, P) projection-head outputs
    expert_features: Optional[torch.Tensor]    # (B, k, H) expert hidden features


class MultiHeadAttentionFusion(nn.Module):
    """Attention-based fusion of expert logits.

    The query is computed from the input embedding (so it depends on the
    sample's semantic content), the keys are learnable per-expert vectors,
    and the values are the experts' logits. We use ``num_heads`` parallel
    attention heads and average their outputs.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_experts: int,
        num_heads: int = 4,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = attention_dim // num_heads

        self.query_proj = nn.Linear(embedding_dim, attention_dim)
        # Learnable key vectors, one per expert.
        self.expert_keys = nn.Parameter(torch.randn(num_experts, attention_dim) * 0.02)
        # Per-head temperature softens / sharpens the attention distribution.
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        embedding: torch.Tensor,
        expert_logits: torch.Tensor,
        domain_probs: Optional[torch.Tensor] = None,
        supervision_mode: str = "none",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted fusion.

        Parameters
        ----------
        embedding     : (B, D)         input semantic embedding.
        expert_logits : (B, K, C)      per-expert classification logits.
        domain_probs  : (B, K)         domain probabilities from classifier.
        supervision_mode : str         supervision mechanism: "none", "gate", or "residual".

        Returns
        -------
        fused_logits  : (B, C)
        weights       : (B, K)        head-averaged attention weights.
        """
        B, K, C = expert_logits.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.query_proj(embedding).view(B, H, Dh)           # (B, H, Dh)
        k = self.expert_keys.view(K, H, Dh).permute(1, 0, 2)    # (H, K, Dh)

        # Dot-product attention: (B, H, K)
        attn_logits = torch.einsum("bhd,hkd->bhk", q, k) * self.scale

        if supervision_mode == "residual" and domain_probs is not None:
            # Add log(domain_probs) to attention logits before softmax
            log_domain = torch.log(domain_probs + 1e-12).unsqueeze(1) # (B, 1, K)
            attn = F.softmax(attn_logits + log_domain, dim=-1)
        else:
            attn = F.softmax(attn_logits, dim=-1)

        # Average across heads to recover per-expert weights.
        weights = attn.mean(dim=1)                              # (B, K)

        if supervision_mode == "gate" and domain_probs is not None:
            gated_weights = domain_probs * weights
            weights = gated_weights / (gated_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # Weighted sum of expert logits.
        fused = torch.einsum("bk,bkc->bc", weights, expert_logits)
        return fused, weights


class HACEVDModel(nn.Module):
    """End-to-end HACE-VD model assembly.

    The backbone is treated as an external feature extractor: ``forward()``
    expects already-computed embeddings of shape ``(B, embedding_dim)``.
    This decoupling keeps the model small and allows offline embedding
    caching for cheap experimentation.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_domains: int,
        expert_hidden: int = 768,
        expert_dropout: float = 0.1,
        expert_activation: str = "tanh",
        domain_hidden: int = 768,
        domain_dropout: float = 0.1,
        fusion_type: str = "multihead_attention",
        attention_heads: int = 4,
        attention_dim: int = 128,
        use_projection: bool = True,
        projection_hidden: int = 256,
        projection_dim: int = 128,
        num_classes: int = 2,
        attention_supervision: str = "none",
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.fusion_type = fusion_type
        self.attention_supervision = attention_supervision

        # ---- k parallel experts ------------------------------------------------
        self.experts = nn.ModuleList(
            DomainExpert(
                embedding_dim=embedding_dim,
                hidden_dim=expert_hidden,
                dropout=expert_dropout,
                activation=expert_activation,
                num_classes=num_classes,
            )
            for _ in range(num_domains)
        )

        # ---- domain classifier -------------------------------------------------
        self.domain_classifier = DomainClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=domain_hidden,
            num_domains=num_domains,
            dropout=domain_dropout,
        )

        # ---- optional contrastive projection ----------------------------------
        self.use_projection = use_projection
        if use_projection:
            self.projection = ProjectionHead(
                in_dim=expert_hidden,
                hidden_dim=projection_hidden,
                out_dim=projection_dim,
            )
        else:
            self.projection = None

        # ---- fusion module -----------------------------------------------------
        if fusion_type == "linear_softmax":
            self.fusion = None                              # CEVD's simple weighted sum
        elif fusion_type == "multihead_attention":
            self.fusion = MultiHeadAttentionFusion(
                embedding_dim=embedding_dim,
                num_experts=num_domains,
                num_heads=attention_heads,
                attention_dim=attention_dim,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(self, embeddings: torch.Tensor) -> HACEVDOutput:
        """Run the ensemble on a batch of frozen embeddings.

        Parameters
        ----------
        embeddings : (B, D) float tensor produced by the frozen backbone.

        Returns
        -------
        HACEVDOutput with the fields populated.
        """
        B, _ = embeddings.shape
        K = self.num_domains

        # Per-expert logits and (optionally) features for contrastive loss.
        expert_features = []
        expert_logits   = []
        for expert in self.experts:
            feat = expert.feature(embeddings)               # (B, H)
            logit = expert.out_proj(expert.dropout(feat))   # (B, C)
            expert_features.append(feat)
            expert_logits.append(logit)
        expert_features = torch.stack(expert_features, dim=1)   # (B, K, H)
        expert_logits   = torch.stack(expert_logits, dim=1)     # (B, K, C)

        # Domain classifier outputs (soft membership distribution).
        domain_probs = self.domain_classifier(embeddings)       # (B, K)

        # Fusion.
        if self.fusion_type == "linear_softmax":
            fusion_weights = domain_probs                       # (B, K)
            fused = torch.einsum("bk,bkc->bc", fusion_weights, expert_logits)
        else:
            fused, fusion_weights = self.fusion(
                embeddings,
                expert_logits,
                domain_probs=domain_probs,
                supervision_mode=self.attention_supervision,
            )

        # Projections (optional).
        if self.projection is not None:
            B_, K_, H = expert_features.shape
            projections = self.projection(expert_features.reshape(B_ * K_, H))
            projections = projections.view(B_, K_, -1)
        else:
            projections = None

        return HACEVDOutput(
            vuln_logits=fused,
            expert_logits=expert_logits,
            domain_probs=domain_probs,
            fusion_weights=fusion_weights,
            projections=projections,
            expert_features=expert_features,
        )
