"""End-to-end HACE-VD model — ban da tich hop 4 co che fusion.

Thay cho models/hacevd_model.py cu. Diem khac so ban cu:
  * import build_fusion tu fusion_hace
  * ho tro fusion_type: linear_softmax | tempered_gate | residual_gate |
    gated_residual | multihead_attention (giu tuong thich nguoc)
  * forward() truyen domain_probs vao module fusion moi
  * expose fusion_gate_reg() de trainer cong penalty cua quality-gate

Cac file .domain_classifier / .expert / .projection_head giu NGUYEN, khong sua.
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
from .fusion_hace import build_fusion


@dataclass
class HACEVDOutput:
    vuln_logits:     torch.Tensor
    expert_logits:   torch.Tensor
    domain_probs:    torch.Tensor
    fusion_weights:  torch.Tensor
    projections:     Optional[torch.Tensor]
    expert_features: Optional[torch.Tensor]


class MultiHeadAttentionFusion(nn.Module):
    """Giu lai khoi attention THUAN cu de tuong thich nguoc (fusion_type=
    'multihead_attention'). KHONG dung domain_probs -> chinh la ban gay ra
    van de o cac luot truoc. Chi giu cho ablation so sanh."""

    def __init__(self, embedding_dim, num_experts, num_heads=4, attention_dim=128):
        super().__init__()
        assert attention_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = attention_dim // num_heads
        self.query_proj = nn.Linear(embedding_dim, attention_dim)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, attention_dim) * 0.02)
        self.scale = self.head_dim ** -0.5

    def forward(self, embedding, expert_logits):
        B, K, C = expert_logits.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.query_proj(embedding).view(B, H, Dh)
        k = self.expert_keys.view(K, H, Dh).permute(1, 0, 2)
        attn = torch.einsum("bhd,hkd->bhk", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        weights = attn.mean(dim=1)
        fused = torch.einsum("bk,bkc->bc", weights, expert_logits)
        return fused, weights


class HACEVDModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_domains: int,
        expert_hidden: int = 768,
        expert_dropout: float = 0.1,
        expert_activation: str = "tanh",
        domain_hidden: int = 768,
        domain_dropout: float = 0.1,
        fusion_type: str = "gated_residual",
        attention_heads: int = 4,
        attention_dim: int = 128,
        use_projection: bool = True,
        projection_hidden: int = 256,
        projection_dim: int = 128,
        num_classes: int = 2,
        # tham so fusion moi (doc tu cfg["fusion"]):
        tau_init: float = 1.0,
        alpha_init: float = 0.0,
        gate_l2: float = 1e-3,
        attention_supervision: str = "none",
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.fusion_type = fusion_type
        self.attention_supervision = attention_supervision


        self.experts = nn.ModuleList(
            DomainExpert(embedding_dim=embedding_dim, hidden_dim=expert_hidden,
                         dropout=expert_dropout, activation=expert_activation,
                         num_classes=num_classes)
            for _ in range(num_domains)
        )
        self.domain_classifier = DomainClassifier(
            embedding_dim=embedding_dim, hidden_dim=domain_hidden,
            num_domains=num_domains, dropout=domain_dropout)

        self.use_projection = use_projection
        self.projection = (ProjectionHead(in_dim=expert_hidden, hidden_dim=projection_hidden,
                                          out_dim=projection_dim) if use_projection else None)

        # ---- fusion ----
        if fusion_type == "multihead_attention":
            self.fusion = MultiHeadAttentionFusion(embedding_dim, num_domains,
                                                   attention_heads, attention_dim)
            self._fusion_takes_domain = False
        elif fusion_type == "linear_softmax":
            self.fusion = None
            self._fusion_takes_domain = False
        else:
            self.fusion = build_fusion(
                fusion_type, embedding_dim, num_domains,
                num_heads=attention_heads, attention_dim=attention_dim,
                tau_init=tau_init, alpha_init=alpha_init, gate_l2=gate_l2)
            self._fusion_takes_domain = True

    def fusion_gate_reg(self) -> torch.Tensor:
        """Penalty cua quality-gate (0 neu fusion khong co gate). Trainer cong vao loss."""
        if self.fusion is not None and hasattr(self.fusion, "gate_reg"):
            return self.fusion.gate_reg()
        # tra ve 0 dung device
        p = next(self.parameters())
        return torch.zeros((), device=p.device, dtype=p.dtype)

    def forward(self, embeddings: torch.Tensor) -> HACEVDOutput:
        expert_features, expert_logits = [], []
        for expert in self.experts:
            feat = expert.feature(embeddings)
            logit = expert.out_proj(expert.dropout(feat))
            expert_features.append(feat)
            expert_logits.append(logit)
        expert_features = torch.stack(expert_features, dim=1)  # (B,K,H)
        expert_logits = torch.stack(expert_logits, dim=1)      # (B,K,C)

        domain_probs = self.domain_classifier(embeddings)      # (B,K)

        if self.fusion_type == "linear_softmax":
            fusion_weights = domain_probs
            fused = torch.einsum("bk,bkc->bc", fusion_weights, expert_logits)
        elif self._fusion_takes_domain:
            fused, fusion_weights = self.fusion(embeddings, expert_logits, domain_probs)
        else:  # multihead_attention thuan
            fused, fusion_weights = self.fusion(embeddings, expert_logits)

        if self.projection is not None:
            B_, K_, H = expert_features.shape
            projections = self.projection(expert_features.reshape(B_ * K_, H)).view(B_, K_, -1)
        else:
            projections = None

        return HACEVDOutput(
            vuln_logits=fused, expert_logits=expert_logits, domain_probs=domain_probs,
            fusion_weights=fusion_weights, projections=projections,
            expert_features=expert_features)
