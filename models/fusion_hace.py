"""Fusion co che cho HACE-VD: khac phuc hai diem yeu cua CEVD.
Dat file nay vao ``models/fusion_hace.py`` (thay cho fusion_residual.py cu).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def _softmax_fuse(weights: torch.Tensor, expert_logits: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bk,bkc->bc", weights, expert_logits)

class TemperedGateFusion(nn.Module):
    def __init__(self, num_experts: int, tau_init: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.eps = eps
        self.log_tau = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(tau_init)))]))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp_min(1e-3)

    def forward(self, embedding, expert_logits, domain_probs):
        prior = torch.log(domain_probs.clamp_min(self.eps))
        weights = F.softmax(prior / self.tau, dim=-1)
        return _softmax_fuse(weights, expert_logits), weights

class _AttentionLogits(nn.Module):
    def __init__(self, embedding_dim, num_experts, num_heads=4, attention_dim=128):
        super().__init__()
        assert attention_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = attention_dim // num_heads
        self.query_proj = nn.Linear(embedding_dim, attention_dim)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, attention_dim) * 0.02)
        self.scale = self.head_dim ** -0.5

    def forward(self, embedding):
        B = embedding.shape[0]
        H, Dh, K = self.num_heads, self.head_dim, self.num_experts
        q = self.query_proj(embedding).view(B, H, Dh)
        k = self.expert_keys.view(K, H, Dh).permute(1, 0, 2)
        logits = torch.einsum("bhd,hkd->bhk", q, k) * self.scale
        return logits.mean(dim=1)  # (B, K)

class ResidualGateFusion(nn.Module):
    def __init__(self, embedding_dim, num_experts, num_heads=4, attention_dim=128,
                 tau_init=1.0, alpha_init=0.0, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.attn = _AttentionLogits(embedding_dim, num_experts, num_heads, attention_dim)
        self.log_tau = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(tau_init)))]))
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))

    @property
    def tau(self):
        return self.log_tau.exp().clamp_min(1e-3)

    def forward(self, embedding, expert_logits, domain_probs):
        prior = torch.log(domain_probs.clamp_min(self.eps))
        attn = self.attn(embedding)
        weights = F.softmax(prior / self.tau + self.alpha * attn, dim=-1)
        return _softmax_fuse(weights, expert_logits), weights

class GatedResidualFusion(nn.Module):
    def __init__(self, embedding_dim, num_experts, num_heads=4, attention_dim=128,
                 tau_init=1.0, alpha_init=0.0, gate_l2=1e-3, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.num_experts = num_experts
        self.gate_l2 = gate_l2
        self.attn = _AttentionLogits(embedding_dim, num_experts, num_heads, attention_dim)
        self.log_tau = nn.Parameter(torch.tensor([float(torch.log(torch.tensor(tau_init)))]))
        self.alpha = nn.Parameter(torch.tensor([float(alpha_init)]))
        # logit_gate khoi tao bang 0.02 -> log_g = -0.02 -> g = 0.98.
        # Khoi tao bang 0.02 de tranh loi chet gradient (zero gradient) cua ReLU tai 0.
        self.logit_gate = nn.Parameter(torch.full((num_experts,), 0.02))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp_min(1e-3)

    @property
    def gate(self) -> torch.Tensor:
        # g = exp(-leaky_relu(logit_gate, 0.01))
        return torch.exp(-F.leaky_relu(self.logit_gate, negative_slope=0.01))

    def gate_reg(self) -> torch.Tensor:
        """Phat de giu g gan 1 (logit_gate gan 0). Cong vao tong loss trong trainer."""
        return self.gate_l2 * (self.logit_gate ** 2).sum()

    def forward(self, embedding, expert_logits, domain_probs):
        prior = torch.log(domain_probs.clamp_min(self.eps))
        attn = self.attn(embedding)
        log_g = -F.leaky_relu(self.logit_gate, negative_slope=0.01).unsqueeze(0)  # (1,K)
        weights = F.softmax(prior / self.tau + self.alpha * attn + log_g, dim=-1)
        return _softmax_fuse(weights, expert_logits), weights

def build_fusion(fusion_type: str, embedding_dim: int, num_experts: int,
                 num_heads: int = 4, attention_dim: int = 128, **kw):
    if fusion_type == "linear_softmax": return None
    if fusion_type == "tempered_gate": return TemperedGateFusion(num_experts, kw.get("tau_init", 1.0))
    if fusion_type == "residual_gate":
        return ResidualGateFusion(embedding_dim, num_experts, num_heads, attention_dim,
                                  tau_init=kw.get("tau_init", 1.0), alpha_init=kw.get("alpha_init", 0.0))
    if fusion_type == "gated_residual":
        return GatedResidualFusion(embedding_dim, num_experts, num_heads, attention_dim,
                                   tau_init=kw.get("tau_init", 1.0), alpha_init=kw.get("alpha_init", 0.0),
                                   gate_l2=kw.get("gate_l2", 1e-3))
    if fusion_type == "multihead_attention": return None
    raise ValueError(f"Unknown fusion type: {fusion_type}")
