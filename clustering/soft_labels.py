"""Soft domain labels.

Given a set of cluster centers and per-sample embeddings, this module computes
a soft membership distribution over the pseudo-domains. We implement the
softmax-of-negative-cosine-distance formulation introduced in the CEVD paper
and add an optional temperature parameter for HACE-VD.
"""

from __future__ import annotations

import numpy as np


def cosine_distance_matrix(emb: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance between rows of ``emb`` and rows of ``centers``.

    Both inputs should already be L2-normalized.
    """
    # cosine distance = 1 - cosine similarity
    sims = emb @ centers.T
    return 1.0 - sims


def generate_soft_labels(
    embeddings: np.ndarray,
    centers:    np.ndarray,
    temperature: float = 1.0,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Return an ``(N, k)`` soft membership matrix.

    Equation (3) from the CEVD paper, optionally with a temperature ``T``:

        s_{i,m} = softmax_m( - d_cos(h_i, mu_m) / T )

    Setting ``temperature == 1.0`` recovers the CEVD definition.
    """
    if l2_normalize:
        emb_norm = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
        ctr_norm = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    else:
        emb_norm, ctr_norm = embeddings, centers

    distances = cosine_distance_matrix(emb_norm, ctr_norm)         # (N, k)
    logits = -distances / max(temperature, 1e-6)
    # numerically stable softmax
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)
