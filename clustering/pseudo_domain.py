"""Offline pseudo-domain construction.

This module supports three clustering algorithms:
    * ``kmeans``  : the CEVD baseline (k-means with cosine distance).
    * ``hdbscan`` : HACE-VD's density-based alternative. Removes the convex-
                    cluster assumption and lets the data determine k.
    * ``gmm``     : optional Gaussian Mixture variant for ablation.

All algorithms operate on L2-normalized embeddings when ``metric == "cosine"``
because cosine distance is equivalent to Euclidean distance on the unit sphere.

Outputs
-------
The ``construct_pseudo_domains`` function returns a ``PseudoDomainResult``
dataclass holding the cluster assignments, the cluster centers (in normalized
embedding space), and the number of effective clusters.

GPU acceleration
-----------------
When available, the HDBSCAN branch uses ``cuml.cluster.HDBSCAN`` (RAPIDS) to
run on GPU, which is dramatically faster than the CPU ``hdbscan`` package on
high-dimensional, large-N inputs. If cuML/GPU is not available, it silently
falls back to the CPU implementation so the code still runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    import hdbscan  # type: ignore
    _HAS_HDBSCAN = True
except ImportError:  # pragma: no cover - tested separately
    _HAS_HDBSCAN = False

try:
    import cupy as cp  # type: ignore
    from cuml.cluster import HDBSCAN as cuHDBSCAN  # type: ignore
    _HAS_CUML = True
except ImportError:  # pragma: no cover - GPU/cuML optional
    _HAS_CUML = False


@dataclass
class PseudoDomainResult:
    assignments: np.ndarray          # (N,) integer cluster index in [0, k-1]
    centers:     np.ndarray          # (k, D) cluster center in normalized space
    k:           int                  # number of effective clusters

    def __post_init__(self) -> None:
        if self.assignments.ndim != 1:
            raise ValueError("assignments must be a 1-d array")
        if self.centers.shape[0] != self.k:
            raise ValueError("centers.shape[0] must equal k")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def compute_centers(embeddings: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
    """Compute (and re-normalize) cluster centroids."""
    D = embeddings.shape[1]
    centers = np.zeros((k, D), dtype=embeddings.dtype)
    for c in range(k):
        mask = assignments == c
        if not mask.any():
            # Should not happen if we filtered out noise upstream.
            centers[c] = np.zeros(D, dtype=embeddings.dtype)
            continue
        centers[c] = embeddings[mask].mean(axis=0)
    return l2_normalize(centers)


def _run_hdbscan(x: np.ndarray, min_cluster_size: int, min_samples: int,
                  core_dist_n_jobs: int = -1, use_gpu: bool = True) -> np.ndarray:
    """Run HDBSCAN on ``x`` (already L2-normalized, euclidean metric).

    Tries cuML (GPU) first when available and ``use_gpu`` is True; falls back
    to the CPU ``hdbscan`` package otherwise. Returns raw labels (-1 = noise).
    """
    if use_gpu and _HAS_CUML:
        try:
            x_gpu = cp.asarray(x, dtype=cp.float32)
            model = cuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
            )
            raw = model.fit_predict(x_gpu)
            return cp.asnumpy(raw).astype(np.int64)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"      [WARN] cuML HDBSCAN failed ({exc}); falling back to CPU hdbscan.")

    if not _HAS_HDBSCAN:
        raise ImportError("Neither cuML nor the CPU 'hdbscan' package is available. "
                           "Run `pip install hdbscan` or install RAPIDS cuml.")
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=core_dist_n_jobs,
    )
    return model.fit_predict(x).astype(np.int64)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def construct_pseudo_domains(
    embeddings: np.ndarray,
    method:     str = "hdbscan",
    metric:     str = "cosine",
    k:          Optional[int] = None,
    min_cluster_size: int = 200,
    min_samples: int = 50,
    l2_normalize_inputs: bool = True,
    random_state: int = 42,
    core_dist_n_jobs: int = -1,
    use_gpu: bool = True,
) -> PseudoDomainResult:
    """Cluster the embeddings and return assignments + centers.

    Parameters
    ----------
    embeddings : (N, D) float array of code embeddings.
    method     : "kmeans", "hdbscan", or "gmm".
    metric     : "cosine" or "euclidean". For ``cosine`` we L2-normalize
                 inputs and run the clustering algorithm with Euclidean
                 distance (equivalent on the unit sphere).
    k          : required when ``method == "kmeans"`` or ``"gmm"``.
    min_cluster_size, min_samples : HDBSCAN parameters.
    core_dist_n_jobs : CPU parallelism for the CPU HDBSCAN fallback.
    use_gpu    : if True (default), try cuML/GPU HDBSCAN before falling back
                 to CPU. Has no effect on kmeans/gmm.
    """
    if l2_normalize_inputs and metric == "cosine":
        x = l2_normalize(embeddings).astype(np.float32)
    else:
        x = embeddings.astype(np.float32)

    if method == "kmeans":
        if k is None:
            raise ValueError("kmeans requires 'k' to be set.")
        model = KMeans(
            n_clusters=k,
            n_init=10,
            init="k-means++",
            random_state=random_state,
        )
        assignments = model.fit_predict(x)
        centers = l2_normalize(model.cluster_centers_)
        return PseudoDomainResult(
            assignments=assignments.astype(np.int64),
            centers=centers.astype(np.float32),
            k=k,
        )

    if method == "gmm":
        if k is None:
            raise ValueError("gmm requires 'k' to be set.")
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=random_state,
            reg_covar=1e-4,
        )
        assignments = model.fit_predict(x)
        centers = compute_centers(x, assignments, k=k)
        return PseudoDomainResult(
            assignments=assignments.astype(np.int64),
            centers=centers.astype(np.float32),
            k=k,
        )

    if method == "hdbscan":
        backend = "cuML (GPU)" if (use_gpu and _HAS_CUML) else "hdbscan (CPU)"
        print(f"      [HDBSCAN] backend={backend}  min_cluster_size={min_cluster_size}  "
              f"min_samples={min_samples}  n={x.shape[0]}  dim={x.shape[1]}")
        raw = _run_hdbscan(
            x, min_cluster_size=min_cluster_size, min_samples=min_samples,
            core_dist_n_jobs=core_dist_n_jobs, use_gpu=use_gpu,
        )
        # HDBSCAN labels noise points as -1. We reassign noise points to the
        # nearest cluster centroid so that every sample gets a pseudo-domain.
        if (raw == -1).any():
            valid_labels = sorted(set(raw) - {-1})
            if not valid_labels:
                # All points labeled as noise -> collapse to a single cluster.
                raw = np.zeros_like(raw)
                k_eff = 1
                centers = compute_centers(x, raw, k=k_eff)
            else:
                centers = compute_centers(x, raw, k=max(valid_labels) + 1)
                centers = centers[valid_labels]
                # Reassign noise to nearest centroid.
                noise_idx = np.where(raw == -1)[0]
                if len(noise_idx) > 0:
                    sims = x[noise_idx] @ centers.T          # (N_noise, k)
                    raw[noise_idx] = np.argmax(sims, axis=1)
                # Remap labels to a contiguous range starting at 0.
                remap = {old: new for new, old in enumerate(valid_labels)}
                raw = np.array([remap.get(int(l), int(l)) for l in raw], dtype=np.int64)
                k_eff = len(valid_labels)
                centers = compute_centers(x, raw, k=k_eff)
        else:
            valid_labels = sorted(set(raw.tolist()))
            remap = {old: new for new, old in enumerate(valid_labels)}
            raw = np.array([remap[int(l)] for l in raw], dtype=np.int64)
            k_eff = len(valid_labels)
            centers = compute_centers(x, raw, k=k_eff)
        return PseudoDomainResult(
            assignments=raw.astype(np.int64),
            centers=centers.astype(np.float32),
            k=int(k_eff),
        )

    raise ValueError(f"Unknown clustering method: {method}")
