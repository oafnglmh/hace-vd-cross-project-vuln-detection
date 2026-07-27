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
import torch

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

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


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


def _kmeans_pytorch(x: np.ndarray, k: int, metric: str = "cosine", random_state: int = 42, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Helper to run KMeans on GPU using PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.from_numpy(x).to(device)
    N, D = x_t.shape
    
    generator = torch.Generator(device=device)
    generator.manual_seed(random_state)
    
    indices = torch.randperm(N, generator=generator)[:k]
    centroids = x_t[indices].clone()
    
    if metric == "cosine":
        centroids = centroids / torch.clamp(centroids.norm(dim=1, keepdim=True), min=1e-12)
        
    assignments = torch.zeros(N, dtype=torch.long, device=device)
    for _ in range(max_iter):
        if metric == "cosine":
            sims = torch.matmul(x_t, centroids.T)  # (N, k)
            new_assignments = torch.argmax(sims, dim=1)
        else:
            dists = torch.cdist(x_t, centroids.unsqueeze(0)).squeeze(0)  # (N, k)
            new_assignments = torch.argmin(dists, dim=1)
            
        if torch.equal(assignments, new_assignments):
            break
        assignments = new_assignments
        
        # Recompute centroids
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = x_t[mask].mean(dim=0)
            else:
                new_centroids[c] = x_t[torch.randint(0, N, (1,), generator=generator)]
                
        if metric == "cosine":
            centroids = new_centroids / torch.clamp(new_centroids.norm(dim=1, keepdim=True), min=1e-12)
        else:
            centroids = new_centroids
            
    return assignments.cpu().numpy().astype(np.int64), centroids.cpu().numpy().astype(np.float32)


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
    umap_components: Optional[int] = None,
    umap_neighbors: int = 15,
    noise_threshold: float = 0.35,
    min_k: int = 4,
    max_k: int = 8,
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
    umap_components : if set, run UMAP dimensionality reduction before clustering.
    umap_neighbors : UMAP parameter.
    noise_threshold : maximum allowed noise ratio (default: 0.35).
    """
    if l2_normalize_inputs and metric == "cosine":
        x = l2_normalize(embeddings).astype(np.float32)
    else:
        x = embeddings.astype(np.float32)

    if method == "kmeans":
        if k is None:
            raise ValueError("kmeans requires 'k' to be set.")
        
        if use_gpu:
            if torch.cuda.is_available():
                try:
                    print("      [KMeans] Running with PyTorch on GPU...")
                    assignments, centers = _kmeans_pytorch(x, k, metric=metric, random_state=random_state)
                    if metric == "cosine":
                        centers = l2_normalize(centers)
                    return PseudoDomainResult(
                        assignments=assignments,
                        centers=centers.astype(np.float32),
                        k=k,
                    )
                except Exception as exc:
                    print(f"      [WARN] PyTorch GPU KMeans failed ({exc}); falling back to CPU KMeans.")
            elif _HAS_CUML:
                try:
                    print("      [KMeans] Running with cuML (RAPIDS) on GPU...")
                    from cuml.cluster import KMeans as cuKMeans
                    x_gpu = cp.asarray(x, dtype=cp.float32)
                    model = cuKMeans(n_clusters=k, init="k-means++", random_state=random_state)
                    assignments = model.fit_predict(x_gpu)
                    assignments = cp.asnumpy(assignments).astype(np.int64)
                    centers = cp.asnumpy(model.cluster_centers_)
                    centers = l2_normalize(centers)
                    return PseudoDomainResult(
                        assignments=assignments,
                        centers=centers.astype(np.float32),
                        k=k,
                    )
                except Exception as exc:
                    print(f"      [WARN] cuML KMeans failed ({exc}); falling back to CPU KMeans.")

        print("      [KMeans] Running on CPU...")
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
        # 1. Dimensionality reduction with UMAP (if configured)
        if umap_components is not None:
            if not _HAS_UMAP:
                raise ImportError("umap-learn package is required but not installed.")
            print(f"      [UMAP] Reducing dimension to {umap_components} (n_neighbors={umap_neighbors})")
            reducer = umap.UMAP(
                n_neighbors=umap_neighbors,
                n_components=umap_components,
                metric=metric,
                random_state=random_state,
                n_jobs=core_dist_n_jobs,
            )
            x_reduced = reducer.fit_transform(x)
        else:
            x_reduced = x

        backend = "cuML (GPU)" if (use_gpu and _HAS_CUML) else "hdbscan (CPU)"
        print(f"      [HDBSCAN] backend={backend}  min_cluster_size={min_cluster_size}  "
              f"min_samples={min_samples}  n={x_reduced.shape[0]}  dim={x_reduced.shape[1]}")
        
        raw = _run_hdbscan(
            x_reduced, min_cluster_size=min_cluster_size, min_samples=min_samples,
            core_dist_n_jobs=core_dist_n_jobs, use_gpu=use_gpu,
        )

        total_samples = len(raw)

        # Guardrail 1: Noise limit check before reassignment
        noise_count = np.sum(raw == -1)
        noise_ratio = noise_count / total_samples
        print(f"      [HDBSCAN Guardrail] Raw noise ratio: {noise_ratio:.2%} ({noise_count}/{total_samples})")
        if noise_ratio > noise_threshold:
            raise ValueError(f"HDBSCAN noise ratio {noise_ratio:.2%} exceeds threshold of {noise_threshold:.2%}. Config rejected.")

        # Guardrail 2: Merge micro-clusters (<2% of dataset) into nearest larger cluster
        min_size_threshold = int(0.02 * total_samples)
        print(f"      [HDBSCAN Guardrail] Min cluster size threshold (2%): {min_size_threshold} samples")

        while True:
            unique_lbls, counts = np.unique(raw[raw != -1], return_counts=True)
            small_clusters = unique_lbls[counts < min_size_threshold]
            large_clusters = unique_lbls[counts >= min_size_threshold]

            if len(small_clusters) == 0:
                break
            if len(large_clusters) == 0:
                print("      [WARN] All valid clusters are below the 2% threshold. Cannot merge further.")
                break

            # Merge the first small cluster to the nearest large cluster
            target_small = small_clusters[0]
            small_mask = raw == target_small
            small_centroid = x[small_mask].mean(axis=0)
            small_centroid /= np.linalg.norm(small_centroid) + 1e-12

            large_centroids = []
            for lc in large_clusters:
                lc_mask = raw == lc
                c_mean = x[lc_mask].mean(axis=0)
                c_mean /= np.linalg.norm(c_mean) + 1e-12
                large_centroids.append(c_mean)
            large_centroids = np.stack(large_centroids)

            sims = large_centroids @ small_centroid
            nearest_large_idx = np.argmax(sims)
            nearest_large_label = large_clusters[nearest_large_idx]

            print(f"      [HDBSCAN Guardrail] Merging cluster {target_small} ({np.sum(small_mask)} samples) "
                  f"into nearest cluster {nearest_large_label}")
            raw[small_mask] = nearest_large_label

        # Guardrail 3: Reassign remaining noise (-1) to the nearest centroid
        valid_labels = sorted(list(set(raw.tolist()) - {-1}))
        if not valid_labels:
            raw = np.zeros_like(raw)
            k_eff = 1
            centers = compute_centers(x, raw, k=k_eff)
        else:
            # Centroids of final valid clusters computed on high-dimensional original x
            centers = compute_centers(x, raw, k=max(valid_labels) + 1)
            centers = centers[valid_labels]

            noise_idx = np.where(raw == -1)[0]
            if len(noise_idx) > 0:
                sims = x[noise_idx] @ centers.T
                nearest_ctr_idx = np.argmax(sims, axis=1)
                raw[noise_idx] = np.array(valid_labels)[nearest_ctr_idx]

            # Remap labels to contiguous range starting at 0
            remap = {old: new for new, old in enumerate(valid_labels)}
            raw = np.array([remap[int(l)] for l in raw], dtype=np.int64)
            k_eff = len(valid_labels)
            centers = compute_centers(x, raw, k=k_eff)

        # Guardrail 4: Validate final ratios and cluster count
        unique_lbls, final_counts = np.unique(raw, return_counts=True)
        final_ratios = final_counts / total_samples
        print(f"      [HDBSCAN Guardrail] Final cluster count k={k_eff}, sizes={final_counts.tolist()}, ratios={np.round(final_ratios, 4).tolist()}")

        if k_eff < min_k or k_eff > max_k:
            raise ValueError(f"Final cluster count k={k_eff} is outside the allowed range [{min_k}, {max_k}].")

        for i, ratio in enumerate(final_ratios):
            if ratio > 0.45:
                raise ValueError(f"Cluster {i} contains {ratio:.2%} of the data, exceeding the maximum allowed ratio of 45%.")
            if ratio < 0.02:
                raise ValueError(f"Cluster {i} contains {ratio:.2%} of the data, which is below the minimum allowed ratio of 2%.")

        return PseudoDomainResult(
            assignments=raw.astype(np.int64),
            centers=centers.astype(np.float32),
            k=int(k_eff),
        )

    raise ValueError(f"Unknown clustering method: {method}")
