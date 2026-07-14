"""Offline pseudo-domain construction.

Takes a cached ``.npy`` of training embeddings and writes a
``.npz`` file containing ``assignments``, ``centers`` and ``k``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clustering import construct_pseudo_domains


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline pseudo-domain construction.")
    parser.add_argument("--emb",    type=Path, required=True, help="Input .npy of (N, D) embeddings")
    parser.add_argument("--out",    type=Path, required=True, help="Output .npz")
    parser.add_argument("--method", type=str,  default="hdbscan", choices=["kmeans", "hdbscan", "gmm"])
    parser.add_argument("--metric", type=str,  default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--k",      type=int,  default=None)
    parser.add_argument("--min-cluster-size", type=int, default=200)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs for HDBSCAN (-1 to use all cores)")
    args = parser.parse_args()

    import time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading embeddings from {args.emb}...")
    emb = np.load(args.emb)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded embeddings of shape {emb.shape}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running clustering method '{args.method}'...")
    start_time = time.time()
    res = construct_pseudo_domains(
        embeddings=emb,
        method=args.method,
        metric=args.metric,
        k=args.k,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        core_dist_n_jobs=args.n_jobs,
    )
    elapsed = time.time() - start_time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Clustering finished in {elapsed:.2f} seconds.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        assignments=res.assignments,
        centers=res.centers,
        k=res.k,
    )
    counts = np.bincount(res.assignments).tolist()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Wrote {args.out}  k={res.k}  cluster sizes={counts}")


if __name__ == "__main__":
    main()
