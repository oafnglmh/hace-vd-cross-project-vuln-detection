"""Data loading utilities for HACE-VD.

The data pipeline is intentionally simple: each sample is represented by a
pre-computed backbone embedding (cached as ``.npy``), a binary label, a
project name, and a pseudo-domain assignment + soft-label vector.

We deliberately decouple the backbone from the training loop so that
experimenting with different fusion / clustering / loss variants only requires
re-running the lightweight head networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


@dataclass
class EmbeddingDataset(Dataset):
    """Dataset of pre-computed embeddings + labels + pseudo-domain info."""

    embeddings:    np.ndarray            # (N, D) float32
    labels:        np.ndarray            # (N,)  int64   in {0, 1}
    projects:      np.ndarray            # (N,)  str object array
    hard_domains:  Optional[np.ndarray] = None    # (N,)  int64
    soft_domains:  Optional[np.ndarray] = None    # (N, K) float32

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "embedding": torch.from_numpy(self.embeddings[idx]).float(),
            "label":     torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "project":   str(self.projects[idx]),
        }
        if self.hard_domains is not None:
            item["hard_domain"] = torch.tensor(int(self.hard_domains[idx]), dtype=torch.long)
        if self.soft_domains is not None:
            item["soft_domain"] = torch.from_numpy(self.soft_domains[idx]).float()
        return item


def oversample_dataset(dataset: EmbeddingDataset) -> EmbeddingDataset:
    """Replicates minority class samples to exactly match the majority class size.

    This ensures balanced training batches without the high variance of
    WeightedRandomSampler bootstrap sampling.
    """
    labels = dataset.labels
    counts = np.bincount(labels)
    if len(counts) < 2 or 0 in counts:
        return dataset

    majority_class = np.argmax(counts)
    minority_class = 1 - majority_class

    majority_idx = np.where(labels == majority_class)[0]
    minority_idx = np.where(labels == minority_class)[0]

    n_majority = len(majority_idx)
    n_minority = len(minority_idx)

    if n_majority == 0 or n_minority == 0:
        return dataset

    # Replicate minority class indices to match majority size
    rng = np.random.default_rng(42)
    extra_minority_idx = rng.choice(minority_idx, size=n_majority - n_minority, replace=True)
    new_minority_idx = np.concatenate([minority_idx, extra_minority_idx])

    all_idx = np.concatenate([majority_idx, new_minority_idx])
    rng.shuffle(all_idx)

    new_embeddings = dataset.embeddings[all_idx]
    new_labels = dataset.labels[all_idx]
    new_projects = dataset.projects[all_idx]
    new_hard_domains = dataset.hard_domains[all_idx] if dataset.hard_domains is not None else None
    new_soft_domains = dataset.soft_domains[all_idx] if dataset.soft_domains is not None else None

    return EmbeddingDataset(
        embeddings=new_embeddings,
        labels=new_labels,
        projects=new_projects,
        hard_domains=new_hard_domains,
        soft_domains=new_soft_domains,
    )


def build_dataloader(
    dataset: EmbeddingDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    oversample: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader.

    If ``oversample`` is true we oversample the dataset to make minority and
    majority classes equal, and use normal shuffling.
    """
    if oversample:
        dataset = oversample_dataset(dataset)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )


def collate(batch: list[dict]) -> dict:
    """Stack tensors and keep string fields as a list."""
    out: dict = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out


# ---------------------------------------------------------------------------
#  Synthetic data factory (used by the smoke test and to demo the API).
# ---------------------------------------------------------------------------

def synthetic_dataset(
    num_samples: int = 200,
    embedding_dim: int = 32,
    num_domains: int = 3,
    seed: int = 0,
) -> EmbeddingDataset:
    """Generate a small synthetic dataset for testing.

    Creates ``num_domains`` Gaussian blobs in embedding space. Within each
    blob, half the samples are vulnerable. Project names are drawn from a
    small pool to simulate cross-project structure.
    """
    rng = np.random.default_rng(seed)
    samples_per_domain = num_samples // num_domains
    embeddings = []
    labels = []
    domains = []
    centers = rng.standard_normal((num_domains, embedding_dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    for d in range(num_domains):
        c = centers[d]
        noise = 0.1 * rng.standard_normal((samples_per_domain, embedding_dim))
        emb = c + noise
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        embeddings.append(emb.astype(np.float32))
        labels.append(rng.integers(0, 2, size=samples_per_domain).astype(np.int64))
        domains.append(np.full(samples_per_domain, d, dtype=np.int64))
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    hard_domains = np.concatenate(domains, axis=0)

    project_pool = ["proj_a", "proj_b", "proj_c", "proj_d"]
    projects = np.array([project_pool[i % len(project_pool)] for i in range(len(labels))], dtype=object)

    # Soft labels: simple softmax over similarities with centers.
    sims = embeddings @ centers.T
    exp = np.exp(sims - sims.max(axis=1, keepdims=True))
    soft = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    return EmbeddingDataset(
        embeddings=embeddings,
        labels=labels,
        projects=projects,
        hard_domains=hard_domains,
        soft_domains=soft,
    )
