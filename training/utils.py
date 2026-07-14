"""Training utilities for HACE-VD.

Small, dependency-light helpers shared across the training and evaluation
scripts: reproducible seeding, device selection, parameter counting, a running
average meter, and JSON I/O for experiment results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs.

    The CEVD protocol reports the mean and standard deviation over five seeds
    (8, 16, 24, 32, 40); calling this at the top of every run is what makes
    those numbers comparable across the baseline and HACE-VD.

    Parameters
    ----------
    seed          : the integer seed.
    deterministic : if True, also request deterministic cuDNN kernels. This can
                    slightly slow training but removes a source of run-to-run
                    variance that would otherwise inflate the reported std.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters of a module.

    With a frozen backbone the only trainable parameters are the experts, the
    domain classifier and the attention / projection heads, so this number
    should stay roughly constant in the number of experts.
    """
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


class AverageMeter:
    """Tracks a running average (e.g. of a loss term across a training epoch)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` to ``path`` as indented JSON (numpy types are coerced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate_seed_results(f1_scores: list[float]) -> dict[str, float]:
    """Summarize per-seed F1 scores as mean and (population) std.

    Matches the ``mean +/- std`` reporting format used in the CEVD paper and in
    the HACE-VD result tables.
    """
    arr = np.asarray(f1_scores, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "n": int(arr.size),
    }
