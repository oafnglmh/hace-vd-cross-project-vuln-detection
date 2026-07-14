"""Classification metrics for vulnerability detection."""

from __future__ import annotations

import numpy as np


def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Compute Precision / Recall / F1 / Accuracy for the positive class.

    The positive class is taken to be ``1`` (vulnerable) which is consistent
    with the CEVD paper.
    """
    labels = np.asarray(labels).astype(int)
    preds  = np.asarray(preds).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def per_project_metrics(
    labels:  np.ndarray,
    preds:   np.ndarray,
    project: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Group metrics by project name."""
    result: dict[str, dict[str, float]] = {}
    for p in np.unique(project):
        mask = project == p
        if mask.sum() == 0:
            continue
        result[str(p)] = binary_metrics(labels[mask], preds[mask])
    return result
