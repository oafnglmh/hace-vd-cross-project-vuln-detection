"""Cross-project evaluator.

The evaluator computes overall and per-project metrics on a held-out set
of unseen projects. Intended usage is post-training: load a checkpoint,
push a test ``DataLoader`` through, and obtain a tidy results dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.hacevd_model import HACEVDModel
from .metrics import binary_metrics, per_project_metrics


class Evaluator:
    def __init__(
        self,
        model: HACEVDModel,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> dict[str, np.ndarray]:
        """Run the model and return predictions + true labels + projects."""
        all_preds, all_labels, all_projects = [], [], []
        for batch in loader:
            embeddings = batch["embedding"].to(self.device)
            out = self.model(embeddings)
            preds = out.vuln_logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["label"].cpu().numpy())
            all_projects.extend(batch["project"])
        return {
            "preds":    np.concatenate(all_preds),
            "labels":   np.concatenate(all_labels),
            "projects": np.array(all_projects, dtype=object),
        }

    def evaluate(self, loader: DataLoader) -> dict:
        preds = self.predict(loader)
        overall = binary_metrics(preds["labels"], preds["preds"])
        per_proj = per_project_metrics(
            preds["labels"], preds["preds"], preds["projects"]
        )
        return {"overall": overall, "per_project": per_proj}
