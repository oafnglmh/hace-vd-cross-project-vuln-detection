"""Cross-project evaluator."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.hacevd_model import HACEVDModel
from .metrics import binary_metrics, per_project_metrics

class Evaluator:
    def __init__(self, model: HACEVDModel, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> dict[str, np.ndarray]:
        all_preds, all_labels, all_projects, all_probs = [], [], [], []
        for batch in loader:
            embeddings = batch["embedding"].to(self.device)
            out = self.model(embeddings)
            preds = out.vuln_logits.argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(out.vuln_logits, dim=-1)[:, 1].cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(batch["label"].cpu().numpy())
            all_projects.extend(batch["project"])
        return {
            "preds":    np.concatenate(all_preds),
            "probs":    np.concatenate(all_probs),
            "labels":   np.concatenate(all_labels),
            "projects": np.array(all_projects, dtype=object),
        }

    def evaluate(self, loader: DataLoader, threshold: float = 0.5) -> dict:
        preds = self.predict(loader)
        bin_preds = (preds["probs"] >= threshold).astype(int)
        overall = binary_metrics(preds["labels"], bin_preds)
        per_proj = per_project_metrics(preds["labels"], bin_preds, preds["projects"])
        
        # Thêm Average Precision (AP)
        from sklearn.metrics import average_precision_score
        overall["ap"] = float(average_precision_score(preds["labels"], preds["probs"]))
        for p in np.unique(preds["projects"]):
            mask = preds["projects"] == p
            if mask.sum() > 0:
                per_proj[str(p)]["ap"] = float(
                    average_precision_score(preds["labels"][mask], preds["probs"][mask])
                )
        return {"overall": overall, "per_project": per_proj, "threshold": threshold}
