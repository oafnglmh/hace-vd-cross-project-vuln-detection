"""Training loop for HACE-VD.

A single ``Trainer`` class encapsulates the optimization logic. The trainer
expects:
    * A pre-built ``HACEVDModel``.
    * Train / validation ``DataLoader`` instances over ``EmbeddingDataset``s.
    * A loss-configuration dictionary specifying the weights of the
      vulnerability, domain, and contrastive losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.hacevd_model import HACEVDModel
from losses import (
    soft_cross_entropy,
    supervised_contrastive_loss,
    weighted_vulnerability_loss,
)
from evaluation.metrics import binary_metrics


@dataclass
class LossConfig:
    vuln_weight:             float = 1.0
    domain_weight:           float = 0.1
    contrastive_weight:      float = 0.1
    contrastive_temperature: float = 0.07


@dataclass
class TrainConfig:
    num_epochs:        int   = 10
    learning_rate:     float = 5e-4
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    early_stopping_patience: int = 3


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_f1:     list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model:        HACEVDModel,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        train_cfg:    TrainConfig,
        loss_cfg:     LossConfig,
        output_dir:   Path,
        device:       Optional[torch.device] = None,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.train_cfg    = train_cfg
        self.loss_cfg     = loss_cfg
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.history = TrainingHistory()

    # ---------------------------------------------------------------------
    #  Single-step compute
    # ---------------------------------------------------------------------
    def compute_losses(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        embeddings   = batch["embedding"].to(self.device)
        labels       = batch["label"].to(self.device)
        soft_targets = batch["soft_domain"].to(self.device)
        hard_domains = batch["hard_domain"].to(self.device)

        out = self.model(embeddings)

        # Vulnerability (Equation (8)).
        loss_v = weighted_vulnerability_loss(out.vuln_logits, labels)

        # Domain classification (Equation (9)).
        loss_d = soft_cross_entropy(out.domain_probs, soft_targets)

        # Optional attention supervision cross entropy loss
        loss_attn_d = torch.zeros((), device=self.device)
        if (
            self.model.fusion_type == "multihead_attention"
            and getattr(self.model, "attention_supervision", "none") == "cross_entropy"
        ):
            loss_attn_d = soft_cross_entropy(out.fusion_weights, soft_targets)

        # Contrastive (HACE-VD specific).
        loss_c = torch.zeros((), device=self.device)
        if self.loss_cfg.contrastive_weight > 0 and out.projections is not None:
            B, K, _ = out.projections.shape
            # We use only the projection corresponding to each sample's hard
            # domain so that the contrastive signal stays specialized.
            idx = hard_domains.clamp(min=0, max=K - 1)
            selected = out.projections[torch.arange(B), idx]            # (B, P)
            loss_c = supervised_contrastive_loss(
                projections=selected,
                labels=labels,
                domains=hard_domains,
                temperature=self.loss_cfg.contrastive_temperature,
            )

        total = (
            self.loss_cfg.vuln_weight        * loss_v
            + self.loss_cfg.domain_weight    * loss_d
            + self.loss_cfg.domain_weight    * loss_attn_d
            + self.loss_cfg.contrastive_weight * loss_c
        )

        # Quality-gate regularization (0 nếu fusion không có gate).
        # Giữ g gần 1 trừ khi dữ liệu thực sự chứng minh expert kém.
        total = total + self.model.fusion_gate_reg()

        log = {
            "loss_total": float(total.item()),
            "loss_vuln":  float(loss_v.item()),
            "loss_domain": float(loss_d.item()),
            "loss_attn_domain": float(loss_attn_d.item()),
            "loss_contrast": float(loss_c.item()),
        }
        return total, log

    # ---------------------------------------------------------------------
    #  Train / val loops
    # ---------------------------------------------------------------------
    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        running = {"loss_total": 0.0, "n": 0}
        pbar = tqdm(self.train_loader, desc="  Training batches", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            loss, log = self.compute_losses(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.train_cfg.grad_clip,
            )
            self.optimizer.step()
            bs = batch["embedding"].size(0)
            running["loss_total"] += log["loss_total"] * bs
            running["n"]          += bs
            pbar.set_postfix(loss=log["loss_total"])
        return {"loss_total": running["loss_total"] / max(running["n"], 1)}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds, all_labels = [], []
        loss_sum = 0.0
        n = 0
        pbar = tqdm(loader, desc="  Evaluating batches", leave=False)
        for batch in pbar:
            embeddings = batch["embedding"].to(self.device)
            labels     = batch["label"].to(self.device)
            soft_targets = batch["soft_domain"].to(self.device)

            out = self.model(embeddings)
            loss_v = weighted_vulnerability_loss(out.vuln_logits, labels)
            loss_d = soft_cross_entropy(out.domain_probs, soft_targets)
            
            loss_attn_d = torch.zeros((), device=self.device)
            if (
                self.model.fusion_type == "multihead_attention"
                and getattr(self.model, "attention_supervision", "none") == "cross_entropy"
            ):
                loss_attn_d = soft_cross_entropy(out.fusion_weights, soft_targets)

            loss = (
                self.loss_cfg.vuln_weight * loss_v
                + self.loss_cfg.domain_weight * loss_d
                + self.loss_cfg.domain_weight * loss_attn_d
            )

            preds = out.vuln_logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            loss_sum += float(loss.item()) * embeddings.size(0)
            n        += embeddings.size(0)
            pbar.set_postfix(loss=loss.item())

        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        metrics = binary_metrics(labels, preds)
        metrics["loss"] = loss_sum / max(n, 1)
        return metrics

    # ---------------------------------------------------------------------
    #  Full fit with early stopping on val-F1
    # ---------------------------------------------------------------------
    def fit(self) -> TrainingHistory:
        best_f1 = -1.0
        no_improve_epochs = 0
        best_path = self.output_dir / "best.pt"

        for epoch in range(1, self.train_cfg.num_epochs + 1):
            train_log = self.train_epoch()
            val_log   = self.evaluate(self.val_loader)
            self.history.train_loss.append(train_log["loss_total"])
            self.history.val_loss.append(val_log["loss"])
            self.history.val_f1.append(val_log["f1"])
            print(
                f"[epoch {epoch:02d}] "
                f"train_loss={train_log['loss_total']:.4f}  "
                f"val_loss={val_log['loss']:.4f}  "
                f"val_f1={val_log['f1']*100:.2f}%"
            )

            fus = getattr(self.model, "fusion", None)
            if fus is not None:
                msg = []
                if hasattr(fus, "tau"):   msg.append(f"tau={fus.tau.item():.3f}")
                if hasattr(fus, "alpha"): msg.append(f"alpha={fus.alpha.item():.3f}")
                if hasattr(fus, "gate"):
                    msg.append("gate=[" + ", ".join(f"{g:.3f}"
                        for g in fus.gate.tolist()) + "]")
                if msg: print("        [fusion] " + "  ".join(msg))

            if val_log["f1"] > best_f1:
                best_f1 = val_log["f1"]
                no_improve_epochs = 0
                torch.save(
                    {"model_state": self.model.state_dict(), "epoch": epoch, "val_f1": best_f1},
                    best_path,
                )
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= self.train_cfg.early_stopping_patience:
                    print(f"Early stopping after epoch {epoch} (no improvement for "
                          f"{no_improve_epochs} epochs).")
                    break
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
        return self.history
