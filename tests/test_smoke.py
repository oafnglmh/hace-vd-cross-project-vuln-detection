"""End-to-end smoke test on synthetic data.

Run with: ``python -m tests.test_smoke``

The test exercises the entire pipeline (clustering, soft labels, model
forward, full backward, train + eval) on a few hundred synthetic samples.
No GPU is required and the test should complete in well under one minute.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from clustering import construct_pseudo_domains, generate_soft_labels, l2_normalize
from data import EmbeddingDataset, build_dataloader, synthetic_dataset
from evaluation import Evaluator
from losses import (
    soft_cross_entropy,
    supervised_contrastive_loss,
    weighted_vulnerability_loss,
)
from models import HACEVDModel
from training import LossConfig, TrainConfig, Trainer


def run_smoke_test() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    print("\n" + "=" * 70)
    print(" HACE-VD smoke test")
    print("=" * 70)

    # ---- 1. Build a tiny synthetic embedding dataset -----------------------
    print("\n[1/6] Building synthetic embeddings...")
    train = synthetic_dataset(num_samples=300, embedding_dim=32, num_domains=3, seed=1)
    val   = synthetic_dataset(num_samples=90,  embedding_dim=32, num_domains=3, seed=2)
    test  = synthetic_dataset(num_samples=90,  embedding_dim=32, num_domains=3, seed=3)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    # ---- 2. Run clustering -------------------------------------------------
    print("\n[2/6] k-means clustering (k=3, cosine)...")
    km = construct_pseudo_domains(
        embeddings=train.embeddings,
        method="kmeans",
        metric="cosine",
        k=3,
        l2_normalize_inputs=True,
    )
    print(f"      cluster sizes = {np.bincount(km.assignments).tolist()}")
    assert km.k == 3

    print("\n[2b/6] HDBSCAN clustering...")
    try:
        h = construct_pseudo_domains(
            embeddings=train.embeddings,
            method="hdbscan",
            metric="cosine",
            min_cluster_size=20,
            min_samples=5,
            l2_normalize_inputs=True,
            min_k=2,
        )
        print(f"      HDBSCAN -> k={h.k}  sizes={np.bincount(h.assignments).tolist()}")
    except ImportError:
        print("      hdbscan not installed, skipping that branch.")

    # ---- 3. Soft labels ----------------------------------------------------
    print("\n[3/6] Generating soft labels...")
    soft = generate_soft_labels(l2_normalize(train.embeddings), km.centers, temperature=1.0,
                                l2_normalize=False)
    assert soft.shape == (len(train), 3)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)
    print(f"      soft shape = {soft.shape}")

    # ---- 4. Build datasets / loaders --------------------------------------
    print("\n[4/6] Building datasets...")
    train.hard_domains = km.assignments
    train.soft_domains = soft.astype(np.float32)
    val.hard_domains   = generate_soft_labels(l2_normalize(val.embeddings),  km.centers).argmax(axis=1).astype(np.int64)
    val.soft_domains   = generate_soft_labels(l2_normalize(val.embeddings),  km.centers).astype(np.float32)
    test.hard_domains  = generate_soft_labels(l2_normalize(test.embeddings), km.centers).argmax(axis=1).astype(np.int64)
    test.soft_domains  = generate_soft_labels(l2_normalize(test.embeddings), km.centers).astype(np.float32)

    train_loader = build_dataloader(train, batch_size=16, oversample=True, shuffle=False)
    val_loader   = build_dataloader(val,   batch_size=16, shuffle=False)
    test_loader  = build_dataloader(test,  batch_size=16, shuffle=False)

    # ---- 5. Instantiate HACE-VD model (all features enabled) --------------
    print("\n[5/6] Instantiating HACE-VD model...")
    model = HACEVDModel(
        embedding_dim=32,
        num_domains=3,
        expert_hidden=32,
        expert_dropout=0.1,
        expert_activation="tanh",
        domain_hidden=32,
        domain_dropout=0.1,
        fusion_type="gated_residual",
        attention_heads=2,
        attention_dim=16,
        use_projection=True,
        projection_hidden=16,
        projection_dim=8,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      params = {n_params}")

    # ---- 6. Train and evaluate --------------------------------------------
    print("\n[6/6] Train + evaluate...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=TrainConfig(num_epochs=3, learning_rate=5e-3, early_stopping_patience=5),
        loss_cfg=LossConfig(contrastive_weight=0.1),
        output_dir=Path("./smoke_out"),
    )
    trainer.fit()

    evaluator = Evaluator(model=model)
    res = evaluator.evaluate(test_loader)
    print(f"\nTest overall: P={res['overall']['precision']*100:.1f}%  "
          f"R={res['overall']['recall']*100:.1f}%  "
          f"F1={res['overall']['f1']*100:.1f}%")

    # Sanity assertions
    assert 0.0 <= res["overall"]["f1"] <= 1.0
    assert len(res["per_project"]) > 0

    # Test the linear-softmax (CEVD-mode) fusion too.
    print("\nTesting CEVD-mode (linear_softmax) fusion...")
    cevd_model = HACEVDModel(
        embedding_dim=32, num_domains=3, expert_hidden=32, expert_dropout=0.1,
        expert_activation="tanh", domain_hidden=32, domain_dropout=0.1,
        fusion_type="linear_softmax", use_projection=False,
    )
    cevd_trainer = Trainer(
        model=cevd_model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=TrainConfig(num_epochs=2, learning_rate=5e-3, early_stopping_patience=5),
        loss_cfg=LossConfig(contrastive_weight=0.0),
        output_dir=Path("./smoke_out_cevd"),
    )
    cevd_trainer.fit()

    # Test individual loss functions
    print("\nTesting loss functions in isolation...")
    B, K, C = 8, 3, 2
    fused = torch.randn(B, C, requires_grad=True)
    labels = torch.randint(0, 2, (B,))
    loss_v = weighted_vulnerability_loss(fused, labels)
    assert loss_v.requires_grad
    loss_v.backward()
    print(f"      weighted_vulnerability_loss OK  loss={loss_v.item():.4f}")

    probs = torch.softmax(torch.randn(B, K), dim=-1)
    soft_t = torch.softmax(torch.randn(B, K), dim=-1)
    loss_d = soft_cross_entropy(probs, soft_t)
    assert torch.isfinite(loss_d)
    print(f"      soft_cross_entropy OK  loss={loss_d.item():.4f}")

    proj = torch.randn(B, 8)
    proj = proj / proj.norm(dim=1, keepdim=True)
    proj.requires_grad_(True)
    domains = torch.randint(0, 3, (B,))
    loss_c = supervised_contrastive_loss(proj, labels, domains, temperature=0.07)
    assert torch.isfinite(loss_c)
    print(f"      supervised_contrastive_loss OK  loss={loss_c.item():.4f}")

    print("\n" + "=" * 70)
    print(" Smoke test PASSED.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
