"""Analyze the entropy of domain classification weights vs attention fusion weights.

Usage:
    python -m scripts.analyze_entropy --config config/hacevd_codebert.yaml --ckpt runs/hacevd_codebert/seed_8/best.pt --centers data/pseudo_domains/hacevd_codebert.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from clustering import generate_soft_labels, l2_normalize
from data import EmbeddingDataset, build_dataloader
from models import HACEVDModel


def calculate_entropy(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Calculate the Shannon entropy of a probability distribution along the last axis."""
    return -np.sum(weights * np.log(weights + eps), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze entropy of domain probabilities vs attention weights.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt",   type=Path, required=True)
    parser.add_argument("--centers", type=Path, required=True,
                        help=".npz produced by scripts.cluster containing 'centers'.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    emb_root = Path(cfg["paths"]["embeddings_root"])

    # Load test split
    test_emb     = np.load(emb_root / "test_embeddings.npy")
    test_labels  = np.load(emb_root / "test_labels.npy")
    test_proj    = np.load(emb_root / "test_projects.npy", allow_pickle=True)

    # Compute domain assignments
    centers = np.load(args.centers)["centers"]
    test_norm = l2_normalize(test_emb)
    soft = generate_soft_labels(test_norm, centers, temperature=cfg["soft_labels"]["temperature"],
                                l2_normalize=False)
    hard = soft.argmax(axis=1)

    ds = EmbeddingDataset(
        embeddings=test_emb.astype(np.float32),
        labels=test_labels.astype(np.int64),
        projects=test_proj.astype(object),
        hard_domains=hard.astype(np.int64),
        soft_domains=soft.astype(np.float32),
    )
    loader = build_dataloader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Instantiate model
    model = HACEVDModel(
        embedding_dim=cfg["backbone"]["embedding_dim"],
        num_domains=int(centers.shape[0]),
        expert_hidden=cfg["experts"]["hidden_dim"],
        expert_dropout=cfg["experts"]["dropout"],
        expert_activation=cfg["experts"]["activation"],
        domain_hidden=cfg["domain_classifier"]["hidden_dim"],
        domain_dropout=cfg["domain_classifier"]["dropout"],
        fusion_type=cfg["fusion"]["type"],
        attention_heads=cfg["fusion"].get("num_heads") or 4,
        attention_dim=cfg["fusion"].get("attention_dim") or 128,
        use_projection=cfg["loss"]["contrastive_weight"] > 0,
        attention_supervision=cfg["fusion"].get("attention_supervision", "none"),
        tau_init=cfg["fusion"].get("tau_init", 1.0),
        alpha_init=cfg["fusion"].get("alpha_init", 0.0),
        gate_l2=cfg["fusion"].get("gate_l2", 1e-3),
    )

    # Load state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.ckpt} on {device}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Collect weights
    all_domain_probs = []
    all_fusion_weights = []

    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            out = model(embeddings)
            all_domain_probs.append(out.domain_probs.cpu().numpy())
            all_fusion_weights.append(out.fusion_weights.cpu().numpy())

    domain_probs = np.concatenate(all_domain_probs, axis=0)      # (N, K)
    fusion_weights = np.concatenate(all_fusion_weights, axis=0)  # (N, K)

    # Calculate entropies
    domain_entropy = calculate_entropy(domain_probs)
    fusion_entropy = calculate_entropy(fusion_weights)

    max_entropy = np.log(centers.shape[0])

    print("\n" + "="*60)
    print(f" Entropy Analysis (RQ3) - {centers.shape[0]} Domains (Max Entropy: {max_entropy:.4f})")
    print("="*60)
    
    print("\nDomain Classifier Probabilities (Linear Softmax weights):")
    print(f"  Mean Entropy:   {np.mean(domain_entropy):.4f}")
    print(f"  Std Dev:        {np.std(domain_entropy):.4f}")
    print(f"  Percentiles (25, 50, 75): {np.percentile(domain_entropy, [25, 50, 75])}")
    print(f"  Min / Max:      {np.min(domain_entropy):.4f} / {np.max(domain_entropy):.4f}")

    print("\nAttention Fusion Weights:")
    print(f"  Mean Entropy:   {np.mean(fusion_entropy):.4f}")
    print(f"  Std Dev:        {np.std(fusion_entropy):.4f}")
    print(f"  Percentiles (25, 50, 75): {np.percentile(fusion_entropy, [25, 50, 75])}")
    print(f"  Min / Max:      {np.min(fusion_entropy):.4f} / {np.max(fusion_entropy):.4f}")

    print("\nInterpretation:")
    print(f"  - High entropy (~{max_entropy:.2f}) indicates weights are close to uniform distribution.")
    print("  - Low entropy (~0.00) indicates weights are collapsed/concentrated on a single expert.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
