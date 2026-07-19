"""Cross-project evaluation: loads a trained checkpoint and reports per-project metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from clustering import generate_soft_labels, l2_normalize
from data import EmbeddingDataset, build_dataloader
from evaluation import Evaluator
from models import HACEVDModel

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt",   type=Path, required=True)
    parser.add_argument("--centers", type=Path, required=True)
    parser.add_argument("--out",    type=Path, default=Path("eval_report.json"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    emb_root = Path(cfg["paths"]["embeddings_root"])

    # Load test split
    test_emb     = np.load(emb_root / "test_embeddings.npy")
    test_labels  = np.load(emb_root / "test_labels.npy")
    test_proj    = np.load(emb_root / "test_projects.npy", allow_pickle=True)

    centers = np.load(args.centers)["centers"]

    def make_loader(emb, labels, proj):
        norm = l2_normalize(emb)
        soft = generate_soft_labels(norm, centers, temperature=cfg["soft_labels"]["temperature"], l2_normalize=False)
        hard = soft.argmax(axis=1)
        ds = EmbeddingDataset(embeddings=emb.astype(np.float32), labels=labels.astype(np.int64), projects=proj.astype(object), hard_domains=hard.astype(np.int64), soft_domains=soft.astype(np.float32))
        return build_dataloader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    test_loader = make_loader(test_emb, test_labels, test_proj)

    model = HACEVDModel(
        embedding_dim=cfg["backbone"]["embedding_dim"], num_domains=int(centers.shape[0]),
        expert_hidden=cfg["experts"]["hidden_dim"], expert_dropout=cfg["experts"]["dropout"], expert_activation=cfg["experts"]["activation"],
        domain_hidden=cfg["domain_classifier"]["hidden_dim"], domain_dropout=cfg["domain_classifier"]["dropout"],
        fusion_type=cfg["fusion"]["type"], attention_heads=cfg["fusion"].get("num_heads") or 4, attention_dim=cfg["fusion"].get("attention_dim") or 128,
        use_projection=cfg["loss"]["contrastive_weight"] > 0, attention_supervision=cfg["fusion"].get("attention_supervision", "none"),
        tau_init=cfg["fusion"].get("tau_init", 1.0), alpha_init=cfg["fusion"].get("alpha_init", 0.0), gate_l2=cfg["fusion"].get("gate_l2", 1e-3),
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    evaluator = Evaluator(model)
    best_threshold = 0.5
    
    # Đánh giá Test với ngưỡng 0.5
    results = evaluator.evaluate(test_loader, threshold=best_threshold)

    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["overall"], indent=2, default=float))

if __name__ == "__main__":
    main()
