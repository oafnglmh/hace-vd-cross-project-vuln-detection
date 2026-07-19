"""Entry point: end-to-end training of HACE-VD (or CEVD baseline)."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from clustering import construct_pseudo_domains, generate_soft_labels, l2_normalize
from data import EmbeddingDataset, build_dataloader
from evaluation import Evaluator
from models import HACEVDModel
from training import LossConfig, TrainConfig, Trainer

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_split(root: Path, split: str) -> dict:
    return {
        "embeddings": np.load(root / f"{split}_embeddings.npy"),
        "labels":     np.load(root / f"{split}_labels.npy"),
        "projects":   np.load(root / f"{split}_projects.npy", allow_pickle=True),
    }

def build_pseudo_domains(cfg: dict, train_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cl = cfg["clustering"]
    sl = cfg["soft_labels"]
    cache_path = cfg.get("paths", {}).get("pseudo_domains")
    cache_path = Path(cache_path) if cache_path else None

    if cache_path is not None and cache_path.exists():
        print(f"[1/4] Loading cached pseudo-domains from {cache_path} ...")
        cached = np.load(cache_path, allow_pickle=True)
        hard = cached["assignments"].astype(np.int64)
        centers = cached["centers"].astype(np.float32)
        k = int(cached["k"])
        emb_norm = l2_normalize(train_emb) if cl.get("l2_normalize", True) else train_emb
        soft = generate_soft_labels(emb_norm, centers, temperature=sl.get("temperature", 1.0), l2_normalize=False)
        return hard, soft, centers, k

    res = construct_pseudo_domains(
        embeddings=train_emb, method=cl["method"], metric=cl["metric"], k=cl.get("k"),
        min_cluster_size=cl.get("min_cluster_size", 200) or 200, min_samples=cl.get("min_samples", 50) or 50,
        l2_normalize_inputs=cl.get("l2_normalize", True), random_state=cl.get("random_state", 42),
        core_dist_n_jobs=cl.get("n_jobs", -1), use_gpu=cl.get("use_gpu", True),
        umap_components=cl.get("umap_components"), umap_neighbors=cl.get("umap_neighbors", 15),
        noise_threshold=cl.get("noise_threshold", 0.35),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, assignments=res.assignments, centers=res.centers, k=res.k)
        print(f"      -> Saved pseudo-domains cache to {cache_path}")

    emb_norm = l2_normalize(train_emb) if cl.get("l2_normalize", True) else train_emb
    soft = generate_soft_labels(emb_norm, res.centers, temperature=sl.get("temperature", 1.0), l2_normalize=False)
    return res.assignments, soft, res.centers, res.k

def make_dataset(emb, labels, projects, hard=None, soft=None):
    return EmbeddingDataset(
        embeddings=emb.astype(np.float32), labels=labels.astype(np.int64), projects=projects.astype(object),
        hard_domains=hard.astype(np.int64) if hard is not None else None,
        soft_domains=soft.astype(np.float32) if soft is not None else None,
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None: cfg["seed"] = args.seed
    set_seed(cfg.get("seed", 8))
    print(f"[seed] Using seed={cfg.get('seed', 8)}")

    emb_root = Path(cfg["paths"]["embeddings_root"])
    train = load_split(emb_root, "train")
    val   = load_split(emb_root, "val")
    test  = load_split(emb_root, "test")

    hard_tr, soft_tr, centers, k = build_pseudo_domains(cfg, train["embeddings"])
    print(f"      -> {k} pseudo-domains  ({np.bincount(hard_tr)} samples each)")

    def soft_for_split(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        emb_norm = l2_normalize(emb)
        soft = generate_soft_labels(emb_norm, centers, temperature=cfg["soft_labels"].get("temperature", 1.0), l2_normalize=False)
        hard = soft.argmax(axis=1)
        return hard, soft

    hard_val, soft_val = soft_for_split(val["embeddings"])
    hard_te,  soft_te  = soft_for_split(test["embeddings"])

    train_ds = make_dataset(train["embeddings"], train["labels"], train["projects"], hard_tr, soft_tr)
    val_ds   = make_dataset(val["embeddings"],   val["labels"],   val["projects"],   hard_val, soft_val)
    test_ds  = make_dataset(test["embeddings"],  test["labels"],  test["projects"],  hard_te,  soft_te)

    train_loader = build_dataloader(train_ds, batch_size=cfg["training"]["batch_size"], oversample=cfg["training"].get("oversample_minority", True), shuffle=False)
    val_loader = build_dataloader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = build_dataloader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    print("[2/4] Instantiating model...")
    model = HACEVDModel(
        embedding_dim=cfg["backbone"]["embedding_dim"], num_domains=k,
        expert_hidden=cfg["experts"]["hidden_dim"], expert_dropout=cfg["experts"]["dropout"], expert_activation=cfg["experts"]["activation"],
        domain_hidden=cfg["domain_classifier"]["hidden_dim"], domain_dropout=cfg["domain_classifier"]["dropout"],
        fusion_type=cfg["fusion"]["type"], attention_heads=cfg["fusion"].get("num_heads") or 4, attention_dim=cfg["fusion"].get("attention_dim") or 128,
        use_projection=cfg["loss"]["contrastive_weight"] > 0, attention_supervision=cfg["fusion"].get("attention_supervision", "none"),
        tau_init=cfg["fusion"].get("tau_init", 1.0), alpha_init=cfg["fusion"].get("alpha_init", 0.0), gate_l2=cfg["fusion"].get("gate_l2", 1e-3),
    )

    loss_cfg = LossConfig(vuln_weight=cfg["loss"]["vuln_weight"], domain_weight=cfg["loss"]["domain_weight"], contrastive_weight=cfg["loss"]["contrastive_weight"], contrastive_temperature=cfg["loss"]["contrastive_temperature"])
    train_cfg = TrainConfig(num_epochs=cfg["training"]["num_epochs"], learning_rate=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"], grad_clip=cfg["training"]["grad_clip"], early_stopping_patience=cfg["training"]["early_stopping_patience"])

    seed_output_dir = Path(cfg["paths"]["output_dir"]) / f"seed_{cfg.get('seed', 8)}"
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, train_cfg=train_cfg, loss_cfg=loss_cfg, output_dir=seed_output_dir)

    print("[3/4] Training...")
    trainer.fit()

    print("[4/4] Cross-project evaluation on unseen projects...")
    evaluator = Evaluator(model=model)
    
    # Sử dụng ngưỡng mặc định 0.5
    best_threshold = 0.5
    
    # Đánh giá tập test
    results = evaluator.evaluate(test_loader, threshold=best_threshold)
    print(f"Overall test (threshold={best_threshold:.4f})  F1={results['overall']['f1']*100:.2f}%  P={results['overall']['precision']*100:.2f}%  R={results['overall']['recall']*100:.2f}%  AP={results['overall']['ap']*100:.2f}%")
    for proj, m in results["per_project"].items():
        print(f"  {proj:>15s}  F1={m['f1']*100:.2f}%  P={m['precision']*100:.2f}%  R={m['recall']*100:.2f}%  AP={m['ap']*100:.2f}%")

    import json
    result_record = {
        "config": str(args.config), "seed": cfg.get("seed", 8),
        "overall": results["overall"], "per_project": results["per_project"], "threshold": best_threshold,
    }
    summary_path = Path(cfg["paths"]["output_dir"]) / "all_seeds_results.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_record) + "\n")
    print(f"[seed] Đã lưu kết quả vào {summary_path}")

if __name__ == "__main__":
    main()
