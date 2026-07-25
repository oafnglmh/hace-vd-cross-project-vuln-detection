"""Toi uu nguong quyet dinh + AP (threshold-free) cho HACE-VD / CEVD.

Dat vao scripts/tune_threshold.py, chay tu THU MUC GOC repo:

    # mot config, tat ca seed da train:
    python -m scripts.tune_threshold --config config/hacevd_gated_codebert.yaml

    # so sanh CEVD vs HACE cung luc (khuyen nghi):
    python -m scripts.tune_threshold \
        --config config/cevd_codebert.yaml \
        --config config/hacevd_gated_codebert.yaml \
        --seeds 8 16 24 32 40

Script KHONG train lai. Voi moi seed no:
  1. Nap checkpoint runs/<out>/seed_<s>/best.pt (key "model_state").
  2. Chay val + test, thu XAC SUAT lop duong (khong argmax).
  3. Quet nguong tren VALIDATION (khong bao gio tren test) -> tau*.
  4. Bao cao F1 test @0.5, @tau* toan cuc, @tau* theo pseudo-domain, va AP.
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

TEST_PROJECTS = {"gpac", "xen", "php", "vim", "imagemagick", "radare", "tensorflow", "openssl"}


def build_model(cfg: dict, k: int) -> HACEVDModel:
    kwargs = dict(
        embedding_dim=cfg["backbone"]["embedding_dim"],
        num_domains=k,
        expert_hidden=cfg["experts"]["hidden_dim"],
        expert_dropout=cfg["experts"]["dropout"],
        expert_activation=cfg["experts"]["activation"],
        domain_hidden=cfg["domain_classifier"]["hidden_dim"],
        domain_dropout=cfg["domain_classifier"]["dropout"],
        fusion_type=cfg["fusion"]["type"],
        attention_heads=cfg["fusion"].get("num_heads") or 4,
        attention_dim=cfg["fusion"].get("attention_dim") or 128,
        use_projection=cfg["loss"]["contrastive_weight"] > 0,
    )
    for key in ("tau_init", "alpha_init", "gate_l2"):
        if key in cfg.get("fusion", {}):
            kwargs[key] = cfg["fusion"][key]
    try:
        return HACEVDModel(**kwargs)
    except TypeError:
        for key in ("tau_init", "alpha_init", "gate_l2"):
            kwargs.pop(key, None)
        return HACEVDModel(**kwargs)


def load_split(root: Path, split: str) -> dict:
    return {
        "embeddings": np.load(root / f"{split}_embeddings.npy"),
        "labels":     np.load(root / f"{split}_labels.npy"),
        "projects":   np.load(root / f"{split}_projects.npy", allow_pickle=True),
    }


def make_loader(cfg, split_data, centers, batch_size):
    emb = split_data["embeddings"].astype(np.float32)
    emb_norm = l2_normalize(emb)
    soft = generate_soft_labels(emb_norm, centers,
                                temperature=cfg["soft_labels"].get("temperature", 1.0),
                                l2_normalize=False)
    hard = soft.argmax(axis=1).astype(np.int64)
    ds = EmbeddingDataset(
        embeddings=emb, labels=split_data["labels"].astype(np.int64),
        projects=split_data["projects"].astype(object),
        hard_domains=hard, soft_domains=soft.astype(np.float32))
    return build_dataloader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def collect(model, loader, device):
    P, Y, PROJ, DOM = [], [], [], []
    model.eval()
    for batch in loader:
        emb = batch["embedding"].to(device)
        out = model(emb)
        prob1 = torch.softmax(out.vuln_logits, dim=-1)[:, 1]
        P.append(prob1.cpu().numpy())
        Y.append(batch["label"].cpu().numpy())
        PROJ.extend(batch["project"])
        DOM.append(out.domain_probs.argmax(dim=-1).cpu().numpy())
    return (np.concatenate(P), np.concatenate(Y).astype(int),
            np.array(PROJ, dtype=object), np.concatenate(DOM).astype(int))


def f1_at(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec


def sweep_threshold(y, p, grid):
    best_f, best_t = -1.0, 0.5
    for t in grid:
        f, _, _ = f1_at(y, p, t)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def average_precision(y, p):
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1)
    npos = max(int(y.sum()), 1)
    rec = tp / npos
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def per_domain_thresholds(y_val, p_val, dom_val, k, grid):
    taus = {}
    for c in range(k):
        m = dom_val == c
        if m.sum() < 20 or y_val[m].sum() < 3:
            taus[c] = 0.5
        else:
            taus[c], _ = sweep_threshold(y_val[m], p_val[m], grid)
    return taus


def apply_per_domain(y, p, dom, taus):
    pred = np.zeros_like(y)
    for c, t in taus.items():
        m = dom == c
        pred[m] = (p[m] >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec


def run_config(cfg_path: Path, seeds, grid, device):
    cfg = yaml.safe_load(cfg_path.read_text())
    emb_root = Path(cfg["paths"]["embeddings_root"])
    npz = np.load(Path(cfg["paths"]["pseudo_domains"]), allow_pickle=True)
    centers = npz["centers"].astype(np.float32); k = int(npz["k"])
    bs = cfg["training"]["batch_size"]
    out_dir = Path(cfg["paths"]["output_dir"])

    val = load_split(emb_root, "val")
    test = load_split(emb_root, "test")

    val_projs = {str(x).lower() for x in np.unique(val["projects"])}
    leak = val_projs & TEST_PROJECTS
    if leak:
        print(f"  *** CANH BAO: validation chua project test {sorted(leak)} ***")
        print(f"      -> val KHONG phai source-held-out. Per-domain van hop le,")
        print(f"         nhung dung tune per-test-project tren tap nay.")

    val_loader = make_loader(cfg, val, centers, bs)
    test_loader = make_loader(cfg, test, centers, bs)

    rows = []
    for s in seeds:
        ckpt_path = out_dir / f"seed_{s}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  [seed {s}] khong thay {ckpt_path}, bo qua.")
            continue
        model = build_model(cfg, k).to(device)
        obj = torch.load(ckpt_path, map_location=device)
        state = obj.get("model_state", obj) if isinstance(obj, dict) else obj
        miss, unexp = model.load_state_dict(state, strict=False)
        if miss or unexp:
            print(f"  [seed {s}] [WARN] key mismatch — sua build_model() cho khop. "
                  f"missing={list(miss)[:3]} unexpected={list(unexp)[:3]}")

        p_val, y_val, _, dom_val = collect(model, val_loader, device)
        p_te, y_te, _, dom_te = collect(model, test_loader, device)

        f1_05, _, _ = f1_at(y_te, p_te, 0.5)
        tau_g, _ = sweep_threshold(y_val, p_val, grid)
        f1_g, pg, rg = f1_at(y_te, p_te, tau_g)
        taus_d = per_domain_thresholds(y_val, p_val, dom_val, k, grid)
        f1_d, pd, rd = apply_per_domain(y_te, p_te, dom_te, taus_d)
        ap = average_precision(y_te, p_te)

        rows.append((s, f1_05 * 100, f1_g * 100, f1_d * 100, ap * 100, tau_g))
        print(f"  [seed {s}] F1@0.5={f1_05*100:5.2f}  "
              f"F1@tau*(global,{tau_g:.2f})={f1_g*100:5.2f}  "
              f"F1@per-domain={f1_d*100:5.2f}  AP={ap*100:5.2f}")

    if not rows:
        return None
    arr = np.array([r[1:5] for r in rows])
    mean, std = arr.mean(0), arr.std(0, ddof=1) if len(rows) > 1 else (arr[0], np.zeros(4))
    print(f"  {'MEAN±STD':>10s}: "
          f"F1@0.5={mean[0]:5.2f}±{std[0]:.2f}  "
          f"F1@global={mean[1]:5.2f}±{std[1]:.2f}  "
          f"F1@domain={mean[2]:5.2f}±{std[2]:.2f}  "
          f"AP={mean[3]:5.2f}±{std[3]:.2f}")
    return {"name": cfg_path.stem, "rows": rows, "mean": mean, "std": std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, action="append", required=True,
                    help="Co the lap lai nhieu lan de so sanh nhieu config.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 16, 24, 32, 40])
    ap.add_argument("--grid-step", type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = np.round(np.arange(0.05, 0.95 + 1e-9, args.grid_step), 4)

    results = []
    for cfg_path in args.config:
        print("=" * 78)
        print(f"CONFIG: {cfg_path}")
        print("=" * 78)
        r = run_config(cfg_path, args.seeds, grid, device)
        if r:
            results.append(r)

    if len(results) >= 2:
        print("\n" + "=" * 78)
        print("SO SANH (mean F1 %, cang cao cang tot)")
        print("=" * 78)
        print(f"{'config':>28s} | {'@0.5':>7s} | {'@global':>8s} | {'@domain':>8s} | {'AP':>7s}")
        for r in results:
            m = r["mean"]
            print(f"{r['name']:>28s} | {m[0]:7.2f} | {m[1]:8.2f} | {m[2]:8.2f} | {m[3]:7.2f}")
        print("\nDoc: neu AP(HACE) > AP(CEVD) -> HACE THUC SU tot hon (khong phu thuoc nguong).")
        print("     Cot @global / @domain cho thay F1 sau khi sua diem van hanh.")


if __name__ == "__main__":
    main()
