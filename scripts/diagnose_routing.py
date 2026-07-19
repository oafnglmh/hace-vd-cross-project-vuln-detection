"""Chan doan: tin hieu dinh tuyen (routing signal) con song khong?

Dat file nay vao thu muc ``scripts/`` cua repo, chay tu THU MUC GOC repo:

    python -m scripts.diagnose_routing --config config/hacevd_codebert.yaml

Script KHONG train lai. No do truc tiep 3 thu:

  PHAN A  Entropy cua soft-label tren train/val/test o nhiet do hien tai.
          -> Tra loi: trong so fuse co bi "phang" (~ trung binh cong) khong?

  PHAN B  Bang quet nhiet do (temperature sweep).
          -> Tra loi: nen dat soft_labels.temperature bang bao nhieu?

  PHAN C  (tuy chon, can --checkpoint) Entropy cua fusion_weights + domain_probs
          THAT su ma model dang dung tren test.

Chi so quan trong nhat: **k_eff = exp(H)** = "so expert huu dung" tren moi mau.
    k_eff ~ K (vd 5.0)  -> moi expert dong gop nhu nhau -> ensemble = trung binh
                           cong -> KHONG co chuyen biet hoa. Day la benh.
    k_eff ~ 1.0         -> hard routing (moi mau 1 expert).
    k_eff ~ 1.5 - 2.5   -> vung lanh manh: co uu tien ro nhung van mem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from clustering import generate_soft_labels, l2_normalize

EPS = 1e-12


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def entropy_rows(p: np.ndarray) -> np.ndarray:
    """Entropy (nats) cua tung hang trong ma tran xac suat (N, K)."""
    q = np.clip(p.astype(np.float64), EPS, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    return -(q * np.log(q)).sum(axis=1)


def describe(soft: np.ndarray, name: str, train_ratios: np.ndarray | None = None) -> dict:
    """In thong ke entropy / k_eff / phan bo argmax cho mot ma tran soft-label."""
    K = soft.shape[1]
    H = entropy_rows(soft)
    H_mean = float(H.mean())
    H_max = float(np.log(K))
    k_eff = float(np.exp(H).mean())
    top1 = soft.max(axis=1)
    srt = np.sort(soft, axis=1)
    gap = srt[:, -1] - srt[:, -2] if K >= 2 else np.zeros(len(soft))

    print(f"\n  [{name}]  N={soft.shape[0]:,}  K={K}")
    print(f"    entropy        : {H_mean:.4f} / {H_max:.4f} (max)   -> H/Hmax = {H_mean / H_max:.3f}")
    print(f"    k_eff = exp(H) : {k_eff:.3f}   (K={K}; ~K nghia la trung binh cong)")
    print(f"    max prob       : mean={top1.mean():.3f}  p5={np.percentile(top1, 5):.3f}  "
          f"p50={np.percentile(top1, 50):.3f}  p95={np.percentile(top1, 95):.3f}")
    print(f"    gap top1-top2  : mean={gap.mean():.4f}  p50={np.percentile(gap, 50):.4f}")

    hist = np.bincount(soft.argmax(axis=1), minlength=K).astype(np.float64)
    hist /= hist.sum()
    print(f"    argmax ratios  : [{', '.join(f'{v:.3f}' for v in hist)}]")
    if train_ratios is not None:
        l1 = float(np.abs(hist - train_ratios).sum())
        print(f"    vs train ratios: [{', '.join(f'{v:.3f}' for v in train_ratios)}]   L1 drift = {l1:.3f}")

    verdict = ("PHANG - trong so gan deu, ensemble ~ trung binh cong" if H_mean / H_max > 0.95
               else "HOI PHANG" if H_mean / H_max > 0.85
               else "LANH MANH - co uu tien ro" if H_mean / H_max > 0.30
               else "RAT NHON - gan hard routing")
    print(f"    => {verdict}")
    return {"H": H_mean, "H_max": H_max, "k_eff": k_eff, "ratios": hist}


def cosine_report(emb_norm: np.ndarray, centers: np.ndarray, name: str) -> None:
    """Dai gia tri cosine-sim voi cac centroid: goc re cua van de nhiet do."""
    sims = emb_norm @ centers.T
    srt = np.sort(sims, axis=1)
    gap = srt[:, -1] - srt[:, -2] if sims.shape[1] >= 2 else np.zeros(len(sims))
    print(f"\n  [{name}] cosine-sim voi {centers.shape[0]} centroid:")
    print(f"    range   : min={sims.min():.4f}  max={sims.max():.4f}")
    print(f"    per-row : mean={sims.mean():.4f}  std={sims.std():.4f}")
    print(f"    gap top1-top2 : mean={gap.mean():.4f}  p50={np.percentile(gap, 50):.4f}  "
          f"p95={np.percentile(gap, 95):.4f}")
    print(f"    -> softmax(gap/T) voi T=1.0 cho ti le top1/top2 ~ {np.exp(gap.mean()):.3f}x "
          f"(1.0x = hoan toan deu)")


# ---------------------------------------------------------------------------
#  Phan C: entropy cua fusion_weights that (can checkpoint)
# ---------------------------------------------------------------------------

def checkpoint_report(cfg: dict, ckpt_path: Path, test_emb: np.ndarray, k: int) -> None:
    import torch
    from models import HACEVDModel

    print("\n" + "=" * 78)
    print("PHAN C: fusion_weights / domain_probs THAT SU tu checkpoint")
    print("=" * 78)

    model = HACEVDModel(
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

    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] thieu key: {list(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] thua key : {list(unexpected)[:5]}{' ...' if len(unexpected) > 5 else ''}")
    model.eval()

    fw_all, dp_all = [], []
    with torch.no_grad():
        for i in range(0, len(test_emb), 4096):
            batch = torch.from_numpy(test_emb[i:i + 4096]).float()
            out = model(batch)
            fw_all.append(out.fusion_weights.cpu().numpy())
            dp_all.append(out.domain_probs.cpu().numpy())
    fusion_weights = np.concatenate(fw_all, axis=0)
    domain_probs = np.concatenate(dp_all, axis=0)

    describe(domain_probs, "TEST domain_probs (do classifier sinh ra)")
    describe(fusion_weights, "TEST fusion_weights (trong so THUC SU dung de fuse)")

    corr = np.corrcoef(domain_probs.ravel(), fusion_weights.ravel())[0, 1]
    print(f"\n  Tuong quan domain_probs <-> fusion_weights: r = {corr:.4f}")
    if corr > 0.95:
        print("    => fusion_weights ~ domain_probs: attention KHONG them thong tin gi "
              "(HACE-VD dang thoai hoa ve CEVD + nhieu).")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Chan doan tin hieu dinh tuyen cua HACE-VD/CEVD")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Tuy chon: file .pt de chay PHAN C")
    ap.add_argument("--max-train", type=int, default=60000,
                    help="Lay mau train cho nhanh (mac dinh 60k)")
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    emb_root = Path(cfg["paths"]["embeddings_root"])
    npz_path = Path(cfg["paths"]["pseudo_domains"])
    T_cfg = cfg["soft_labels"].get("temperature", 1.0)

    print("=" * 78)
    print(f"CONFIG : {args.config}")
    print(f"NPZ    : {npz_path}")
    print(f"T hien tai (soft_labels.temperature) = {T_cfg}")
    print("=" * 78)

    cached = np.load(npz_path, allow_pickle=True)
    centers = cached["centers"].astype(np.float32)
    hard_tr = cached["assignments"].astype(np.int64)
    k = int(cached["k"])
    train_ratios = np.bincount(hard_tr, minlength=k).astype(np.float64)
    train_ratios /= train_ratios.sum()
    print(f"\nk = {k}   cluster sizes = {np.bincount(hard_tr, minlength=k).tolist()}")
    print(f"centers.shape = {centers.shape}   (D phai khop voi embedding test!)")

    train_emb = np.load(emb_root / "train_embeddings.npy")
    test_emb = np.load(emb_root / "test_embeddings.npy")
    print(f"train_emb.shape = {train_emb.shape}   test_emb.shape = {test_emb.shape}")

    if centers.shape[1] != test_emb.shape[1]:
        print("\n  *** CANH BAO NGHIEM TRONG ***")
        print(f"  centers co {centers.shape[1]} chieu nhung embedding test co {test_emb.shape[1]} chieu.")
        print("  => centers dang o KHONG GIAN KHAC (vd UMAP) nhung test lai duoc so khop o 768 chieu.")
        print("  Day chinh la bug dinh tuyen. Phai transform test bang dung UMAP model da fit tren train.")
        return

    rng = np.random.default_rng(0)
    idx = rng.choice(len(train_emb), size=min(args.max_train, len(train_emb)), replace=False)
    tr_norm = l2_normalize(train_emb[idx])
    te_norm = l2_normalize(test_emb)

    # ---------------- PHAN A ----------------
    print("\n" + "=" * 78)
    print(f"PHAN A: soft-label o nhiet do HIEN TAI (T = {T_cfg})")
    print("=" * 78)
    cosine_report(tr_norm, centers, "TRAIN")
    cosine_report(te_norm, centers, "TEST")

    soft_tr = generate_soft_labels(embeddings=tr_norm, centers=centers,
                                   temperature=T_cfg, l2_normalize=False)
    soft_te = generate_soft_labels(embeddings=te_norm, centers=centers,
                                   temperature=T_cfg, l2_normalize=False)
    describe(soft_tr, "TRAIN soft-label  (= muc tieu ma domain_classifier hoc theo!)",
             train_ratios=train_ratios)
    describe(soft_te, "TEST soft-label   (= dinh tuyen luc suy dien)", train_ratios=train_ratios)

    # ---------------- PHAN B ----------------
    print("\n" + "=" * 78)
    print("PHAN B: quet nhiet do -> chon T")
    print("=" * 78)
    print(f"\n  {'T':>8s} | {'H/Hmax tr':>10s} | {'k_eff tr':>9s} | {'H/Hmax te':>10s} | "
          f"{'k_eff te':>9s} | {'maxprob te':>10s} | danh gia")
    print("  " + "-" * 84)
    best = []
    for T in args.temps:
        s_tr = generate_soft_labels(embeddings=tr_norm, centers=centers,
                                    temperature=T, l2_normalize=False)
        s_te = generate_soft_labels(embeddings=te_norm, centers=centers,
                                    temperature=T, l2_normalize=False)
        h_tr = entropy_rows(s_tr).mean() / np.log(k)
        h_te = entropy_rows(s_te).mean() / np.log(k)
        ke_tr = float(np.exp(entropy_rows(s_tr)).mean())
        ke_te = float(np.exp(entropy_rows(s_te)).mean())
        mp = float(s_te.max(axis=1).mean())
        tag = ("phang"        if h_te > 0.95
               else "hoi phang" if h_te > 0.85
               else "TOT"       if 0.30 <= h_te <= 0.85
               else "qua nhon")
        if tag == "TOT":
            best.append((abs(ke_te - 2.0), T, ke_te))
        print(f"  {T:>8.3f} | {h_tr:>10.3f} | {ke_tr:>9.3f} | {h_te:>10.3f} | "
              f"{ke_te:>9.3f} | {mp:>10.3f} | {tag}")

    if best:
        best.sort()
        _, T_best, ke_best = best[0]
        print(f"\n  >> Goi y: soft_labels.temperature = {T_best}  (k_eff test = {ke_best:.2f} "
              f"~ 2 expert huu dung / mau)")
    else:
        print("\n  >> Khong co T nao roi vao vung lanh manh. Xem lai cosine range o PHAN A: "
              "neu gap top1-top2 qua nho thi centroid khong tach duoc test -> loi nam o "
              "khong gian embedding / clustering, khong phai nhiet do.")

    # ---------------- PHAN C ----------------
    if args.checkpoint is not None:
        checkpoint_report(cfg, args.checkpoint, test_emb, k)
    else:
        print("\n[i] Bo qua PHAN C (khong co --checkpoint). Them --checkpoint runs/hacevd_codebert/"
              "seed_8/best.pt de do fusion_weights that.")


if __name__ == "__main__":
    main()
