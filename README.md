# HACE-VD: Hierarchical Adaptive Clustering with Contrastive Ensemble for Cross-Project Vulnerability Detection

This repository contains the implementation of **HACE-VD**, an improved framework
for cross-project software vulnerability detection. HACE-VD extends the recent
**CEVD** baseline (Cao et al., *Future Internet*, 2026) along three axes:

1. **Density-based pseudo-domain construction** with HDBSCAN instead of k-means
   (removes the convex-cluster assumption and the manual `k` selection via the elbow method).
2. **Contrastive expert regularization** that encourages experts to specialize on
   their pseudo-domain while staying discriminative on the shared task.
3. **Multi-head attention fusion** that replaces the linear softmax-weighted sum
   of expert logits with an attention block conditioned on the input embedding.

The codebase is organized so that the original **CEVD** behavior is recoverable
by toggling configuration flags (see `config/default.yaml`). This means a single
run script can reproduce both the baseline (CEVD) and the proposed method (HACE-VD)
for fair comparison.

---

## 1. Repository Layout

```
code/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # central config (toggles CEVD vs HACE-VD)
├── data/
│   ├── data_loader.py        # MegaVul/BigVul/Devign loaders
│   └── preprocessing.py      # tree-sitter parse + dedup
├── models/
│   ├── backbone.py           # CodeBERT / VulBERTa encoder (frozen)
│   ├── expert.py             # MLP expert head
│   ├── domain_classifier.py  # MLP + attention head
│   ├── projection_head.py    # for contrastive loss
│   └── hacevd_model.py       # end-to-end model
├── clustering/
│   ├── pseudo_domain.py      # k-means (CEVD) + HDBSCAN (HACE-VD)
│   └── soft_labels.py        # softmax / temperature soft labels
├── losses/
│   ├── contrastive.py        # supervised contrastive (InfoNCE-style)
│   ├── ensemble.py           # weighted cross-entropy
│   └── domain_loss.py        # soft cross-entropy for domain classifier
├── training/
│   ├── trainer.py
│   └── utils.py
├── evaluation/
│   ├── evaluator.py
│   └── metrics.py
├── scripts/
│   ├── extract_embeddings.py # cache backbone embeddings
│   ├── cluster.py            # offline pseudo-domain construction
│   ├── train.py              # full training pipeline
│   └── evaluate.py           # per-project cross-project evaluation
└── tests/
    └── test_smoke.py         # end-to-end smoke test on tiny synthetic data
```

---

## 2. Setup

Tested on Python 3.10, PyTorch 2.1+, CUDA 12.1 (one RTX 4090 is enough for the
full experiment; the smoke test runs on CPU in under one minute).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the offline smoke test (no GPU required):

```bash
python -m tests.test_smoke
```

---

## 3. Data Preparation

HACE-VD uses the three public datasets that CEVD uses: **Devign**, **BigVul**,
and **MegaVul**. We merge the three sources, drop syntactically invalid functions
with `tree-sitter`, and remove content-hash duplicates. The eight projects
listed in the CEVD paper (GPAC, Xen, PHP, Vim, ImageMagick, Radare, TensorFlow,
OpenSSL) are held out as the cross-project test set.

```bash
python -m scripts.preprocess \
    --devign  data/raw/devign.jsonl \
    --bigvul  data/raw/bigvul.csv  \
    --megavul data/raw/megavul.jsonl \
    --out     data/processed/
```

After preprocessing, each sample is stored as `{id, project, func, label, cwe}`.

---

## 4. Reproducing CEVD (baseline)

```bash
# 1. Cache CodeBERT [CLS] embeddings of every function (train/val/test).
python -m scripts.extract_embeddings \
    --backbone microsoft/codebert-base \
    --input    data/processed/train.jsonl \
    --out-dir  data/embeddings/codebert \
    --split    train

# 2. Offline pseudo-domain construction (CEVD: k-means + cosine, k=5).
python -m scripts.cluster \
    --method  kmeans \
    --metric  cosine \
    --k       5 \
    --emb     data/embeddings/codebert/train_embeddings.npy \
    --out     data/pseudo_domains/cevd_codebert.npz

# 3. End-to-end ensemble training (CEVD configuration).
python -m scripts.train --config config/cevd_codebert.yaml

# 4. Cross-project evaluation on the 8 unseen projects.
python -m scripts.evaluate \
    --config  config/cevd_codebert.yaml \
    --ckpt    runs/cevd_codebert/best.pt \
    --centers data/pseudo_domains/cevd_codebert.npz
```

---

## 5. Running HACE-VD (proposed)

```bash
# 1. Same embedding extraction step (reuse the cache from Section 4 if present).
python -m scripts.extract_embeddings \
    --backbone microsoft/codebert-base \
    --input    data/processed/train.jsonl \
    --out-dir  data/embeddings/codebert \
    --split    train

# 2. HDBSCAN-based pseudo-domain construction (k is determined automatically).
python -m scripts.cluster \
    --method  hdbscan \
    --metric  cosine \
    --min-cluster-size 200 \
    --min-samples 50 \
    --emb     data/embeddings/codebert/train_embeddings.npy \
    --out     data/pseudo_domains/hacevd_codebert.npz

# 3. End-to-end training with contrastive + attention modules.
python -m scripts.train --config config/hacevd_codebert.yaml

# 4. Cross-project evaluation.
python -m scripts.evaluate \
    --config  config/hacevd_codebert.yaml \
    --ckpt    runs/hacevd_codebert/best.pt \
    --centers data/pseudo_domains/hacevd_codebert.npz
```

The `config/hacevd_codebert.yaml` file enables the three additions:
- `clustering.method: hdbscan`
- `loss.contrastive_weight: 0.1`
- `fusion.type: multihead_attention`

Switching these flags to `kmeans`, `0.0`, and `linear_softmax` reproduces CEVD.

---

## 6. Suggested Experimental Protocol

Five independent random seeds (8, 16, 24, 32, 40) as in the CEVD paper.
For each backbone (CodeBERT, VulBERTa) and each method (CEVD, HACE-VD) report
mean ± std of Precision, Recall, and F1 on the 8 held-out projects. Run the
ablations described in `scripts/ablation.sh` (clustering choice, contrastive
weight λ_c, attention heads h).

---

## 7. License & Citation

Released under the MIT license. If you use this code, please cite both the
HACE-VD paper and the CEVD baseline.
