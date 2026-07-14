#!/usr/bin/env bash
# ===========================================================================
#  HACE-VD ablation study.
# ---------------------------------------------------------------------------
#  Runs the four configurations that isolate each contribution, on the
#  CodeBERT backbone, across the five CEVD seeds (8 16 24 32 40):
#
#    (1) CEVD baseline           : kmeans      + linear_softmax       + SupCon 0.0
#    (2) + HDBSCAN               : hdbscan     + linear_softmax       + SupCon 0.0
#    (3) + HDBSCAN + Attention   : hdbscan     + multihead_attention  + SupCon 0.0
#    (4) HACE-VD (full)          : hdbscan     + multihead_attention  + SupCon 0.1
#
#  Each (config x seed) run writes its own training log under runs/ablation/.
#  Embeddings are extracted once and reused by every run.
#
#  Usage:
#     bash scripts/ablation.sh
#
#  Prerequisites: data/processed/{train,val,test}.jsonl produced by
#  `python -m scripts.preprocess ...`.
# ===========================================================================
set -euo pipefail

BACKBONE="microsoft/codebert-base"
SEEDS=(8 16 24 32 40)
EMB_DIR="data/embeddings/codebert"
PROC_DIR="data/processed"
BASE_CFG="config/hacevd_codebert.yaml"   # used as the field template
RUN_ROOT="runs/ablation"
TMP_CFG_DIR="${RUN_ROOT}/_configs"

mkdir -p "${RUN_ROOT}" "${TMP_CFG_DIR}"

# ---------------------------------------------------------------------------
# 1. Extract [CLS] embeddings once (skipped if already cached).
# ---------------------------------------------------------------------------
if [[ ! -f "${EMB_DIR}/train_embeddings.npy" ]]; then
  echo "==> Extracting CodeBERT embeddings (train/val/test)..."
  for split in train val test; do
    python -m scripts.extract_embeddings \
      --backbone "${BACKBONE}" \
      --input    "${PROC_DIR}/${split}.jsonl" \
      --out-dir  "${EMB_DIR}" \
      --split    "${split}"
  done
else
  echo "==> Reusing cached embeddings in ${EMB_DIR}"
fi

# ---------------------------------------------------------------------------
# 2. Helper: write a config variant by overriding four fields + seed + paths.
#    Args: <name> <method> <k> <fusion> <contrastive_weight> <seed> <out_cfg>
# ---------------------------------------------------------------------------
write_config() {
  local name="$1" method="$2" k="$3" fusion="$4" cw="$5" seed="$6" out="$7"
  python - "$BASE_CFG" "$name" "$method" "$k" "$fusion" "$cw" "$seed" "$EMB_DIR" "$RUN_ROOT" "$out" <<'PY'
import sys, yaml
(base, name, method, k, fusion, cw, seed, emb_dir, run_root, out) = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["seed"] = int(seed)
cfg["clustering"]["method"] = method
cfg["clustering"]["k"] = (int(k) if k != "null" else None)
cfg["fusion"]["type"] = fusion
cfg["loss"]["contrastive_weight"] = float(cw)
cfg["paths"]["embeddings_root"] = emb_dir
cfg["paths"]["pseudo_domains"] = f"data/pseudo_domains/{name}_seed{seed}.npz"
cfg["paths"]["output_dir"] = f"{run_root}/{name}/seed{seed}"
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
}

# ---------------------------------------------------------------------------
# 3. Configuration matrix: "name|method|k|fusion|contrastive_weight"
# ---------------------------------------------------------------------------
CONFIGS=(
  "cevd_baseline|kmeans|5|linear_softmax|0.0"
  "hdbscan_only|hdbscan|null|linear_softmax|0.0"
  "hdbscan_attention|hdbscan|null|multihead_attention|0.0"
  "hacevd_full|hdbscan|null|multihead_attention|0.1"
)

# ---------------------------------------------------------------------------
# 4. Run every (config x seed).
# ---------------------------------------------------------------------------
for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r name method k fusion cw <<< "${entry}"
  echo ""
  echo "############################################################"
  echo "# Ablation config: ${name}"
  echo "#   clustering=${method}  k=${k}  fusion=${fusion}  SupCon=${cw}"
  echo "############################################################"
  for seed in "${SEEDS[@]}"; do
    cfg="${TMP_CFG_DIR}/${name}_seed${seed}.yaml"
    write_config "${name}" "${method}" "${k}" "${fusion}" "${cw}" "${seed}" "${cfg}"
    log="${RUN_ROOT}/${name}/seed${seed}.log"
    mkdir -p "$(dirname "${log}")"
    echo "  -> seed ${seed}  (config ${cfg})"
    python -m scripts.train --config "${cfg}" | tee "${log}"
  done
done

echo ""
echo "==> Ablation finished. Per-run logs are under ${RUN_ROOT}/<config>/seedN.log"
echo "    Grep the final 'Overall test F1' line of each log and aggregate the"
echo "    five seeds per config (mean +/- std) for the ablation table."
