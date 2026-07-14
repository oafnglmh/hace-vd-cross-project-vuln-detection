"""Entry point: build the cleaned, project-split dataset.

Usage (matches the README):

    python -m scripts.preprocess \
        --devign  data/raw/devign.jsonl \
        --bigvul  data/raw/bigvul.csv  \
        --megavul data/raw/megavul.jsonl \
        --out     data/processed/

Writes ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` (records of the form
``{id, project, func, label, cwe}``) plus a ``stats.json`` manifest into the
output directory. At least one of --devign/--bigvul/--megavul must be given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.preprocessing import preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge + clean + split the vulnerability datasets.")
    parser.add_argument("--devign",  type=Path, default=None, help="Devign JSONL")
    parser.add_argument("--bigvul",  type=Path, default=None, help="BigVul CSV")
    parser.add_argument("--megavul", type=Path, default=None, help="MegaVul JSONL")
    parser.add_argument("--out",     type=Path, required=True, help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of source projects used for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffling seed for the train/val split")
    args = parser.parse_args()

    if args.devign is None and args.bigvul is None and args.megavul is None:
        parser.error("Provide at least one of --devign / --bigvul / --megavul.")

    stats = preprocess(
        devign=args.devign,
        bigvul=args.bigvul,
        megavul=args.megavul,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("[preprocess] done.")
    print("[preprocess] " + stats.summary())
    print(f"[preprocess] wrote train/val/test.jsonl + stats.json to {args.out}")


if __name__ == "__main__":
    main()
