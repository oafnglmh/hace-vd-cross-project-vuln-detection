"""Extract and cache [CLS] embeddings from a frozen CodePTM.

Reads a JSONL of preprocessed samples and writes three .npy files
(embeddings, labels, projects) ready for the training script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models import CodePTMBackbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache backbone [CLS] embeddings.")
    parser.add_argument("--backbone", type=str, default="microsoft/codebert-base")
    parser.add_argument("--input",    type=Path, required=True,
                        help="JSONL with fields {func, label, project}")
    parser.add_argument("--out-dir",  type=Path, required=True,
                        help="Directory where embeddings_*.npy will be written")
    parser.add_argument("--split",    type=str, required=True,
                        help="Name to use in the output files (train/val/test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = CodePTMBackbone(args.backbone, max_length=args.max_length, freeze=True).to(device)

    funcs:    list[str] = []
    labels:   list[int] = []
    projects: list[str] = []
    with args.input.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            funcs.append(rec["func"])
            labels.append(int(rec["label"]))
            projects.append(str(rec["project"]))

    embeddings = []
    for i in tqdm(range(0, len(funcs), args.batch_size), desc=f"encode {args.split}"):
        batch = funcs[i : i + args.batch_size]
        emb = backbone.encode_text(batch, device=device).cpu().numpy()
        embeddings.append(emb)
    embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)

    np.save(args.out_dir / f"{args.split}_embeddings.npy", embeddings)
    np.save(args.out_dir / f"{args.split}_labels.npy", np.array(labels, dtype=np.int64))
    np.save(args.out_dir / f"{args.split}_projects.npy", np.array(projects, dtype=object))
    print(f"Saved {len(labels)} embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
