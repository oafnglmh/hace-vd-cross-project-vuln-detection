import argparse
import pickle
import numpy as np
import os

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def convert(model):
    print(f"\n--- Converting {model} dataset ---")
    src = rf"E:\Ongoing_Projects\hacevd_code\code\cevd_data\{model}_clean_preprocess"
    dst = rf"E:\Ongoing_Projects\hacevd_code\code\data\embeddings\{model}"
    
    if not os.path.exists(src):
        print(f"Directory {src} does not exist. Skipping.")
        return
        
    os.makedirs(dst, exist_ok=True)
    
    for split in ["train", "val"]:
        fpath = os.path.join(src, f"all_{split}.pkl")
        if not os.path.exists(fpath):
            print(f"MISSING: {fpath}")
            continue
        d = load_pkl(fpath)
        emb = d["embeddings"].astype(np.float32)
        lbl = np.array(d["labels"], dtype=np.int64)
        proj = np.array(["source"] * len(lbl), dtype=object)
        np.save(os.path.join(dst, f"{split}_embeddings.npy"), emb)
        np.save(os.path.join(dst, f"{split}_labels.npy"), lbl)
        np.save(os.path.join(dst, f"{split}_projects.npy"), proj)
        print(split, emb.shape)

    projects_8 = ["gpac", "xen", "php", "vim", "imagemagick", "radare", "tensorflow", "openssl"]
    all_emb, all_lbl, all_proj = [], [], []
    for p in projects_8:
        fpath = os.path.join(src, f"{p}_test.pkl")
        if not os.path.exists(fpath):
            print("MISSING:", fpath)
            continue
        d = load_pkl(fpath)
        all_emb.append(d["embeddings"].astype(np.float32))
        all_lbl.append(np.array(d["labels"], dtype=np.int64))
        all_proj.extend([p] * len(d["labels"]))

    if all_emb:
        np.save(os.path.join(dst, "test_embeddings.npy"), np.concatenate(all_emb, axis=0))
        np.save(os.path.join(dst, "test_labels.npy"), np.concatenate(all_lbl, axis=0))
        np.save(os.path.join(dst, "test_projects.npy"), np.array(all_proj, dtype=object))
        print("test:", np.concatenate(all_emb, axis=0).shape)
    else:
        print("No test files found to merge.")

def main():
    parser = argparse.ArgumentParser(description="Convert pickle datasets to numpy embeddings.")
    parser.add_argument("--model", type=str, choices=["codebert", "vulberta", "both"], default="both",
                        help="Which model's dataset to convert (default: both)")
    args = parser.parse_args()
    
    if args.model in ["codebert", "both"]:
        convert("codebert")
    if args.model in ["vulberta", "both"]:
        convert("vulberta")

if __name__ == "__main__":
    main()