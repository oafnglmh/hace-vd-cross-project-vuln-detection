"""Data preprocessing for HACE-VD.

Reproduces the data-preparation pipeline of the CEVD baseline (Cao et al.,
*Future Internet*, 2026):

    1. Merge the three public C/C++ datasets (Devign, BigVul, MegaVul) into a
       single list of records ``{id, project, func, label, cwe}``.
    2. Drop syntactically invalid / incomplete functions with ``tree-sitter``.
    3. Remove content-hash duplicates and label conflicts.
    4. Split by *project*: hold out eight target projects as an unseen
       cross-project test set, then split the remaining source projects 80/20
       into train / validation.

The eight held-out projects match the CEVD paper exactly: GPAC, Xen, PHP, Vim,
ImageMagick, Radare, TensorFlow, OpenSSL.

NOTE ON SCHEMAS. Public releases of these datasets use slightly different field
names across versions. The loaders below accept a list of candidate keys for
each field and fall back gracefully, so you may need to adjust ``*_KEYS`` to
match the exact files you downloaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

# The eight unseen target projects (held out for cross-project testing).
TARGET_PROJECTS: tuple[str, ...] = (
    "GPAC", "Xen", "PHP", "Vim", "ImageMagick", "Radare", "TensorFlow", "OpenSSL",
)

# Candidate field names (first match wins) for the heterogeneous public schemas.
_FUNC_KEYS    = ("func", "processed_func", "func_before", "code", "function")
_LABEL_KEYS   = ("label", "target", "vul", "is_vul", "vulnerable")
_PROJECT_KEYS = ("project", "project_name", "repo", "repo_name", "proj")
_CWE_KEYS     = ("cwe", "CWE ID", "cwe_id", "cwe_ids", "cwes")


@dataclass
class Record:
    id: str
    project: str
    func: str
    label: int
    cwe: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "project": self.project, "func": self.func,
                "label": self.label, "cwe": self.cwe}


@dataclass
class PreprocessStats:
    raw: int = 0
    after_parse: int = 0
    after_dedup: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    removed_invalid: int = 0
    removed_duplicate: int = 0
    per_source: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"raw={self.raw}  "
            f"-{self.removed_invalid} (invalid) -> {self.after_parse}  "
            f"-{self.removed_duplicate} (dup) -> {self.after_dedup}  "
            f"| train={self.n_train} val={self.n_val} test={self.n_test}"
        )


# ---------------------------------------------------------------------------
#  Field extraction
# ---------------------------------------------------------------------------

def _first(record: dict, keys: Iterable[str], default: Any = None) -> Any:
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return default


def _to_label(value: Any) -> Optional[int]:
    """Coerce a heterogeneous label field to {0, 1}, or None if unusable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return 1 if int(value) != 0 else 0
    except (TypeError, ValueError):
        s = str(value).strip().lower()
        if s in {"1", "true", "vulnerable", "vul", "yes"}:
            return 1
        if s in {"0", "false", "non-vulnerable", "nonvul", "no", "clean"}:
            return 0
    return None


def _normalize_record(raw: dict, idx: int, source: str) -> Optional[Record]:
    func = _first(raw, _FUNC_KEYS)
    label = _to_label(_first(raw, _LABEL_KEYS))
    if not func or label is None:
        return None
    project = str(_first(raw, _PROJECT_KEYS, default="unknown")).strip()
    cwe = _first(raw, _CWE_KEYS, default="")
    if isinstance(cwe, (list, tuple)):
        cwe = ";".join(str(c) for c in cwe)
    return Record(
        id=f"{source}_{idx}",
        project=project,
        func=str(func),
        label=label,
        cwe=str(cwe),
    )


# ---------------------------------------------------------------------------
#  Dataset loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, source: str) -> list[Record]:
    out: list[Record] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = _normalize_record(raw, i, source)
            if rec is not None:
                out.append(rec)
    return out


def _load_csv(path: Path, source: str) -> list[Record]:
    out: list[Record] = []
    # BigVul rows can be large; raise the field-size limit defensively.
    csv.field_size_limit(10_000_000)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, raw in enumerate(reader):
            rec = _normalize_record(raw, i, source)
            if rec is not None:
                out.append(rec)
    return out


def load_dataset(path: Path, source: str) -> list[Record]:
    """Load one dataset file, dispatching on extension (.jsonl/.json vs .csv)."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return _load_jsonl(path, source)
    if suffix in {".csv", ".tsv"}:
        return _load_csv(path, source)
    raise ValueError(f"Unsupported file type for {path} (expected .jsonl/.json/.csv)")


# ---------------------------------------------------------------------------
#  tree-sitter syntactic filter
# ---------------------------------------------------------------------------

def _build_parser():
    """Return a tree-sitter C/C++ parser, or None if tree-sitter is absent."""
    try:
        from tree_sitter import Parser  # type: ignore
    except ImportError:
        return None
    language = None
    # Newer tree-sitter-language bindings expose a ready Language object.
    try:
        import tree_sitter_c as tsc  # type: ignore
        from tree_sitter import Language  # type: ignore
        language = Language(tsc.language())
    except Exception:
        try:
            import tree_sitter_cpp as tscpp  # type: ignore
            from tree_sitter import Language  # type: ignore
            language = Language(tscpp.language())
        except Exception:
            return None
    try:
        parser = Parser(language)
    except TypeError:
        parser = Parser()
        parser.set_language(language)
    return parser


def _is_syntactically_valid(parser, func: str) -> bool:
    """A function is kept if it parses without ERROR nodes and looks complete."""
    if not func.strip():
        return False
    try:
        tree = parser.parse(bytes(func, "utf-8"))
    except Exception:
        return False
    root = tree.root_node
    if root.has_error:
        return False
    # Completeness heuristic: a full function definition has balanced braces.
    return func.count("{") > 0 and func.count("{") == func.count("}")


def parse_filter(records: list[Record]) -> tuple[list[Record], int]:
    """Drop syntactically invalid functions. No-op (with warning) if tree-sitter
    is unavailable, so the rest of the pipeline still runs."""
    parser = _build_parser()
    if parser is None:
        print("[preprocess] WARNING: tree-sitter not available; "
              "skipping the syntactic-validity filter.")
        return records, 0
    kept = [r for r in records if _is_syntactically_valid(parser, r.func)]
    return kept, len(records) - len(kept)


# ---------------------------------------------------------------------------
#  Content-hash deduplication
# ---------------------------------------------------------------------------

def _content_hash(func: str) -> str:
    """Hash on whitespace-normalized source so trivially different copies collide."""
    normalized = re.sub(r"\s+", " ", func).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(records: list[Record]) -> tuple[list[Record], int]:
    """Remove exact (content-hash) duplicates and conflicting-label duplicates.

    The first occurrence of each hash is kept. If the same code appears later
    with a *different* label (a label conflict), that copy is dropped.
    """
    seen: dict[str, int] = {}     # hash -> label of the kept copy
    kept: list[Record] = []
    removed = 0
    for r in records:
        h = _content_hash(r.func)
        if h not in seen:
            seen[h] = r.label
            kept.append(r)
        else:
            removed += 1  # duplicate (or label conflict) -> drop
    return kept, removed


# ---------------------------------------------------------------------------
#  Project-based split
# ---------------------------------------------------------------------------

def _is_target(project: str) -> bool:
    p = project.strip().lower()
    return any(p == t.lower() for t in TARGET_PROJECTS)


def project_split(
    records: list[Record],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Hold out the eight target projects as test; split the rest 80/20.

    The split is done over *source records* (not projects) for train/val, while
    the test set is defined purely by project membership to guarantee a genuine
    cross-project evaluation with no leakage.
    """
    test = [r for r in records if _is_target(r.project)]
    source = [r for r in records if not _is_target(r.project)]

    rng = random.Random(seed)
    rng.shuffle(source)
    n_val = int(round(len(source) * val_ratio))
    val = source[:n_val]
    train = source[n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")


def preprocess(
    devign: Optional[Path],
    bigvul: Optional[Path],
    megavul: Optional[Path],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> PreprocessStats:
    """Run the full pipeline and write ``train/val/test.jsonl`` to ``out_dir``."""
    stats = PreprocessStats()
    merged: list[Record] = []
    for path, source in ((devign, "devign"), (bigvul, "bigvul"), (megavul, "megavul")):
        if path is None:
            continue
        recs = load_dataset(Path(path), source)
        stats.per_source[source] = len(recs)
        merged.extend(recs)
    stats.raw = len(merged)
    if stats.raw == 0:
        raise ValueError("No records loaded. Check the input paths and field names.")

    merged, stats.removed_invalid = parse_filter(merged)
    stats.after_parse = len(merged)

    merged, stats.removed_duplicate = deduplicate(merged)
    stats.after_dedup = len(merged)

    train, val, test = project_split(merged, val_ratio=val_ratio, seed=seed)
    stats.n_train, stats.n_val, stats.n_test = len(train), len(val), len(test)

    out_dir = Path(out_dir)
    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val,   out_dir / "val.jsonl")
    _write_jsonl(test,  out_dir / "test.jsonl")

    # A small machine-readable manifest of what we produced.
    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "raw": stats.raw,
                "after_parse": stats.after_parse,
                "after_dedup": stats.after_dedup,
                "removed_invalid": stats.removed_invalid,
                "removed_duplicate": stats.removed_duplicate,
                "n_train": stats.n_train,
                "n_val": stats.n_val,
                "n_test": stats.n_test,
                "per_source": stats.per_source,
                "target_projects": list(TARGET_PROJECTS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stats
