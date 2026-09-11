#!/usr/bin/env python3
"""
Chapter 10, Step 3.3 — locate the authoritative V-column grouping.

Read-only. Writes nothing, changes nothing, loads no full DataFrame.
Run from the project root:  python scripts/diagnose_v_groups.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train_v1.parquet"
ARTDIR = ROOT / "artifacts" / "features_v1"
ORACLE = ROOT / "reports" / "tables" / "table_8_5_v_reduction.csv"

V_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def head(title: str) -> None:
    print(f"\n=== {title} ===")


def looks_like_group_map(obj) -> tuple[bool, int, int]:
    """(is_map, n_groups, n_columns) for a dict/list of column-name collections."""
    if isinstance(obj, dict):
        vals = list(obj.values())
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple, set)):
        vals = list(obj)
    else:
        return False, 0, 0
    flat = [c for v in vals for c in v if isinstance(c, str)]
    if not flat or not any(V_RE.match(c) for c in flat):
        return False, 0, 0
    return True, len(vals), len(flat)


# --------------------------------------------------------------------------
# A. The oracle: what Chapter 8 itself reported
# --------------------------------------------------------------------------
head("A. Chapter 8 oracle — reports/tables/table_8_5_v_reduction.csv")
oracle = None
if ORACLE.exists():
    oracle = pd.read_csv(ORACLE)
    print(oracle.to_string(index=False))
    after_col = next((c for c in oracle.columns if c.lower() in {"after", "retained"}), None)
    before_col = next((c for c in oracle.columns if c.lower() in {"before", "n", "count"}), None)
    print(f"\nrows (= groups Ch8 saw): {len(oracle)}")
    if before_col:
        print(f"columns before reduction: {int(oracle[before_col].sum())}  (expect 339)")
    if after_col:
        print(f"columns after reduction:  {int(oracle[after_col].sum())}  (expect 144, "
              f"or 143 if the constant column was already removed)")
        print(f"group sizes after: min {int(oracle[after_col].min())}, "
              f"max {int(oracle[after_col].max())}")
else:
    print(f"NOT FOUND: {ORACLE}")
    print("Without it you have no oracle. Re-run `make ch8-figures` — it is cheap and")
    print("regenerates this table from the fitted encoder without touching data/processed/.")

# --------------------------------------------------------------------------
# B. The fitted encoder — the authoritative artifact
# --------------------------------------------------------------------------
head("B. fitted encoder — artifacts/features_v1/feature_encoder.joblib")
enc = None
cand = sorted(ARTDIR.glob("*.joblib")) if ARTDIR.exists() else []
if not cand:
    print(f"no .joblib in {ARTDIR}")
else:
    import joblib

    # A pickled custom encoder can only be unpickled if its defining module is
    # importable. Without this, joblib.load raises
    #   AttributeError: Can't get attribute '<Class>' on <module '__main__'>
    sys.path.insert(0, str(ROOT))
    try:
        import src.features.encoders as _enc_mod  # noqa: F401
        print("imported src.features.encoders (needed to unpickle the encoder)")
    except Exception as exc:                      # noqa: BLE001
        print(f"could not import src.features.encoders: {type(exc).__name__}: {exc}")

    for p in cand:
        print(f"\n-- {p.name}")
        try:
            obj = joblib.load(p)
        except Exception as exc:                      # noqa: BLE001
            print(f"   load failed: {type(exc).__name__}: {exc}")
            continue
        print(f"   type: {type(obj).__module__}.{type(obj).__name__}")
        state = obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})
        for name, val in state.items():
            if name.startswith("__"):
                continue
            is_map, ng, nc = looks_like_group_map(val)
            size = len(val) if hasattr(val, "__len__") else "-"
            tag = f"  <-- GROUP MAP? {ng} groups, {nc} v-columns" if is_map else ""
            print(f"   .{name:<28} {type(val).__name__:<12} len={size}{tag}")
            if is_map and enc is None:
                enc = val
if enc is not None:
    print("\nFOUND a stored grouping in the encoder. Use it; stop here.")

# --------------------------------------------------------------------------
# C. encoder_summary.json
# --------------------------------------------------------------------------
head("C. artifacts/features_v1/encoder_summary.json")
summ_path = ARTDIR / "encoder_summary.json"
if summ_path.exists():
    summ = json.load(open(summ_path))

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else k)
        else:
            is_map, ng, nc = looks_like_group_map(node)
            size = len(node) if hasattr(node, "__len__") else "-"
            tag = f"  <-- GROUP MAP? {ng} groups, {nc} v-columns" if is_map else ""
            print(f"  {prefix:<40} {type(node).__name__:<8} len={size}{tag}")

    walk(summ)
else:
    print(f"NOT FOUND: {summ_path}")

# --------------------------------------------------------------------------
# D. Recompute from the feature matrix — schema first, never the whole frame
# --------------------------------------------------------------------------
head("D. recompute from data/processed/train_v1.parquet")
if not TRAIN.exists():
    print(f"NOT FOUND: {TRAIN}")
    sys.exit(1)

pf = pq.ParquetFile(TRAIN)
names = list(pf.schema_arrow.names)
nrows = pf.metadata.num_rows
v_cols = [c for c in names if V_RE.match(c)]
print(f"parquet: {nrows:,} rows x {len(names)} columns")
print(f"columns matching ^v\\d+$ : {len(v_cols)}")
near = [c for c in names if c.lower().startswith("v") and not V_RE.match(c)]
print(f"other columns starting with 'v' (correctly excluded): {near}")

if not v_cols:
    print("\nNo V columns matched. Check the naming convention in the list above.")
    sys.exit(1)

# read only the V columns, row-group by row-group, accumulating null counts
missing = pd.Series(0, index=v_cols, dtype="int64")
for rg in range(pf.num_row_groups):
    chunk = pf.read_row_group(rg, columns=v_cols).to_pandas()
    missing = missing.add(chunk.isna().sum(), fill_value=0)
    del chunk
missing = missing.astype("int64")

groups: dict[int, list[str]] = {}
for col, m in missing.items():
    groups.setdefault(int(m), []).append(col)
ordered = sorted(groups.items(), key=lambda kv: kv[0])

print(f"\ndistinct missing counts (= groups): {len(ordered)}")
print(f"{'#':>3}  {'missing_count':>13}  {'share':>7}  {'n_cols':>6}  members")
for i, (m, cols) in enumerate(ordered, 1):
    shown = ", ".join(cols[:6]) + (" ..." if len(cols) > 6 else "")
    print(f"{i:>3}  {m:>13,}  {100 * m / nrows:>6.2f}%  {len(cols):>6}  {shown}")
print(f"\ntotal columns: {sum(len(c) for _, c in ordered)}")
print(f"group sizes: min {min(len(c) for _, c in ordered)}, "
      f"max {max(len(c) for _, c in ordered)}")

# --------------------------------------------------------------------------
# E. Verdict
# --------------------------------------------------------------------------
head("E. verdict")
n_rec = len(ordered)
n_col = sum(len(c) for _, c in ordered)
if oracle is not None:
    n_orc = len(oracle)
    if n_rec == n_orc:
        print(f"PASS  recomputed {n_rec} groups == {n_orc} in Table 8.5.")
    elif n_rec == n_orc - 1:
        print(f"EXPECTED  recomputed {n_rec} groups vs {n_orc} in Table 8.5.")
        print("      One group lost its only retained column when the constant column")
        print("      was dropped (144 -> 143). Report 14 groups and say so in §10.4.")
    else:
        print(f"MISMATCH  recomputed {n_rec} groups vs {n_orc} in Table 8.5.")
        print("      Do NOT proceed. Use the encoder map (B) as the authority.")
else:
    print(f"recomputed {n_rec} groups, {n_col} columns — no oracle available to check against.")
print(f"columns accounted for: {n_col} (Chapter 8 §8.7 says 143 anonymised features)")
