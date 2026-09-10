"""Recover the §8.7 V-group membership by replaying fit_v_reduction's blocking rule."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features import config as cfg

VCOLS  = Path("data/interim/vcols.parquet")
TRAIN  = Path("data/processed/train_v1.parquet")
OUT    = Path("artifacts/explain_v1/v_groups.json")
CHUNK  = 40

# --- 1. the training row set, in vdf order -----------------------------
keys = pq.read_table(VCOLS, columns=[cfg.KEY]).to_pandas()[cfg.KEY]
train_ids = pq.read_table(TRAIN, columns=[cfg.KEY]).to_pandas()[cfg.KEY]
train_pos = np.flatnonzero(keys.isin(set(train_ids)).to_numpy())
print(f"v_train rows: {len(train_pos):,} (expect 410,601)")

# --- 2. the same 150k sample fit_v_reduction drew ----------------------
n = min(len(train_pos), cfg.V_REDUCTION_SAMPLE_ROWS)
sample_pos = train_pos[
    pd.DataFrame(index=range(len(train_pos))).sample(n=n, random_state=42).index.to_numpy()
]
print(f"sample rows : {len(sample_pos):,} (expect {cfg.V_REDUCTION_SAMPLE_ROWS:,})")

# --- 3. NaN count per V column, on that sample -------------------------
v_columns = [c for c in pq.ParquetFile(VCOLS).schema_arrow.names if c != cfg.KEY]
print(f"V columns   : {len(v_columns)} (expect 339)")

nan_counts = {}
for i in range(0, len(v_columns), CHUNK):
    part = v_columns[i:i + CHUNK]
    df = pq.read_table(VCOLS, columns=part).to_pandas().iloc[sample_pos]
    nan_counts.update(df.isna().sum().astype(int).to_dict())
    print(f"  scanned {min(i + CHUNK, len(v_columns))}/{len(v_columns)}", flush=True)

# --- 4. block, then restrict to the retained columns -------------------
retained = [c for c in pq.ParquetFile(TRAIN).schema_arrow.names if c in set(v_columns)]
print(f"retained    : {len(retained)} (expect 143)")

blocks_all = defaultdict(list)
for c in v_columns:
    blocks_all[nan_counts[c]].append(c)

blocks_kept = {k: [c for c in v if c in set(retained)] for k, v in blocks_all.items()}
blocks_kept = {k: v for k, v in sorted(blocks_kept.items()) if v}

merged_from = None
ks = list(blocks_kept)
if len(ks) == 14:
    a, b = ks[-2], ks[-1]
    blocks_kept[a] = blocks_kept[a] + blocks_kept.pop(b)
    blocks_all[a] = blocks_all[a] + blocks_all[b]
    merged_from = [int(a), int(b)]
    print(f"merged high-missingness blocks {a:,} + {b:,} -> one group")

print(f"\nblocks before reduction: {len(blocks_all)}")
print(f"groups after reduction : {len(blocks_kept)}")
print(f"{'group':>6} {'missing_count':>14} {'missing_share':>14} {'before':>7} {'after':>6}")
for i, (mc, cols) in enumerate(blocks_kept.items(), 1):
    print(f"{i:>6} {mc:>14,} {mc/len(sample_pos):>14.4f} "
          f"{len(blocks_all[mc]):>7} {len(cols):>6}")

# --- 5. write it down once ---------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "provenance": {
        "source": "replay of encoders.fit_v_reduction blocking rule",
        "frame": str(VCOLS), "rows_sampled": int(len(sample_pos)),
        "sample_seed": 42, "retained_from": str(TRAIN),
        "merged_from": merged_from,
    },
    "groups": [
        {"group": f"V{i:02d}", "missing_count": int(mc),
         "missing_share": round(mc / len(sample_pos), 6),
         "n_before": len(blocks_all[mc]), "n_after": len(cols), "columns": cols}
        for i, (mc, cols) in enumerate(blocks_kept.items(), 1)
    ],
}, indent=2))
print(f"\nwrote {OUT}")