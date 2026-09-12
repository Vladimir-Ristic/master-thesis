"""Featurize one real transaction through the serving path."""

import sys

import pandas as pd

from src.features import config as fcfg
from src.serving.artifacts import load_artifacts
from src.serving.db import connect
from src.serving.featurize import NAMED_FIELDS, featurize
from src.serving.schemas import RawTransaction

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 3490865

base = pd.read_parquet(fcfg.DATA_INTERIM / "base.parquet")
vdf = pd.read_parquet(fcfg.DATA_INTERIM / "vcols.parquet")
b = base.loc[base[fcfg.KEY] == tid].iloc[0]
v = vdf.loc[vdf[fcfg.KEY] == tid].iloc[0]

payload = {f: (None if pd.isna(b[f]) else b[f]) for f in NAMED_FIELDS}
payload["v"] = {c: (None if pd.isna(v[c]) else float(v[c])) for c in vdf.columns if c != fcfg.KEY}
tx = RawTransaction(**payload)

art = load_artifacts()
with connect() as conn, conn.cursor() as cur:
    X, _ = featurize([tx], art, cur)

print("shape:", X.shape, "(expect (1, 290))")
print("columns match contract:", list(X.columns) == art.feature_names)
print(X.iloc[0, :8].to_string())