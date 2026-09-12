"""POST one real transaction to the running service."""

import json
import sys

import httpx
import pandas as pd

from src.features import config as fcfg
from src.serving.featurize import NAMED_FIELDS

tid = int(sys.argv[1]) if len(sys.argv) > 1 else 3438309
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/score"


def _clean(x):
    if pd.isna(x):
        return None
    return str(x) if isinstance(x, str) else float(x)


base = pd.read_parquet(fcfg.DATA_INTERIM / "base.parquet")
vdf = pd.read_parquet(fcfg.DATA_INTERIM / "vcols.parquet")
b = base.loc[base[fcfg.KEY] == tid].iloc[0]
v = vdf.loc[vdf[fcfg.KEY] == tid].iloc[0]

payload = {f: _clean(b[f]) for f in NAMED_FIELDS}
payload["transactionid"] = int(tid)
payload["transactiondt"] = int(b[fcfg.TIME])
payload["v"] = {c: _clean(v[c]) for c in vdf.columns if c != fcfg.KEY}

r = httpx.post(url, json=payload, timeout=120)
print(r.status_code)
print(json.dumps(r.json(), indent=2))