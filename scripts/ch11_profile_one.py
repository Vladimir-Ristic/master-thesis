"""Break the per-transaction cost into stages, without HTTP or a client process."""

import statistics
import sys
import time

import pandas as pd

from src.features import config as fcfg
from src.serving.artifacts import load_artifacts
from src.serving.db import connect
from src.serving.explain import explainer, reason_codes
from src.serving.featurize import NAMED_FIELDS, featurize
from src.serving.schemas import RawTransaction
from src.serving.score import score_frame
from src.serving.history import aggregates_for

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
tid = int(sys.argv[2]) if len(sys.argv) > 2 else 3438309


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
tx = RawTransaction(**payload)
del base, vdf  # free ~115 MB before timing anything

art = load_artifacts()
explainer()

t = {"featurize": [], "score": [], "explain": []}
with connect() as conn, conn.cursor() as cur:
    cur.execute(
        "SELECT uid_card, uid_card_addr, uid_account, transactiondt, transactionamt"
        " FROM tx_history WHERE transactionid = %s", (tid,)
    )
    uc, uca, ua, dt, amt = cur.fetchone()
    uids = {"uid_card": uc, "uid_card_addr": uca, "uid_account": ua}
    t["history_sql"] = []
    for _ in range(N):
        h0 = time.perf_counter()
        aggregates_for(cur, uids, dt, float(amt))
        t["history_sql"].append((time.perf_counter() - h0) * 1000)
        
    for _ in range(N):
        a0 = time.perf_counter()
        X = featurize([tx], art, cur)
        a1 = time.perf_counter()
        score_frame(X, art)
        a2 = time.perf_counter()
        reason_codes(X, art)
        a3 = time.perf_counter()
        t["featurize"].append((a1 - a0) * 1000)
        t["score"].append((a2 - a1) * 1000)
        t["explain"].append((a3 - a2) * 1000)

for k, vals in t.items():
    print(f"{k:10s} median {statistics.median(vals):7.1f} ms | min {min(vals):7.1f} | max {max(vals):7.1f}")
print(f"{'total':10s} median {sum(statistics.median(v) for v in t.values()):7.1f} ms")