"""Compare the serving-side entity aggregates against the trained values."""

import sys
import numpy as np
import pandas as pd

from src.serving.db import connect
from src.serving.history import aggregates_for

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500

v = pd.read_parquet("data/processed/valid_v1.parquet")
sample = v.sample(N, random_state=42)
ids = tuple(int(i) for i in sample.transactionid)

with connect() as conn, conn.cursor() as cur:
    cur.execute(
        "SELECT transactionid, transactiondt, transactionamt, uid_card, uid_card_addr,"
        " uid_account FROM tx_history WHERE transactionid IN %s",
        (ids,),
    )
    rows = {r[0]: r for r in cur.fetchall()}
    recs = [
        aggregates_for(
            cur,
            {"uid_card": rows[t][3], "uid_card_addr": rows[t][4], "uid_account": rows[t][5]},
            rows[t][1],
            float(rows[t][2]),
        )
        for t in ids
    ]

got = pd.DataFrame(recs, index=list(ids)).astype("float32")
exp = sample.set_index("transactionid").reindex(got.index)[got.columns].astype("float32")

print(f"{'column':32s} {'max abs diff':>13s} {'mismatched':>11s}")
total = 0
for c in got.columns:
    a, b = got[c].to_numpy(), exp[c].to_numpy()
    ok = np.isclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True)
    d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
    total += int((~ok).sum())
    print(f"{c:32s} {d.max():13.3e} {int((~ok).sum()):11d}")

print(f"\nrows compared: {N}   mismatched cells: {total}")

for c in ("acct_amt_std_hist", "acct_amt_std_7d", "amt_acct_zscore"):
    a, b = got[c].to_numpy(), exp[c].to_numpy()
    bad = np.where(~np.isclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True))[0]
    if len(bad):
        print(f"\n--- {c} ---")
        for i in bad[:10]:
            print(f"  tid={got.index[i]}  serving={a[i]!r}  training={b[i]!r}")

sys.exit(1 if total else 0)