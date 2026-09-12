"""Demonstrate that the service repopulates the history store it consumes.

Phase 1 snapshots and deletes one day from tx_history; the batch flow is then run
for that day; phase 2 asserts the service wrote back exactly what the Chapter 8
backfill had put there. If the keys differed, every later aggregate on those
accounts would silently diverge.
"""

import argparse
import sys

import pandas as pd

from src.serving.db import connect
from src.serving.history import STORE_COLUMNS

DAY = 154
LO, HI = DAY * 86400, (DAY + 1) * 86400
SNAP = "/tmp/ch11_day154_snapshot.csv"


def read_day() -> pd.DataFrame:
    sql = (
        f"SELECT {', '.join(STORE_COLUMNS)} FROM tx_history"
        " WHERE transactiondt >= %s AND transactiondt < %s ORDER BY transactionid"
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (LO, HI))
        return pd.DataFrame(cur.fetchall(), columns=STORE_COLUMNS)


ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["snapshot", "verify"], required=True)
phase = ap.parse_args().phase

if phase == "snapshot":
    before = read_day()
    before.to_csv(SNAP, index=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM tx_history WHERE transactiondt >= %s AND transactiondt < %s", (LO, HI))
        conn.commit()
        cur.execute("SELECT count(*) FROM tx_history")
        print(f"day {DAY}: {len(before):,} rows snapshotted and deleted; "
              f"store now {cur.fetchone()[0]:,} rows")
else:
    before = pd.read_csv(SNAP)
    after = read_day()
    print(f"before {len(before):,} rows | after {len(after):,} rows")
    if len(before) != len(after):
        print("ROW COUNT DIFFERS"); sys.exit(1)
    bad = []
    for c in STORE_COLUMNS:
        a = before[c].astype(str).to_numpy()
        b = after[c].astype(str).to_numpy()
        n = int((a != b).sum())
        if n:
            bad.append(f"{c}: {n}")
    print("columns differing:", bad or "none")
    sys.exit(1 if bad else 0)