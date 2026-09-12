"""Chapter 11: the decision log.

One row per scored transaction. This table is what makes Chapter 12 possible -
drift measured against a recorded reference rather than a remembered one - and
what makes Chapter 13's test-partition results the deployed system's rather than
a notebook's.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from src.serving.history import STORE_COLUMNS

from psycopg2.extras import Json, execute_batch

from src.serving.artifacts import Artifacts
from src.serving.db import connect, create_tables
from src.serving.schemas import ReasonCode
from src.serving.score import Scored

_INSERT = """
INSERT INTO predictions
    (transactionid, model_name, model_version, probability_raw, probability,
     threshold, decision, reason_codes, features, latency_ms)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_HISTORY = """
INSERT INTO tx_history
    (transactionid, transactiondt, transactionamt, uid_card, uid_card_addr,
     uid_account, productcd_key, devicetype_key)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transactionid) DO NOTHING
"""

def record(
    cur,
    art: Artifacts,
    scored: list[Scored],
    reasons: dict[int, list[ReasonCode]] | None = None,
    features: dict[int, dict] | None = None,
    latency_ms: dict[int, float] | None = None,
) -> int:
    reasons = reasons or {}
    features = features or {}
    latency_ms = latency_ms or {}
    rows = [
        (
            s.transactionid, art.model_name, art.model_version,
            s.probability_raw, s.probability, art.threshold, s.decision,
            Json([r.model_dump() for r in reasons[s.transactionid]])
            if s.transactionid in reasons else None,
            Json(features[s.transactionid]) if s.transactionid in features else None,
            latency_ms.get(s.transactionid),
        )
        for s in scored
    ]
    execute_batch(cur, _INSERT, rows, page_size=500)
    return len(rows)


def _py(v):
    """psycopg2 does not adapt numpy scalars."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def record_history(cur, hist: pd.DataFrame) -> int:
    """A scored transaction becomes history for the next one.

    Without this the store goes stale and every account eventually looks dormant.
    ON CONFLICT DO NOTHING makes a replay idempotent, which matters because the
    batch flow can be re-run for a day.
    """
    rows = [
        tuple(_py(v) for v in rec)
        for rec in hist[STORE_COLUMNS].itertuples(index=False, name=None)
    ]
    execute_batch(cur, _INSERT_HISTORY, rows, page_size=500)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-tables", action="store_true")
    args = ap.parse_args()
    if args.create_tables:
        create_tables()


if __name__ == "__main__":
    main()