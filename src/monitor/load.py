"""Chapter 12: reading the decision log.

The transaction day is derived from tx_history, never from scored_at - scored_at
is the wall-clock time of the scoring run, so grouping by it would collapse the
whole window into the hour the backfill happened to execute.

Only the monitored columns are projected out of the features JSONB. Missingness
is preserved as NaN: a JSON null means the feature was unknown, not zero, and
reading it as zero is the same defect the serving contract was built to prevent.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd
import sqlalchemy as sa

from src.monitor import config as M



@lru_cache(maxsize=1)
def reference() -> dict:
    if not M.REFERENCE_PATH.exists():
        raise SystemExit(f"no reference at {M.REFERENCE_PATH}; run make ch12-reference first")
    return json.loads(M.REFERENCE_PATH.read_text())


def monitored_features() -> list[str]:
    feats = reference()["monitored_features"]
    bad = [f for f in feats if not (f.isidentifier() and f == f.lower())]
    if bad:
        raise SystemExit(f"unsafe feature names in reference: {bad}")
    return feats


def decision_log(day_min: int | None = None, day_max: int | None = None) -> pd.DataFrame:
    """One row per scored transaction, with the monitored features expanded."""
    feats = monitored_features()
    cols = ",\n           ".join(
        f"(p.features->>'{f}')::float8 AS \"{f}\"" for f in feats
    )
    where = ["p.features IS NOT NULL"]
    params: dict[str, int] = {}
    if day_min is not None:
        where.append("h.transactiondt / :spd >= :dmin")
        params |= {"dmin": day_min}
    if day_max is not None:
        where.append("h.transactiondt / :spd <= :dmax")
        params |= {"dmax": day_max}
    params["spd"] = M.SECONDS_PER_DAY

    sql = f"""
        SELECT (h.transactiondt / :spd)::int AS tx_day,
               p.transactionid,
               p.probability,
               p.probability_raw,
               (p.decision = 'review') AS flagged,
               p.reason_codes,
               {cols}
        FROM predictions p
        JOIN tx_history h USING (transactionid)
        WHERE {' AND '.join(where)}
        ORDER BY tx_day, p.transactionid
    """
    with M.engine().connect() as c:
        df = pd.read_sql(sa.text(sql), c, params=params)

    absent = [f for f in feats if f in df and df[f].isna().all()]
    if absent:
        print(f"  WARNING: monitored features entirely null in the log: {absent}")
    return df


def labels() -> pd.Series:
    """Matured labels, indexed by transactionid.

    In production these arrive days to weeks after the decision, through chargeback
    or manual review. Here the validation partition supplies them, which is what
    lets Section 12.4 quantify the lagging signal the leading ones stand in for.
    """
    v = pd.read_parquet(M.VALID_PARQUET, columns=["transactionid", "isfraud"])
    return v.set_index("transactionid")["isfraud"].astype(int)