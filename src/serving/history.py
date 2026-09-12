"""Chapter 11: the entity history store.

Chapter 8 computed entity aggregates over the whole frame with vectorised prefix
sums. A service scoring one transaction cannot do that, so the same quantities are
recovered from a database. Four properties of windows.py must be reproduced exactly,
and each is a way this could disagree with training without raising:

  * the window is [t - W, t) - strictly earlier, ties on the timestamp excluded;
  * cnt is a row count, while the amount statistics run over non-null amounts;
  * the variance is ddof = 1, so stddev_samp and never stddev_pop;
  * past_distinct_count gives a missing value its own level, while SQL's
    COUNT(DISTINCT) skips NULLs - so the store holds a rendered sentinel instead.

The keys themselves are built by entity.add_entity_keys, not reimplemented here.
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd

from src.features import config as fcfg
from src.features import entity, transforms
from src.serving.db import connect, create_tables

NULL_SENTINEL = "__NULL__"
SEAL_DAY = 155  # days 155-182 are the test partition and are never stored

STORE_COLUMNS = [
    "transactionid", "transactiondt", "transactionamt",
    "uid_card", "uid_card_addr", "uid_account",
    "productcd_key", "devicetype_key",
]

_BLOCK_KEY = {"acct": "uid_account", "cardaddr": "uid_card_addr", "card": "uid_card"}

_AGG_SQL = """
SELECT
    count(*) FILTER (WHERE transactiondt >= %(t)s - 3600)   AS cnt_1h,
    count(*) FILTER (WHERE transactiondt >= %(t)s - 86400)  AS cnt_24h,
    count(*) FILTER (WHERE transactiondt >= %(t)s - 604800) AS cnt_7d,
    count(*)                                                AS cnt_hist,
    count(transactionamt) FILTER (WHERE transactiondt >= %(t)s - 86400)  AS n_24h,
    sum(transactionamt)   FILTER (WHERE transactiondt >= %(t)s - 86400)  AS s_24h,
    count(transactionamt) FILTER (WHERE transactiondt >= %(t)s - 604800) AS n_7d,
    sum(transactionamt)   FILTER (WHERE transactiondt >= %(t)s - 604800) AS s_7d,
    sum(transactionamt * transactionamt)
        FILTER (WHERE transactiondt >= %(t)s - 604800)                   AS q_7d,
    count(transactionamt)                AS n_hist,
    sum(transactionamt)                  AS s_hist,
    sum(transactionamt * transactionamt) AS q_hist,
    max(transactiondt)                   AS prev_dt,
    count(DISTINCT productcd_key)        AS distinct_productcd_hist,
    count(DISTINCT devicetype_key)       AS distinct_devicetype_hist
FROM tx_history
WHERE {key_col} = %(uid)s AND transactiondt < %(t)s
"""

_FIELDS = (
    "cnt_1h cnt_24h cnt_7d cnt_hist n_24h s_24h n_7d s_7d q_7d "
    "n_hist s_hist q_hist prev_dt distinct_productcd_hist distinct_devicetype_hist"
).split()


def _mean(n: float, s: float) -> float:
    """windows.py: np.where(n > 0, s / n, nan), stored as float32."""
    return float(np.float32(s / n)) if n and n > 0 else np.nan


def _sum(n: float, s: float) -> float:
    return float(np.float32(s)) if n and n > 0 else np.nan


def _std(n: float, s: float, q: float) -> float:
    """windows.py: (q - s^2/n)/(n-1), non-finite -> NaN, then clipped at zero."""
    if not n or n <= 1:
        return np.nan
    var = (q - (s * s) / n) / (n - 1)
    if not np.isfinite(var):
        return np.nan
    return float(np.float32(np.sqrt(max(var, 0.0))))


def _f(x) -> float:
    return np.nan if x is None else float(x)


def _ratio(num: float, den: float) -> float:
    """Scalar equivalent of Chapter 8's array division: a non-finite result
    (including division by zero) becomes NaN, as np.errstate + isfinite did."""
    if den is None or not np.isfinite(den) or den == 0.0:
        return float("nan")
    out = float(num) / float(den)
    return out if np.isfinite(out) else float("nan")


def aggregates_for(cur, uids: dict[str, str], t: int, amt: float) -> dict[str, float]:
    """The 25 entity-aggregate columns for one transaction."""
    blocks = {}
    for block, key_col in _BLOCK_KEY.items():
        cur.execute(_AGG_SQL.format(key_col=key_col), {"t": int(t), "uid": uids[key_col]})
        blocks[block] = dict(zip(_FIELDS, (_f(v) for v in cur.fetchone())))

    a, ca, c = blocks["acct"], blocks["cardaddr"], blocks["card"]
    out = {
        "acct_cnt_1h": a["cnt_1h"],
        "acct_cnt_24h": a["cnt_24h"],
        "acct_cnt_7d": a["cnt_7d"],
        "acct_cnt_hist": a["cnt_hist"],
        "acct_amt_sum_24h": _sum(a["n_24h"], a["s_24h"]),
        "acct_amt_mean_24h": _mean(a["n_24h"], a["s_24h"]),
        "acct_amt_mean_7d": _mean(a["n_7d"], a["s_7d"]),
        "acct_amt_std_7d": _std(a["n_7d"], a["s_7d"], a["q_7d"]),
        "acct_amt_mean_hist": _mean(a["n_hist"], a["s_hist"]),
        "acct_amt_std_hist": _std(a["n_hist"], a["s_hist"], a["q_hist"]),
        "acct_distinct_productcd_hist": a["distinct_productcd_hist"],
        "acct_distinct_devicetype_hist": a["distinct_devicetype_hist"],
        "cardaddr_cnt_24h": ca["cnt_24h"],
        "cardaddr_cnt_7d": ca["cnt_7d"],
        "cardaddr_amt_mean_7d": _mean(ca["n_7d"], ca["s_7d"]),
        "card_cnt_24h": c["cnt_24h"],
        "card_cnt_7d": c["cnt_7d"],
        "card_amt_mean_7d": _mean(c["n_7d"], c["s_7d"]),
        "card_cnt_hist": c["cnt_hist"],
        "amt_to_acct_mean_ratio": _ratio(amt, _mean(a["n_hist"], a["s_hist"])),
        "amt_acct_zscore": _ratio(
            amt - _mean(a["n_hist"], a["s_hist"]), _std(a["n_hist"], a["s_hist"], a["q_hist"])
        ),
        "amt_to_card_mean_ratio": _ratio(amt, _mean(c["n_hist"], c["s_hist"])),
    }
    for block, prefix in (("acct", "acct"), ("cardaddr", "cardaddr"), ("card", "card")):
        prev = blocks[block]["prev_dt"]
        out[f"{prefix}_secs_since_prev"] = np.nan if np.isnan(prev) else float(t) - prev
    return out


def _store_frame(upto_day: int = SEAL_DAY) -> pd.DataFrame:
    base = pd.read_parquet(fcfg.DATA_INTERIM / "base.parquet")
    base = base.sort_values(fcfg.TIME, kind="mergesort").reset_index(drop=True)
    base = transforms.apply_stateless(base)
    base = entity.add_entity_keys(base)
    base = base.loc[base["tx_day"] < upto_day].copy()
    base["productcd_key"] = base["productcd"].fillna(NULL_SENTINEL).astype(str)
    base["devicetype_key"] = base["devicetype"].fillna(NULL_SENTINEL).astype(str)
    return base[STORE_COLUMNS]


def backfill(upto_day: int = SEAL_DAY) -> int:
    frame = _store_frame(upto_day)
    buf = io.StringIO()
    frame.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE tx_history")
        cur.copy_expert(
            f"COPY tx_history ({','.join(STORE_COLUMNS)}) FROM STDIN WITH CSV", buf
        )
        conn.commit()
        cur.execute("SELECT count(*), min(transactiondt), max(transactiondt) FROM tx_history")
        n, lo, hi = cur.fetchone()
    print(f"tx_history: {n:,} rows, dt {lo}-{hi} (days {lo // 86400}-{hi // 86400})")
    return n

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-tables", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--upto-day", type=int, default=SEAL_DAY)
    args = ap.parse_args()
    if args.create_tables:
        create_tables()
    if args.backfill:
        backfill(args.upto_day)


if __name__ == "__main__":
    main()