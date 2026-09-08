"""
Entity keys and transaction-aggregation features.

The IEEE-CIS dataset ships no customer identifier, which is a problem because the
single most consistently productive family of features in the card-fraud
literature -- comparing a transaction against the recent behaviour of the account
that made it -- presupposes one. Whitrow et al. formalise this as the transaction
aggregation strategy, and Bahnsen et al. extend it with per-account
distributional summaries; both require an account key.

Three nested pseudo-account keys are constructed here, from coarse to fine, and
aggregates are computed over each. The keys are heuristics, not ground truth, so
the coarser keys are retained alongside the finest one: where the fine key
fragments a real account, the coarse key still carries usable history.

All aggregates come from ``windows.py`` and are strictly past-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .windows import past_distinct_count, past_window_aggregates, seconds_since_previous


def _as_key(series: pd.Series) -> pd.Series:
    """
    Stable string rendering that keeps missing values as one distinct level.

    Vectorised deliberately: a per-row lambda over three key columns at 590k rows
    costs more than the aggregations themselves.
    """
    values = series.to_numpy()
    if pd.api.types.is_float_dtype(series):
        missing = np.isnan(values)
        rendered = np.char.mod("%d", np.nan_to_num(values, nan=0.0).astype(np.int64))
    else:
        as_str = pd.Series(values).astype("string")
        missing = as_str.isna().to_numpy()
        rendered = as_str.fillna("NA").to_numpy().astype(str)
    return pd.Series(np.where(missing, "NA", rendered), index=series.index, dtype="object")


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build three nested pseudo-account keys.

    ``uid_card``          card1 alone -- the card identifier itself.
    ``uid_card_addr``     card1 + billing region, separating cards reused across
                          regions, which a single card1 value does not.
    ``uid_account``       the above plus ``d1_anchor``. D1 is the number of days
                          since the card's first observed transaction, so
                          ``tx_day - d1`` is the day that card began transacting
                          -- a value that is constant for the life of the card and
                          therefore acts as a stable account-opening stamp. The
                          triple is the closest available proxy for an account.

    None of these is guaranteed to be a real account. They are treated as
    engineered grouping keys whose usefulness is decided empirically in Chapter 9,
    not as recovered identity.
    """
    card1 = _as_key(df["card1"])
    addr1 = _as_key(df["addr1"])
    anchor = _as_key(df["d1_anchor"]) if "d1_anchor" in df.columns else _as_key(df["d1"])

    df["uid_card"] = card1
    df["uid_card_addr"] = card1 + "_" + addr1
    df["uid_account"] = df["uid_card_addr"] + "_" + anchor
    return df


def _add_window_block(
    df: pd.DataFrame,
    key_col: str,
    prefix: str,
    windows: dict[str, int],
    with_mean: tuple[str, ...] = (),
    with_std: tuple[str, ...] = (),
    with_sum: tuple[str, ...] = (),
) -> pd.DataFrame:
    key = df[key_col]
    t = df[cfg.TIME]
    amt = df["transactionamt"]

    for label, seconds in windows.items():
        agg = past_window_aggregates(key, t, amt, seconds)
        df[f"{prefix}_cnt_{label}"] = agg.count
        if label in with_sum:
            df[f"{prefix}_amt_sum_{label}"] = agg.sum
        if label in with_mean:
            df[f"{prefix}_amt_mean_{label}"] = agg.mean
        if label in with_std:
            df[f"{prefix}_amt_std_{label}"] = agg.std
    return df


def add_entity_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach past-only aggregation features for all three entity keys.

    Counts answer velocity questions ("how many transactions has this account
    made in the last hour"), which is the classic signature of a compromised
    card. The amount statistics answer a different question -- whether this
    transaction is typical for this account -- and are used both directly and as
    a ratio, since a $400 charge means something different on an account whose
    history averages $30 than on one averaging $380.
    """
    t = df[cfg.TIME]
    amt = df["transactionamt"]

    df = _add_window_block(
        df, "uid_account", "acct", cfg.WINDOWS,
        with_mean=("24h", "7d"), with_std=("7d",), with_sum=("24h",),
    )
    df = _add_window_block(
        df, "uid_card_addr", "cardaddr", {"24h": cfg.WINDOWS["24h"], "7d": cfg.WINDOWS["7d"]},
        with_mean=("7d",),
    )
    df = _add_window_block(
        df, "uid_card", "card", {"24h": cfg.WINDOWS["24h"], "7d": cfg.WINDOWS["7d"]},
        with_mean=("7d",),
    )

    # Full-history (expanding) summaries, still strictly past-only.
    acct_hist = past_window_aggregates(df["uid_account"], t, amt, None)
    df["acct_cnt_hist"] = acct_hist.count
    df["acct_amt_mean_hist"] = acct_hist.mean
    df["acct_amt_std_hist"] = acct_hist.std

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = amt.to_numpy(dtype="float64") / acct_hist.mean.astype("float64")
        z = (amt.to_numpy(dtype="float64") - acct_hist.mean.astype("float64")) / acct_hist.std.astype("float64")
    df["amt_to_acct_mean_ratio"] = np.where(np.isfinite(ratio), ratio, np.nan).astype("float32")
    df["amt_acct_zscore"] = np.where(np.isfinite(z), z, np.nan).astype("float32")

    card_hist = past_window_aggregates(df["uid_card"], t, amt, None)
    df["card_cnt_hist"] = card_hist.count
    with np.errstate(invalid="ignore", divide="ignore"):
        cratio = amt.to_numpy(dtype="float64") / card_hist.mean.astype("float64")
    df["amt_to_card_mean_ratio"] = np.where(np.isfinite(cratio), cratio, np.nan).astype("float32")

    # Inter-arrival times: the direct measure of transaction velocity.
    df["acct_secs_since_prev"] = seconds_since_previous(df["uid_account"], t)
    df["cardaddr_secs_since_prev"] = seconds_since_previous(df["uid_card_addr"], t)
    df["card_secs_since_prev"] = seconds_since_previous(df["uid_card"], t)

    # Breadth of prior behaviour on the account.
    if "productcd" in df.columns:
        df["acct_distinct_productcd_hist"] = past_distinct_count(
            df["uid_account"], t, df["productcd"]
        )
    if "devicetype" in df.columns:
        df["acct_distinct_devicetype_hist"] = past_distinct_count(
            df["uid_account"], t, df["devicetype"]
        )
    return df


def entity_feature_names(df: pd.DataFrame) -> list[str]:
    prefixes = ("acct_", "cardaddr_", "card_", "amt_to_", "amt_acct_")
    return [c for c in df.columns if c.startswith(prefixes)]
