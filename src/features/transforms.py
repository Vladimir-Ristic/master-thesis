"""
Stateless row-wise transforms.

Every function here is a pure function of a single transaction's own fields. No
statistic is estimated from the data, so none of them can transport information
across the train/validation/test boundary. That property is the reason these are
kept apart from ``encoders.py``, which does estimate parameters and therefore may
only ever be fitted on the training period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# --------------------------------------------------------------------------
# Temporal decomposition
# --------------------------------------------------------------------------
def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose TransactionDT into day / hour / weekday plus cyclical encodings.

    TransactionDT is a timedelta in seconds from an undisclosed reference point,
    so ``day`` and ``weekday`` are relative indices, not calendar dates. The
    cyclical sine/cosine pair is used rather than the raw integer so that hour 23
    and hour 0 are adjacent in feature space, which a tree can otherwise only
    approximate with a chain of splits.
    """
    t = df[cfg.TIME].to_numpy(dtype=np.int64)
    df["tx_day"] = (t // cfg.SECONDS_PER_DAY).astype("int32")
    hour = ((t // 3_600) % 24).astype("int16")
    weekday = (df["tx_day"] % 7).astype("int16")

    df["tx_hour"] = hour
    df["tx_weekday"] = weekday
    df["tx_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["tx_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["tx_weekday_sin"] = np.sin(2 * np.pi * weekday / 7).astype("float32")
    df["tx_weekday_cos"] = np.cos(2 * np.pi * weekday / 7).astype("float32")
    # Overnight window, where staffed manual review is thinnest in practice.
    df["tx_is_night"] = ((hour >= 0) & (hour < 6)).astype("int8")
    return df


def normalise_day_counters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monotone D-column day counters into fixed anchor days.

    D1-D15 are timedeltas measured backwards from the transaction, so for a given
    card they grow by one for every day that passes. Under a time-based split the
    evaluation period therefore has systematically larger values than the
    training period purely as a function of when a transaction occurred -- a
    distribution shift introduced by the encoding rather than by fraud behaviour.
    Subtracting the day index yields the (constant) day on which the counter
    started, which is stationary across the split. Both forms are retained: the
    anchor for stability, the raw counter because recency is itself informative.
    """
    for col in cfg.D_COUNTER_COLS:
        if col in df.columns:
            df[f"{col}_anchor"] = (df["tx_day"] - df[col]).astype("float32")
    return df


# --------------------------------------------------------------------------
# Transaction amount
# --------------------------------------------------------------------------
def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose TransactionAmt.

    The log transform compresses the four-order-of-magnitude range documented in
    Section 7.4 so that split points are not dominated by the extreme upper tail.
    The fractional part is retained separately because amounts carrying non-zero
    cents in this dataset are characteristic of foreign-currency conversion,
    whereas domestic pricing clusters on the round values visible as the two
    density peaks in Figure 7.2.
    """
    amt = df["transactionamt"].astype("float64")
    df["amt_log"] = np.log1p(amt).astype("float32")
    cents = (amt - np.floor(amt)).round(6)
    df["amt_cents"] = cents.astype("float32")
    df["amt_is_round"] = (cents == 0).astype("int8")
    df["amt_decimal_digits"] = _decimal_digits(cents.to_numpy())
    return df


def _decimal_digits(cents: np.ndarray, max_places: int = 6) -> np.ndarray:
    """
    Count significant decimal places, vectorised.

    A per-row string formatting call is unusable at 590k rows; stripping trailing
    zeros from the scaled integer gives the same answer in a fixed number of
    array operations.
    """
    scaled = np.where(np.isnan(cents), 0, np.round(cents * 10**max_places)).astype(np.int64)
    digits = np.full(scaled.shape, max_places, dtype=np.int16)
    work = scaled.copy()
    for _ in range(max_places):
        divisible = (work % 10 == 0) & (digits > 0)
        work = np.where(divisible, work // 10, work)
        digits = np.where(divisible, digits - 1, digits)
    out = digits.astype("float32")
    out[np.isnan(cents)] = np.nan
    return out


# --------------------------------------------------------------------------
# Email domains
# --------------------------------------------------------------------------
def _provider(domain: pd.Series) -> pd.Series:
    return domain.str.split(".").str[0]


def _suffix(domain: pd.Series) -> pd.Series:
    parts = domain.str.split(".")
    return parts.str[1:].str.join(".").replace("", np.nan)


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split email domains into provider and country/TLD suffix, and compare them.

    ``gmail.com`` and ``gmail`` appear as separate raw categories, and regional
    variants such as ``yahoo.com.mx`` fragment a single provider across many
    low-count levels. Splitting the string keeps provider identity in one column
    and geography in another, which both reduces cardinality and separates two
    signals that the raw field confounds. Whether purchaser and recipient domains
    agree is a match flag in the same spirit as the M-columns.
    """
    for side in ("p", "r"):
        col = f"{side}_emaildomain"
        if col not in df.columns:
            continue
        domain = df[col].astype("string").str.lower()
        df[f"{side}_email_provider"] = _provider(domain).astype("object")
        df[f"{side}_email_suffix"] = _suffix(domain).astype("object")

    if {"p_emaildomain", "r_emaildomain"}.issubset(df.columns):
        p = df["p_emaildomain"].astype("string").str.lower()
        r = df["r_emaildomain"].astype("string").str.lower()
        both = p.notna() & r.notna()
        match = np.where(both, (p == r), np.nan)
        df["email_domains_match"] = pd.Series(match, index=df.index, dtype="float32")
    return df


# --------------------------------------------------------------------------
# M-column normalisation
# --------------------------------------------------------------------------
def normalise_match_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the T/F match flags to 1/0 while keeping NaN distinct from False.

    M1-M9 are binary match indicators, except M4, which carries three levels and
    is left as a categorical for the encoder to handle. Collapsing NaN into 0
    would assert that an unrecorded check failed; keeping it as NaN lets the
    gradient boosting learner route missing values to whichever side of a split
    the training data supports.
    """
    mapping = {"T": 1.0, "F": 0.0}
    for col in cfg.M_COLS:
        if col in df.columns and col != "m4":
            df[col] = df[col].map(mapping).astype("float32")
    return df


def apply_stateless(df: pd.DataFrame) -> pd.DataFrame:
    """Run every stateless transform in dependency order."""
    df = add_time_parts(df)
    df = normalise_day_counters(df)
    df = add_amount_features(df)
    df = add_email_features(df)
    df = normalise_match_flags(df)
    return df
