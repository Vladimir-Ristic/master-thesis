"""
The aggregation primitives are checked against a naive reference implementation.

This is the single most important test file in Chapter 8. Every entity feature is
built on ``windows.py``, and a vectorised window that is off by one row is exactly
the kind of defect that produces a quietly optimistic model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.windows import (
    past_distinct_count,
    past_window_aggregates,
    seconds_since_previous,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 700
    df = pd.DataFrame(
        {
            # deliberately includes a null key and heavy timestamp collisions
            "key": rng.choice(["a", "b", "c", None, "d"], size=n),
            "t": np.sort(rng.integers(0, 150_000, size=n)),
            "v": rng.normal(100, 25, size=n),
            "o": rng.choice(["x", "y", "z"], size=n),
        }
    )
    df.loc[rng.random(n) < 0.12, "v"] = np.nan
    return df.sample(frac=1.0, random_state=5).reset_index(drop=True)


def _same_key(df: pd.DataFrame, k) -> pd.Series:
    return df["key"].isna() if pd.isna(k) else (df["key"] == k)


def _reference(df: pd.DataFrame, window: int | None):
    counts, sums, means, stds = [], [], [], []
    for _, row in df.iterrows():
        mask = _same_key(df, row["key"]) & (df["t"] < row["t"])
        if window is not None:
            mask &= df["t"] >= row["t"] - window
        vals = df.loc[mask, "v"].dropna()
        counts.append(int(mask.sum()))
        sums.append(vals.sum() if len(vals) else np.nan)
        means.append(vals.mean() if len(vals) else np.nan)
        stds.append(vals.std(ddof=1) if len(vals) > 1 else np.nan)
    return np.array(counts, float), np.array(sums), np.array(means), np.array(stds)


@pytest.mark.parametrize("window", [None, 3_600, 25_000])
def test_aggregates_match_reference(frame, window):
    got = past_window_aggregates(frame["key"], frame["t"], frame["v"], window)
    ref_count, ref_sum, ref_mean, ref_std = _reference(frame, window)

    assert np.allclose(got.count, ref_count)
    assert np.allclose(got.sum, ref_sum, equal_nan=True, atol=1e-2)
    assert np.allclose(got.mean, ref_mean, equal_nan=True, atol=1e-3)
    assert np.allclose(got.std, ref_std, equal_nan=True, atol=1e-2)


def test_current_row_and_timestamp_ties_are_excluded():
    """Two transactions at the same instant must not see each other."""
    df = pd.DataFrame({"key": ["a", "a", "a"], "t": [100, 100, 200], "v": [1.0, 2.0, 3.0]})
    got = past_window_aggregates(df["key"], df["t"], df["v"], None)
    assert got.count[0] == 0
    assert got.count[1] == 0          # tie with row 0, not a predecessor
    assert got.count[2] == 2
    assert got.sum[2] == pytest.approx(3.0)


def test_row_order_does_not_change_results(frame):
    a = past_window_aggregates(frame["key"], frame["t"], frame["v"], 3_600)
    shuffled = frame.sample(frac=1.0, random_state=99)
    b = past_window_aggregates(shuffled["key"], shuffled["t"], shuffled["v"], 3_600)
    b_realigned = pd.Series(b.count, index=shuffled.index).sort_index().to_numpy()
    assert np.allclose(a.count, b_realigned)


def test_seconds_since_previous_matches_reference(frame):
    got = seconds_since_previous(frame["key"], frame["t"])
    ref = []
    for _, row in frame.iterrows():
        prev = frame.loc[_same_key(frame, row["key"]) & (frame["t"] < row["t"]), "t"]
        ref.append(row["t"] - prev.max() if len(prev) else np.nan)
    assert np.allclose(got, np.array(ref, float), equal_nan=True)


def test_past_distinct_count_matches_reference(frame):
    got = past_distinct_count(frame["key"], frame["t"], frame["o"])
    ref = [
        frame.loc[_same_key(frame, row["key"]) & (frame["t"] < row["t"]), "o"].nunique()
        for _, row in frame.iterrows()
    ]
    assert np.allclose(got, np.array(ref, float))


def test_first_transaction_of_an_entity_has_no_history():
    df = pd.DataFrame({"key": ["a", "b", "a"], "t": [10, 20, 30], "v": [1.0, 2.0, 3.0]})
    got = past_window_aggregates(df["key"], df["t"], df["v"], None)
    assert got.count[0] == 0 and np.isnan(got.mean[0])
    assert got.count[1] == 0 and np.isnan(got.mean[1])
    assert got.count[2] == 1
