"""
Strictly past-only windowed aggregation primitives.

Every aggregate produced here is computed over transactions that are (a) on the
same entity key and (b) strictly earlier in time than the row being described.
Transactions sharing an identical timestamp with the target row are excluded as
well, so no aggregate can ever contain information the row itself contributed.

The implementation is fully vectorised. Rows are sorted by (entity key, time);
because the key codes are integers and time is an integer number of seconds, the
composite value ``key * SCALE + t`` is globally monotonic, which lets a single
``np.searchsorted`` locate window boundaries without ever crossing an entity
boundary. Prefix sums then give sum / mean / std in constant time per row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Larger than any TransactionDT value in the dataset (183 days ~ 1.58e7 seconds),
# so key blocks can never overlap in the composite ordering key.
_SCALE = np.int64(10**9)


@dataclass(frozen=True)
class WindowAggregates:
    """Past-only aggregates for one entity key, aligned to the caller's row order."""

    count: np.ndarray
    sum: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def _encode_key(key: pd.Series) -> np.ndarray:
    """Map an arbitrary entity key to dense non-negative integer codes."""
    codes, _ = pd.factorize(key, use_na_sentinel=False)
    return codes.astype(np.int64)


def _prepare(key: pd.Series, t: pd.Series):
    key_codes = _encode_key(key)
    t_sec = np.asarray(t, dtype=np.int64)
    composite = key_codes * _SCALE + t_sec
    order = np.argsort(composite, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return key_codes, t_sec, composite, order, inverse


def past_window_aggregates(
    key: pd.Series,
    t: pd.Series,
    value: pd.Series,
    window_seconds: int | None,
) -> WindowAggregates:
    """
    Aggregate ``value`` over prior transactions sharing ``key``.

    ``window_seconds=None`` gives an expanding (all-history) aggregate; an integer
    restricts the lookback to ``[t - window_seconds, t)``.
    """
    key_codes, t_sec, composite, order, inverse = _prepare(key, t)

    comp_sorted = composite[order]
    t_sorted = t_sec[order]
    key_sorted = key_codes[order]
    val_sorted = np.asarray(value, dtype=np.float64)[order]
    val_sorted = np.nan_to_num(val_sorted, nan=0.0)
    val_present = (~np.isnan(np.asarray(value, dtype=np.float64)[order])).astype(np.float64)

    # Upper boundary: first row of this key at or after the current timestamp.
    # side="left" therefore excludes both the row itself and any exact-timestamp tie.
    hi = np.searchsorted(comp_sorted, key_sorted * _SCALE + t_sorted, side="left")

    if window_seconds is None:
        lo = np.searchsorted(comp_sorted, key_sorted * _SCALE, side="left")
    else:
        lo_time = t_sorted - np.int64(window_seconds)
        lo = np.searchsorted(comp_sorted, key_sorted * _SCALE + lo_time, side="left")

    cs_n = np.concatenate(([0.0], np.cumsum(val_present)))
    cs_v = np.concatenate(([0.0], np.cumsum(val_sorted)))
    cs_v2 = np.concatenate(([0.0], np.cumsum(val_sorted**2)))

    n = cs_n[hi] - cs_n[lo]
    s = cs_v[hi] - cs_v[lo]
    s2 = cs_v2[hi] - cs_v2[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        var = np.where(n > 1, (s2 - (s * s) / np.where(n > 0, n, 1)) / (n - 1), np.nan)
    var = np.where(np.isfinite(var), np.clip(var, 0.0, None), np.nan)
    std = np.sqrt(var)

    total = hi - lo  # row count in window, independent of value nullity
    return WindowAggregates(
        count=total[inverse].astype(np.float32),
        sum=np.where(n > 0, s, np.nan)[inverse].astype(np.float32),
        mean=mean[inverse].astype(np.float32),
        std=std[inverse].astype(np.float32),
    )


def seconds_since_previous(key: pd.Series, t: pd.Series) -> np.ndarray:
    """Seconds elapsed since this entity's previous transaction; NaN if none."""
    key_codes, t_sec, composite, order, inverse = _prepare(key, t)
    comp_sorted = composite[order]
    t_sorted = t_sec[order]
    key_sorted = key_codes[order]

    hi = np.searchsorted(comp_sorted, key_sorted * _SCALE + t_sorted, side="left")
    block_start = np.searchsorted(comp_sorted, key_sorted * _SCALE, side="left")

    prev_idx = hi - 1
    has_prev = prev_idx >= block_start
    safe_idx = np.where(has_prev, prev_idx, 0)
    delta = (t_sorted - t_sorted[safe_idx]).astype(np.float64)
    delta = np.where(has_prev, delta, np.nan)
    return delta[inverse].astype(np.float32)


def past_distinct_count(
    key: pd.Series,
    t: pd.Series,
    other: pd.Series,
) -> np.ndarray:
    """
    Number of distinct ``other`` values seen on this entity strictly before ``t``.

    Implemented as a running count of first occurrences of each (key, other)
    pair, which is exact and avoids materialising per-entity sets.
    """
    key_codes, _ = pd.factorize(key, use_na_sentinel=False)
    other_codes, _ = pd.factorize(other, use_na_sentinel=False)
    # Combine the two code arrays arithmetically rather than materialising tuples,
    # which matters at 590k rows.
    stride = np.int64(other_codes.max()) + np.int64(2)
    pair_codes = key_codes.astype(np.int64) * stride + other_codes.astype(np.int64)
    is_first = _first_occurrence_flag(pd.Series(pair_codes), t)
    agg = past_window_aggregates(key, t, pd.Series(is_first.astype(float)), None)
    return np.nan_to_num(agg.sum, nan=0.0).astype(np.float32)


def _first_occurrence_flag(codes: pd.Series, t: pd.Series) -> np.ndarray:
    """1 for the chronologically first row of each code, 0 otherwise."""
    arr = np.asarray(codes, dtype=np.int64)
    t_sec = np.asarray(t, dtype=np.int64)
    order = np.lexsort((t_sec, arr))
    flag_sorted = np.zeros(len(arr), dtype=np.int8)
    sorted_codes = arr[order]
    flag_sorted[0] = 1
    flag_sorted[1:] = (sorted_codes[1:] != sorted_codes[:-1]).astype(np.int8)
    flag = np.empty(len(arr), dtype=np.int8)
    flag[order] = flag_sorted
    return flag
