"""Purged walk-forward cross-validation inside the training partition.

Chapter 8 established the outer split: train, a seven-day embargo, validation,
test. Hyperparameter search needs the same discipline one level down. Random
k-fold inside the training partition would place a pseudo-account's later
transactions in the training fold and its earlier ones in the validation fold,
and would let an entity aggregate computed at time t sit in a fold that also
contains the transactions it summarises.

Folds are therefore expanding-window and separated from their validation block
by a purge interval at least as wide as the widest feature lookback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass(frozen=True)
class Fold:
    index: int
    train_days: tuple[int, int]  # inclusive
    purge_days: tuple[int, int]  # inclusive
    valid_days: tuple[int, int]  # inclusive
    train_idx: np.ndarray
    valid_idx: np.ndarray

    def describe(self) -> str:
        return (
            f"fold {self.index}: train d{self.train_days[0]}-{self.train_days[1]} "
            f"({self.train_idx.size:,} rows) | purge d{self.purge_days[0]}-{self.purge_days[1]} "
            f"| valid d{self.valid_days[0]}-{self.valid_days[1]} ({self.valid_idx.size:,} rows)"
        )


def make_folds(
    day: np.ndarray,
    n_folds: int = C.N_FOLDS,
    purge_days: int = C.PURGE_DAYS,
    max_window_days: int = C.MAX_WINDOW_DAYS,
) -> list[Fold]:
    """Build `n_folds` expanding-window folds with a purge interval.

    The day range is cut into `n_folds + 1` equal blocks. Block 0 is training
    material only; block k (k >= 1) is fold k's validation block, and fold k
    trains on everything up to `purge_days` before that block starts.
    """
    if purge_days < max_window_days:
        raise AssertionError(
            f"Purge interval of {purge_days} days is narrower than the widest "
            f"feature window ({max_window_days} days). Widen the purge, or the "
            "inner boundary is porous by construction."
        )
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")

    day = np.asarray(day)
    d_min, d_max = int(day.min()), int(day.max())
    span = d_max - d_min + 1
    block = span // (n_folds + 1)

    if block <= purge_days:
        raise AssertionError(
            f"A {span}-day training partition cannot carry {n_folds} folds with a "
            f"{purge_days}-day purge: each block would be {block} days. Reduce "
            "n_folds or widen the partition."
        )

    folds: list[Fold] = []
    for k in range(1, n_folds + 1):
        v_start = d_min + k * block
        v_end = (d_min + (k + 1) * block - 1) if k < n_folds else d_max
        p_start = v_start - purge_days
        p_end = v_start - 1
        t_start, t_end = d_min, p_start - 1

        if t_end < t_start:
            raise AssertionError(f"Fold {k} has an empty training block.")

        train_idx = np.flatnonzero((day >= t_start) & (day <= t_end))
        valid_idx = np.flatnonzero((day >= v_start) & (day <= v_end))
        if train_idx.size == 0 or valid_idx.size == 0:
            raise AssertionError(f"Fold {k} has an empty partition on this data.")

        folds.append(
            Fold(
                index=k,
                train_days=(t_start, t_end),
                purge_days=(p_start, p_end),
                valid_days=(v_start, v_end),
                train_idx=train_idx,
                valid_idx=valid_idx,
            )
        )

    assert_folds_valid(folds, purge_days=purge_days, max_window_days=max_window_days)
    return folds


def assert_folds_valid(
    folds: list[Fold],
    purge_days: int = C.PURGE_DAYS,
    max_window_days: int = C.MAX_WINDOW_DAYS,
) -> None:
    """Invariants that must hold for every fold, asserted rather than documented."""
    for f in folds:
        if f.train_days[1] >= f.valid_days[0]:
            raise AssertionError(f"Fold {f.index}: training block reaches into validation.")
        gap = f.valid_days[0] - f.train_days[1] - 1
        if gap < max_window_days:
            raise AssertionError(
                f"Fold {f.index}: purge gap is {gap} days, needs {max_window_days}."
            )
        if gap != purge_days:
            raise AssertionError(
                f"Fold {f.index}: purge gap is {gap} days, expected {purge_days}."
            )
        if np.intersect1d(f.train_idx, f.valid_idx).size:
            raise AssertionError(f"Fold {f.index}: row index appears in both blocks.")

    for a, b in zip(folds, folds[1:]):
        if b.valid_days[0] <= a.valid_days[0]:
            raise AssertionError("Validation blocks are not moving forward in time.")
        if b.train_idx.size < a.train_idx.size:
            raise AssertionError("Training blocks are not expanding.")


def fold_table(folds: list[Fold], y: np.ndarray) -> list[dict]:
    """Per-fold composition, reported as Table 9.1."""
    rows = []
    for f in folds:
        yt, yv = y[f.train_idx], y[f.valid_idx]
        rows.append(
            {
                "fold": f.index,
                "train_days": f"{f.train_days[0]}-{f.train_days[1]}",
                "train_rows": int(f.train_idx.size),
                "train_fraud_rate": float(yt.mean()),
                "purge_days": f"{f.purge_days[0]}-{f.purge_days[1]}",
                "valid_days": f"{f.valid_days[0]}-{f.valid_days[1]}",
                "valid_rows": int(f.valid_idx.size),
                "valid_fraud_rate": float(yv.mean()),
            }
        )
    return rows
