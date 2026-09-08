"""
Time-based train / validation / test partition with an embargo interval.

Section 7.6 showed the fraud rate drifting from roughly 2% to above 5% across the
183-day window, which means the dataset is not an i.i.d. sample and a random
split would evaluate the model on the same distribution it was trained on. Under
a chronological split the evaluation period is one the model has never seen,
which is the condition a deployed system actually faces. Bergmeir and Benitez set
out the general argument for chronological evaluation of serially dependent data;
Dal Pozzolo et al. (2018) make the fraud-specific case.

The embargo is the second half of the design. Purging an interval between the
fitting period and the evaluation period is standard practice for financial
machine learning, where a feature computed at time t summarises a window reaching
backwards into the past. Here the widest such window is seven days, so a
seven-day interval is discarded outright: without it, the earliest validation
rows would carry aggregates computed over transactions the model was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass(frozen=True)
class SplitMasks:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    embargo: np.ndarray


def make_splits(day: pd.Series) -> SplitMasks:
    d = day.to_numpy()

    def between(bounds):
        lo, hi = bounds
        return (d >= lo) & (d < hi)

    masks = SplitMasks(
        train=between(cfg.TRAIN_DAYS),
        valid=between(cfg.VALID_DAYS),
        test=between(cfg.TEST_DAYS),
        embargo=between(cfg.EMBARGO_DAYS),
    )
    _assert_well_formed(d, masks)
    return masks


def _assert_well_formed(d: np.ndarray, m: SplitMasks) -> None:
    overlap = (m.train.astype(int) + m.valid.astype(int)
               + m.test.astype(int) + m.embargo.astype(int))
    assert overlap.max() <= 1, "split masks overlap"

    assert cfg.TRAIN_DAYS[1] == cfg.EMBARGO_DAYS[0], "embargo must abut the training period"
    assert cfg.EMBARGO_DAYS[1] == cfg.VALID_DAYS[0], "embargo must abut the validation period"
    embargo_width = cfg.EMBARGO_DAYS[1] - cfg.EMBARGO_DAYS[0]
    required = cfg.MAX_WINDOW_SECONDS / cfg.SECONDS_PER_DAY
    assert embargo_width >= required, (
        f"embargo of {embargo_width}d is narrower than the widest feature "
        f"window ({required}d)"
    )

    if m.train.any() and m.valid.any():
        assert d[m.train].max() < d[m.valid].min(), "train period is not strictly earlier"
    if m.valid.any() and m.test.any():
        assert d[m.valid].max() < d[m.test].min(), "validation period is not strictly earlier"


def split_summary(df: pd.DataFrame, masks: SplitMasks) -> pd.DataFrame:
    """Row counts, day ranges and fraud rates per partition -- Table 8.1."""
    rows = []
    for name, mask in [
        ("Train", masks.train),
        ("Embargo (discarded)", masks.embargo),
        ("Validation", masks.valid),
        ("Test", masks.test),
    ]:
        sub = df.loc[mask]
        rows.append(
            {
                "Partition": name,
                "Days": f"{int(sub['tx_day'].min())}-{int(sub['tx_day'].max())}"
                if len(sub) else "-",
                "Transactions": len(sub),
                "Share of dataset": len(sub) / len(df),
                "Fraud count": int(sub[cfg.TARGET].sum()) if len(sub) else 0,
                "Fraud rate": float(sub[cfg.TARGET].mean()) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)
