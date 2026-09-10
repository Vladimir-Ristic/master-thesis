"""The purged walk-forward fold generator is the protocol in code, so it is the
most heavily tested module here. An off-by-one in the purge interval is exactly
the defect that produces an optimistic hyperparameter choice."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import config as C, cv


def test_three_expanding_folds_are_produced(folds):
    assert len(folds) == C.N_FOLDS
    sizes = [f.train_idx.size for f in folds]
    assert sizes == sorted(sizes), "training blocks must expand"


def test_purge_gap_is_exactly_the_configured_width(folds):
    for f in folds:
        gap = f.valid_days[0] - f.train_days[1] - 1
        assert gap == C.PURGE_DAYS


def test_no_row_appears_in_both_blocks_of_a_fold(folds):
    for f in folds:
        assert np.intersect1d(f.train_idx, f.valid_idx).size == 0


def test_training_block_strictly_precedes_validation_block(train, folds):
    for f in folds:
        assert train.day[f.train_idx].max() < train.day[f.valid_idx].min()


def test_no_purged_day_appears_in_any_block(train, folds):
    for f in folds:
        purged = set(range(f.purge_days[0], f.purge_days[1] + 1))
        used = set(train.day[f.train_idx]).union(train.day[f.valid_idx])
        assert not (purged & used), f"fold {f.index} uses purged days"


def test_validation_blocks_move_forward(folds):
    starts = [f.valid_days[0] for f in folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_purge_narrower_than_widest_window_is_refused():
    day = np.repeat(np.arange(1, 120), 5)
    with pytest.raises(AssertionError, match="narrower than the widest"):
        cv.make_folds(day, purge_days=3, max_window_days=7)


def test_too_short_a_partition_is_refused():
    day = np.repeat(np.arange(1, 20), 5)
    with pytest.raises(AssertionError, match="cannot carry"):
        cv.make_folds(day, n_folds=3, purge_days=7)


def test_generator_is_deterministic(train):
    a = cv.make_folds(train.day)
    b = cv.make_folds(train.day)
    for fa, fb in zip(a, b):
        assert fa.train_days == fb.train_days and fa.valid_days == fb.valid_days
        assert np.array_equal(fa.train_idx, fb.train_idx)


def test_fold_table_reports_every_fold(train, folds):
    rows = cv.fold_table(folds, train.y)
    assert len(rows) == len(folds)
    assert all(0 < r["train_fraud_rate"] < 1 for r in rows)
    assert all(r["valid_rows"] > 0 for r in rows)


def test_a_cached_result_from_another_run_is_discarded(train):
    """A two-trial rehearsal must not satisfy a twenty-trial search."""
    from src.models import grid

    current = grid.fingerprint(train)
    other = dict(current, profile="a-different-profile", n_rows=20_000)

    assert grid._stale_reasons({"fingerprint": current}, current) == []
    reasons = grid._stale_reasons({"fingerprint": other}, current)
    assert any("profile" in r for r in reasons)
    assert any("n_rows" in r for r in reasons)
    assert grid._stale_reasons({}, current) == [
        "written before run fingerprints were recorded"
    ]
