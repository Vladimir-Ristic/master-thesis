from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import config as C, data


def test_the_test_partition_cannot_be_read_by_chapter_9():
    assert C.ALLOW_TEST_READ is False
    with pytest.raises(PermissionError, match="Chapter 13"):
        data.load_partition("test")


def test_target_key_and_day_are_excluded_from_the_feature_matrix(train):
    for col in (C.TARGET, C.KEY, "tx_day"):
        assert col not in train.X.columns


def test_amount_is_retained_as_a_feature_and_as_cost_input(train):
    assert C.AMOUNT in train.X.columns
    assert train.amount.min() > 0
    assert np.allclose(train.X[C.AMOUNT].to_numpy(dtype=np.float64), train.amount, rtol=1e-5)


def test_features_are_float32(train):
    assert set(train.X.dtypes.unique()) == {np.dtype("float32")}


def test_partition_reports_its_own_prevalence(train):
    assert 0.0 < train.prevalence < 0.15


def test_a_contract_mismatch_is_refused(train, valid):
    data.assert_same_contract(train, valid)
    broken = data.Partition(
        name="broken",
        X=valid.X.drop(columns=[valid.X.columns[3]]),
        y=valid.y,
        day=valid.day,
        amount=valid.amount,
        key=valid.key,
    )
    with pytest.raises(AssertionError, match="Feature contract differs"):
        data.assert_same_contract(train, broken)


def test_overlapping_partitions_are_refused(train):
    with pytest.raises(AssertionError):
        data.assert_disjoint(train, train)


def test_partitions_must_be_chronologically_ordered(train, valid):
    data.assert_disjoint(train, valid)
    with pytest.raises(AssertionError, match="not strictly before"):
        data.assert_disjoint(valid, train)


def test_a_missing_day_column_is_a_clear_error():
    frame = pd.DataFrame(
        {
            C.TARGET: [0, 1],
            C.AMOUNT: [10.0, 20.0],
            "f1": [0.1, 0.2],
        }
    )
    with pytest.raises(KeyError, match="No day index column"):
        data.from_frame("train", frame)


def test_a_missing_amount_column_is_a_clear_error():
    frame = pd.DataFrame({C.TARGET: [0, 1], "tx_day": [1, 2], "f1": [0.1, 0.2]})
    with pytest.raises(KeyError, match="Section 8.10"):
        data.from_frame("train", frame)


def test_subsample_preserves_time_order(train):
    part = data.subsample(train, 5_000)
    assert part.n == 5_000
    assert np.all(np.diff(part.day) >= 0)
    assert part.day.max() == train.day.max()


def test_a_partition_that_is_not_chapter_8s_is_refused(train, monkeypatch):
    """The silent failure this check exists to prevent: a leftover fixture."""
    monkeypatch.setattr(C, "STRICT_SHAPE_CHECK", True)
    with pytest.raises(AssertionError, match="does not match Chapter 8"):
        data.assert_matches_chapter8(train, path="data/processed/train_v1.parquet")


def test_the_error_names_the_repair(train, monkeypatch):
    monkeypatch.setattr(C, "STRICT_SHAPE_CHECK", True)
    try:
        data.assert_matches_chapter8(train)
    except AssertionError as exc:
        message = str(exc)
    assert "make ch8-build" in message
    assert "rows" in message and "Table 8.1" in message


def test_the_check_is_skipped_outside_the_canonical_directory(train):
    """The synthetic clone is deliberately smaller, so it must not be checked."""
    assert data._is_canonical_directory() is False
    data.load_partition("train")  # would raise if the check applied here
