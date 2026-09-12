"""Chapter 12 contract tests: the invariants every later step depends on."""

from __future__ import annotations

import sqlalchemy as sa

from src.monitor import config as M


def test_window_sits_inside_validation():
    assert M.VALID_DAY_MIN <= M.WINDOW_START <= M.WINDOW_END <= M.VALID_DAY_MAX


def test_window_never_reaches_the_test_partition():
    assert M.WINDOW_END < 155


def test_thresholds_are_ordered():
    assert 0 < M.PSI_WARN < M.PSI_ALERT < 1


def test_decision_log_is_reachable():
    with M.engine().connect() as c:
        assert c.execute(sa.text("select count(*) from predictions")).scalar() >= 0


def test_reference_paths_resolve():
    M.ensure_dirs()
    assert M.TRAIN_PARQUET.exists()
    assert M.ARTIFACTS_DIR.is_dir()
