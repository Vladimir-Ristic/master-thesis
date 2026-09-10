"""The protocol checks, including the case where they must refuse to pass."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import config as C, guards


@pytest.fixture(scope="module")
def report(synthetic_partitions, tmp_path_factory):
    C.TABLES = tmp_path_factory.mktemp("tables")
    C.ARTIFACTS = tmp_path_factory.mktemp("artifacts")
    return guards.run_all()


def test_every_check_is_reported(report):
    assert len(report) == 12
    assert set(report.columns) == {"check", "result", "detail"}


def test_the_protocol_checks_that_do_not_need_artifacts_pass(report):
    needs_artifacts = {
        "All arms searched under the same protocol",
        "Decision threshold selected on validation",
        "Calibrators fitted on out-of-fold training rows",
    }
    subset = report[~report["check"].isin(needs_artifacts)]
    failed = subset[subset["result"] != "Pass"]
    assert failed.empty, failed.to_string(index=False)


def test_missing_search_artifacts_are_reported_as_a_failure(report):
    """A check that cannot pass silently is worth nothing."""
    row = report[report["check"] == "All arms searched under the same protocol"].iloc[0]
    assert row["result"] == "FAIL"
    row = report[report["check"] == "Decision threshold selected on validation"].iloc[0]
    assert row["result"] == "FAIL"
    row = report[
        report["check"] == "Calibrators fitted on out-of-fold training rows"
    ].iloc[0]
    assert row["result"] == "FAIL"


def test_the_sealed_test_partition_check_actually_opens_nothing(report):
    row = report[report["check"] == "Test partition sealed against Chapter 9"].iloc[0]
    assert row["result"] == "Pass"
    assert "refused" in row["detail"]


def test_the_report_is_written_to_a_csv(report):
    path = C.TABLES / "table_9_7_protocol_checks.csv"
    assert path.exists()
    assert len(pd.read_csv(path)) == len(report)
