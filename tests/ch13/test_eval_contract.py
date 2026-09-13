"""Chapter 13 contract tests.

These assert the things that must be true before the seal opens, and they must
keep passing after it. None of them reads the test partition.
"""

from __future__ import annotations

import csv
import json

import pytest

from src.eval import config as C
from src.eval import seal
from src.serving import config as S
from src.serving.artifacts import load_artifacts

CH9_THRESHOLD_REPR = "0.08704082880739572"


def test_threshold_identical_across_all_sources():
    """Chapter 10 lost a transaction to six decimals. Four sources, one double."""
    t = C.operating_threshold()
    assert repr(t) == CH9_THRESHOLD_REPR
    assert t == C.CH9_THRESHOLD
    assert t == float(load_artifacts().threshold)


def test_threshold_is_not_the_pandas_parse():
    """pandas' default float parser is one ULP low on this value."""
    pd = pytest.importorskip("pandas")
    loose = float(
        pd.read_csv(S.DECISION_RULES_PATH)
        .query("rule == @S.DECISION_RULE")["threshold"]
        .iloc[0]
    )
    assert loose != C.CH9_THRESHOLD, "pandas now parses correctly; simplify config"
    assert abs(loose - C.CH9_THRESHOLD) < 1e-16


def test_sealed_digests_match():
    assert len(seal.verify_digests()) == 4


def test_spec_exists_and_is_frozen():
    spec = json.loads(C.SPEC_PATH.read_text())
    assert spec["decision_policy"]["threshold"] == C.CH9_THRESHOLD
    assert spec["rq1"]["arms"] == C.RQ1_ARMS
    assert spec["rq1"]["selection_on_test"] is False
    assert spec["rq2"]["kernel_shap"] is False
    assert spec["monitoring"]["both_reported"] is True


def test_spec_md5_matches_ledger_once_open():
    if not seal.is_open():
        pytest.skip("seal not yet open")
    assert seal.assert_opened_once()["spec_md5"] == seal.spec_md5()


def test_resampling_never_touches_validation_or_test():
    """The standing rule, as a test rather than a comment."""
    spec = json.loads(C.SPEC_PATH.read_text())
    assert "inside the training fit only" in spec["rq1"]["resampling"]


def test_feature_contract_is_290():
    art = load_artifacts()
    assert len(art.feature_names) == C.S.EXPECTED_N_FEATURES == 290


def test_decision_literals_match_the_deployed_service():
    assert C.DECISION_REVIEW == "review"
    assert C.DECISION_ACCEPT == "accept"


def test_run_labels_are_distinct():
    labels = {C.RUN_LABEL_TEST, C.RUN_LABEL_CH12_WINDOW, C.RUN_LABEL_CH12_INJECTED}
    assert len(labels) == 3


def test_deployed_rule_row_is_unique():
    n = sum(
        1
        for r in csv.DictReader(S.DECISION_RULES_PATH.open(newline=""))
        if r["rule"] == S.DECISION_RULE
    )
    assert n == 1