"""Chapter 11 fast loop.

Contract-level checks that need no service, no database and no tracking server -
the same discipline as the Chapter 9 and 10 test batches.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "artifacts" / "models_v1" / "feature_names_v1.json"


def _contract():
    return json.loads(CONTRACT.read_text())


def test_the_feature_contract_is_pinned_and_committed():
    assert CONTRACT.exists(), f"missing {CONTRACT}"
    c = _contract()
    assert c["model"] == "fraud_detector"
    assert c["version"] == 1
    assert c["n_features"] == 290


def test_the_contract_holds_290_unique_feature_names():
    names = _contract()["feature_names"]
    assert len(names) == 290
    assert len(set(names)) == 290


def test_no_target_key_or_absolute_day_index_is_a_feature():
    names = set(_contract()["feature_names"])
    for forbidden in ("isfraud", "transactionid", "tx_day"):
        assert forbidden not in names, forbidden