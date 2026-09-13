"""Chapter 13: the frozen measurement specification.

Written and reviewed BEFORE the test partition is opened. Its md5 is recorded in
the seal ledger at the moment of opening, and a protocol check compares the two
afterwards. That is what makes "the test partition was read once, under a
specification fixed in advance" an auditable claim rather than an assertion.

Nothing here reads the test partition. Every comparator is read from a Chapter 9
artifact rather than retyped, for the reason Step 1 made concrete: a value
retyped or reparsed is a value that can drift by one unit in the last place.

Run once:   make ch13-spec
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone

from src.eval import config as C
from src.serving import config as S

SPEC_VERSION = "1.0"


def _deployed_rule_row() -> dict:
    with S.DECISION_RULES_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["rule"] == S.DECISION_RULE:
                return row
    raise RuntimeError(f"rule {S.DECISION_RULE!r} not found")


def _validation_comparators() -> dict:
    """The commissioned operating point, from Chapter 9's own artifacts."""
    row = _deployed_rule_row()
    sel = json.loads(S.SELECTION_SUMMARY_PATH.read_text())
    tp, fp = int(row["tp"]), int(row["fp"])
    fn, tn = int(row["fn"]), int(row["tn"])
    out = {
        "source": "table_9_4_decision_rules.csv + selection_summary.json",
        "arm": sel["selected_arm"],
        "n": tp + fp + fn + tn,
        "flagged": tp + fp,
        "pr_auc": sel["valid_pr_auc"],
        "no_skill_pr_auc": sel["valid_no_skill_pr_auc"],
        "review_rate": float(row["review_rate"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "fpr": fp / (fp + tn),
        "cost_ed": float(row["cost_ed"]),
        "savings_ed": float(row["savings_ed"]),
    }
    hits = sorted((S.REPORTS_ROOT / "tables").rglob("table_9_3_validation.csv"))
    if hits:
        with hits[0].open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["arm_id"] == C.DEPLOYED_ARM:
                    out["roc_auc"] = float(r["valid_roc_auc"])
                    out["brier_calibrated"] = float(r["brier_calibrated"])
                    break
    else:
        print("  WARNING: table_9_3_validation.csv not found; ROC/Brier omitted")
    return out


def build() -> dict:
    return {
        "spec_version": SPEC_VERSION,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chapter": 13,
        "principle": (
            "The test partition is opened once, under this specification. "
            "Whatever these measurements produce is what Chapter 13 reports."
        ),
        "digests": C.EXPECTED_MD5,
        "partition": {
            "day_min": C.TEST_DAY_MIN,
            "day_max": C.TEST_DAY_MAX,
            "expected_rows": C.EXPECTED_TEST_ROWS,
            "source_of_payloads": "src.flows.score_batch.load_day (data/interim)",
            "seal_enforced_by": "day filter, not file access",
        },
        "run": {
            "path": "HTTP POST to the Chapter 11 service, unchanged",
            "order": "strictly ascending by transaction day",
            "batch_size": C.SCORE_BATCH_SIZE,
            "write_back_to_tx_history": True,
            "run_label": C.RUN_LABEL_TEST,
            "labels_joined": "after the decision log is written, never during",
        },
        "decision_policy": {
            "rule": S.DECISION_RULE,
            "threshold": C.operating_threshold(),
            "threshold_hex": C.operating_threshold().hex(),
            "review_literal": C.DECISION_REVIEW,
            "ca_review": C.CA_REVIEW,
            "cost_matrix": "example-dependent (Bahnsen): review costs CA_REVIEW; "
                           "a missed fraud costs its own amount",
        },
        "metrics": {
            "primary": "pr_auc",
            "discrimination": ["pr_auc", "no_skill_pr_auc", "lift_over_no_skill",
                               "roc_auc"],
            "calibration": ["brier", "mean_predicted_probability", "prevalence"],
            "operating_point": ["flagged", "review_rate", "precision", "recall",
                                "fpr"],
            "cost": ["cost_ed", "savings_ed"],
            "note": "PR-AUC is reported with its own no-skill baseline beside it; "
                    "the quantity compared across partitions is lift over baseline, "
                    "because PR-AUC is prevalence-sensitive.",
        },
        "uncertainty": {
            "method": "day-block bootstrap; the resampled unit is the transaction "
                      "day, not the row, because entities and days are shared",
            "n_resamples": C.N_BOOTSTRAP,
            "seed": C.SEED,
            "interval": "percentile, 95%",
            "applied_to": ["pr_auc", "precision", "recall"],
        },
        "rq1": {
            "question": "class imbalance handling, out of sample",
            "arms": C.RQ1_ARMS,
            "model_family": "held fixed at LightGBM; only the imbalance "
                            "treatment varies",
            "refit": "comparison arms refit on the training partition from "
                     "Chapter 9 best_params; each loads its own calibrator",
            "deployed_arm_source": "the registered model, not a refit",
            "resampling": "inside the training fit only; never applied to "
                          "validation or test data",
            "selection_on_test": False,
            "decision_rules_compared": ["default 0.50",
                                        S.DECISION_RULE],
        },
        "rq2": {
            "whole_sample": {"method": "TreeSHAP", "n": C.SHAP_SAMPLE_N,
                             "seed": C.SEED,
                             "compare_to": "Chapter 10 named/anonymised 77.6/22.4"},
            "served": {"reason_code_named_share": True,
                       "named_reason_coverage": True,
                       "compare_to": "Chapter 12 pooled 88.1% and 99.2%"},
            "kernel_shap": False,
        },
        "rq3": {
            "parity_recheck": {"n": C.PARITY_SAMPLE_N, "seed": C.SEED,
                               "compare_to": "Table 11.1: 287 of 290 exact, "
                                             "0 decisions changed"},
            "operational": ["wall_clock", "throughput_ms_per_tx", "failed_requests"],
        },
        "monitoring": {
            "reference": "artifacts/monitor_v1/reference.json, NOT regenerated",
            "signals": ["feature_psi_20", "score_psi", "review_rate",
                        "attribution_named_share", "recall_7day_pooled", "verdict"],
            "emergent_level_variants": [
                "ch12_window_medians_carried_forward",
                "test_window_own_medians",
            ],
            "both_reported": True,
            "label_maturation": "7-day pooling retained even though every test "
                                "label is available, so the lagging signal is "
                                "comparable with Chapter 12's",
        },
        "excluded_deliberately": [
            "no retraining or refit of the deployed model",
            "no threshold reselection",
            "no hindsight-optimal threshold",
            "no fix to the windows.py prefix-sum variance defect",
            "no second read of the test partition",
        ],
        "validation_comparators": _validation_comparators(),
    }


def render(spec: dict) -> str:
    return json.dumps(spec, indent=2, sort_keys=True) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rewrite the spec; refused once the seal is open")
    a = ap.parse_args()

    C.ensure_dirs()
    if C.SEAL_PATH.exists():
        raise SystemExit(
            f"REFUSED: {C.SEAL_PATH} exists. The seal is open and the "
            f"specification is frozen."
        )
    if C.SPEC_PATH.exists() and not a.force:
        raise SystemExit(f"REFUSED: {C.SPEC_PATH} exists. Use --force to rewrite.")

    text = render(build())
    C.SPEC_PATH.write_text(text)
    digest = hashlib.md5(text.encode()).hexdigest()
    print(f"wrote {C.SPEC_PATH}")
    print(f"md5   {digest}")
    print(f"bytes {len(text)}")


if __name__ == "__main__":
    main()