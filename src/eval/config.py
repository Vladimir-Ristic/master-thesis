"""Chapter 13: evaluation configuration.

Constants and paths only. Nothing here reads the test partition, and nothing here
redeclares a value that Chapters 9 through 12 already fixed: the database engine
comes from the Chapter 12 monitor, every path root and the cost constant come from
the Chapter 11 serving config, and the operating threshold is read from the
Chapter 9 decision-rules table rather than retyped.

The two md5 constants were recorded at Step 0 preflight, before the seal was opened.
Protocol checks compare against them afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path

import csv

from src.monitor import config as M
from src.serving import config as S

engine = M.engine

# --- paths -----------------------------------------------------------------
ARTIFACTS_DIR = Path(os.getenv("CH13_ARTIFACTS_DIR", S.ARTIFACTS_ROOT / "eval_v1"))
TABLES_DIR = Path(os.getenv("CH13_TABLES_DIR", S.REPORTS_ROOT / "tables" / "ch13"))
FIGURES_DIR = Path(os.getenv("CH13_FIGURES_DIR", S.REPORTS_ROOT / "figures" / "ch13"))

SPEC_PATH = ARTIFACTS_DIR / "measurement_spec.json"
SEAL_PATH = ARTIFACTS_DIR / "test_seal.json"

TEST_PARQUET = S.PROJECT_ROOT / "data" / "processed" / "test_v1.parquet"
TRAIN_PARQUET = M.TRAIN_PARQUET
VALID_PARQUET = M.VALID_PARQUET

# --- the test partition ----------------------------------------------------
# Days asserted against the parquet at Step 6 rather than trusted from here.
TEST_DAY_MIN = int(os.getenv("CH13_TEST_DAY_MIN", 155))
TEST_DAY_MAX = int(os.getenv("CH13_TEST_DAY_MAX", 182))
EXPECTED_TEST_ROWS = 78542  # 590,540 total less the 511,998 already in tx_history

# --- run labels (Step 2) ---------------------------------------------------
# The decision log is shared with Chapter 12. Every query in either chapter
# filters on a label; nothing is ever deleted.
RUN_LABEL_TEST = "test"
RUN_LABEL_CH12_WINDOW = "ch12_window"
RUN_LABEL_CH12_INJECTED = "ch12_injected"
CH12_WINDOW_DAYS = (M.WINDOW_START, M.WINDOW_END)          # 127-147
CH12_INJECTED_DAYS = (M.SYNTHETIC_DAY_MIN, 149)            # 148-149

# --- decision semantics (Chapter 11) ---------------------------------------
DECISION_REVIEW = "review"
DECISION_ACCEPT = "accept"

# --- Step 0 preflight record -----------------------------------------------
# The seal has two doors. The run reads raw interim source through
# src.flows.score_batch.load_day; the arms and parity checks read the engineered
# test matrix. Both are recorded. The interim files span days 1-182, so the seal
# is enforced by the day filter rather than by file access - which is why the
# ledger records the filter alongside the digests.
EXPECTED_MD5 = {
    "data/interim/base.parquet": "4f4555230c94a152e3781231033bb191",
    "data/interim/vcols.parquet": "8285f675b73a5d9009ee35a128ee83ca",
    "data/processed/test_v1.parquet": "781176784c6ca1a062c351a0f1a2b501",
    "artifacts/monitor_v1/reference.json": "0fafb81b9f3e9391dfc38d3605333f2d",
}
PRE_RUN_LOG_ROWS = 61728
CH9_THRESHOLD = 0.08704082880739572

# --- RQ1: the comparison arms (Fork 2) -------------------------------------
# Model family held fixed; only the imbalance treatment varies. Each arm loads
# its own calibrator - the odds factors differ by up to 27.5x and swapping them
# would compare score scales rather than models.
DEPLOYED_ARM = "lgbm__natural"
COMPARISON_ARMS = ["lgbm__weighted", "lgbm__smote", "lgbm__smote_weighted"]
RQ1_ARMS = [DEPLOYED_ARM, *COMPARISON_ARMS]

# --- measurement constants (frozen into the spec at Step 3) ----------------
SEED = 42
N_BOOTSTRAP = 2000       # day-block resamples; the block is the day, not the row
SHAP_SAMPLE_N = 5000     # reproduces the Chapter 10 procedure exactly
PARITY_SAMPLE_N = 5000   # reproduces the Chapter 11 parity harness
SCORE_BATCH_SIZE = 200   # matches the Chapter 12 window exactly; see score_window.py

CA_REVIEW = S.CA_REVIEW  # 4.0, the Bahnsen review cost fixed in Chapter 9


def operating_threshold() -> float:
    """The deployed threshold, at full precision, from Chapter 9's own table."""
    with S.DECISION_RULES_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["rule"] == S.DECISION_RULE:
                return float(row["threshold"])
    raise RuntimeError(
        f"rule {S.DECISION_RULE!r} not found in {S.DECISION_RULES_PATH}"
    )


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)