"""Chapter 9 — locked constants.

Every number that governs the modelling protocol lives here, so the chapter can
cite one file rather than a scatter of literals. Nothing in this module reads
data or imports a model library.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths

ROOT = Path(__file__).resolve().parents[2]

# The search budget is declared further down, but the profile name is needed
# here: a rehearsal on synthetic data must not write into the directories the
# real run reads, or its two-trial results become the real run's cache.
PROFILE = os.environ.get("CH9_PROFILE", "search").lower()
_SUFFIX = "_smoke" if PROFILE == "smoke" else ""

# CH9_DATA_DIR lets the smoke run and the test suite point at the synthetic
# clone without editing anything.
DATA_PROCESSED = Path(os.environ.get("CH9_DATA_DIR", ROOT / "data" / "processed"))
DATA_INTERIM = ROOT / "data" / "interim"
ARTIFACTS = ROOT / "artifacts" / f"models_v1{_SUFFIX}"
FIGURES = ROOT / "reports" / "figures" / f"ch9{_SUFFIX}"
TABLES = ROOT / "reports" / "tables" / f"ch9{_SUFFIX}" if _SUFFIX else ROOT / "reports" / "tables"

PARTITION_FILES = {
    "train": "train_v1.parquet",
    "valid": "valid_v1.parquet",
    "test": "test_v1.parquet",
}

# ---------------------------------------------------------------- schema

TARGET = "isfraud"
KEY = "transactionid"
AMOUNT = "transactionamt"

# Candidate names for the absolute day index produced in Chapter 8. The first
# one present is used as the fold key.
DAY_COL_CANDIDATES = ("tx_day", "day", "transactiondt")

# Chapter 8 excluded the D-column anchors from the feature matrix because an
# absolute day index cannot transfer to accounts first observed after the
# training period. The same argument applies to the day index itself, so it is
# used as a partitioning key and held out of the feature matrix.
DROP_ABSOLUTE_TIME_FROM_FEATURES = True

# ---------------------------------------------------------------- protocol

SEED = 42

# Widest feature lookback in Chapter 8 (acct_*_7d). The purge interval between
# every inner training fold and its validation fold must be at least this wide,
# exactly as the outer embargo is.
MAX_WINDOW_DAYS = 7

N_FOLDS = 3
PURGE_DAYS = 7

# The test partition is read once, in Chapter 13. Chapter 9 must not open it.
ALLOW_TEST_READ = False

# What Chapter 8 recorded in Table 8.1. Checked on load, because the failure
# mode is silent: a leftover synthetic or partial parquet in data/processed/
# runs perfectly well and produces numbers that mean nothing. Disabled
# automatically when CH9_DATA_DIR points somewhere else, and overridable with
# CH9_STRICT_SHAPE=0 if Chapter 8 is ever rebuilt with different boundaries —
# in which case update these figures and Table 8.1 together.
EXPECTED_PARTITIONS = {
    "train": {"rows": 410_601, "days": (1, 119), "prevalence": 0.03512},
    "valid": {"rows": 77_822, "days": (127, 154), "prevalence": 0.03389},
    "test": {"rows": 78_542, "days": (155, 182), "prevalence": 0.03532},
}
STRICT_SHAPE_CHECK = (
    os.environ.get("CH9_STRICT_SHAPE", "1") == "1"
    and "CH9_DATA_DIR" not in os.environ
)

PRIMARY_METRIC = "pr_auc"  # threshold-free; project standing rule

# ---------------------------------------------------------------- cost models

# Example-dependent matrix (Elkan; Bahnsen et al.): a false negative costs the
# full transaction amount, any positive prediction costs a fixed manual review,
# a true negative costs nothing.
CA_REVIEW = 4.00  # USD, per reviewed transaction
CA_SENSITIVITY = (1.0, 2.0, 4.0, 8.0, 16.0)

# Fixed-ratio matrix, reported alongside as an amount-independent robustness
# check against the imbalance literature's usual convention.
FIXED_COST_FN = 100.0
FIXED_COST_FP = 1.0

# ---------------------------------------------------------------- arms

MODELS = ("logreg", "lgbm", "xgb")
IMBALANCE_ARMS = ("natural", "weighted", "smote", "smote_weighted")

# SMOTE configuration is fixed for the grid so that the resampling ratio does
# not confound the model comparison; it is swept separately (Table 9.4).
SMOTE_RATIO = 0.10
SMOTE_K = 5
SMOTE_RATIO_SWEEP = (0.05, 0.10, 0.25, 0.50, 1.00)

# ---------------------------------------------------------------- search budget

_BUDGETS = {
    # trials per boosting arm, LogReg C-grid points, row cap during search
    "smoke": {"trials": 2, "logreg_grid": 2, "row_cap": 20_000, "n_estimators": (50, 150)},
    "search": {"trials": 20, "logreg_grid": 4, "row_cap": None, "n_estimators": (200, 900)},
    "full": {"trials": 60, "logreg_grid": 6, "row_cap": None, "n_estimators": (200, 2000)},
}
if PROFILE not in _BUDGETS:
    raise ValueError(f"CH9_PROFILE must be one of {sorted(_BUDGETS)}, got {PROFILE!r}")

BUDGET = _BUDGETS[PROFILE]
N_TRIALS = BUDGET["trials"]
LOGREG_GRID = [10.0 ** k for k in range(-3, -3 + BUDGET["logreg_grid"])]
ROW_CAP = BUDGET["row_cap"]
N_ESTIMATORS_RANGE = BUDGET["n_estimators"]

# LightGBM/XGBoost histogram resolution. 63 bins costs a little accuracy and
# buys a large amount of wall-clock on a 290-column matrix.
MAX_BIN = 63
N_JOBS = int(os.environ.get("CH9_N_JOBS", "-1"))

# ---------------------------------------------------------------- tracking

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.environ.get("CH9_EXPERIMENT", "ch9-model-development")
REGISTERED_MODEL_NAME = "fraud_detector"

USE_CACHE = os.environ.get("CH9_CACHE", "1") == "1"


def arm_id(model: str, arm: str) -> str:
    return f"{model}__{arm}"


ALL_ARMS = [(m, a) for m in MODELS for a in IMBALANCE_ARMS]
