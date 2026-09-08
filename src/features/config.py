"""
Single source of truth for Chapter 8's feature pipeline.

Nothing in this module computes anything. Every boundary, window length and
column-family definition that the thesis has to defend in prose lives here, so
the written chapter and the executed code cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "features_v1"
REPORT_TABLES = PROJECT_ROOT / "reports" / "tables"
REPORT_FIGURES = PROJECT_ROOT / "reports" / "figures" / "ch8"

FEATURE_VERSION = "v1"

# --------------------------------------------------------------------------
# Raw staging tables (Chapter 7) and CSV fallback
# --------------------------------------------------------------------------
RAW_TX_TABLE = "raw_train_transaction"
RAW_ID_TABLE = "raw_train_identity"
# RAW_CSV_DIR lets the test fixture point the CSV reader at a synthetic clone
# without touching any other setting.
_RAW_CSV_DIR = Path(os.environ.get("CH8_RAW_CSV_DIR", PROJECT_ROOT / "data" / "raw"))
RAW_TX_CSV = _RAW_CSV_DIR / "train_transaction.csv"
RAW_ID_CSV = _RAW_CSV_DIR / "train_identity.csv"

TARGET = "isfraud"
KEY = "transactionid"
TIME = "transactiondt"

SECONDS_PER_DAY = 86_400

# --------------------------------------------------------------------------
# Time-based split (RQ1/RQ3 — locked in Chapter 7, Section 7.6)
# --------------------------------------------------------------------------
# Day indices are floor(TransactionDT / 86400); the dataset spans days 0-182.
TRAIN_DAYS = (0, 120)        # [start, end)
EMBARGO_DAYS = (120, 127)    # discarded entirely — see EMBARGO_RATIONALE
VALID_DAYS = (127, 155)
TEST_DAYS = (155, 183)

EMBARGO_RATIONALE = (
    "The embargo width equals the longest lookback window used by any engineered "
    "feature (7 days). A validation row inside the embargo would carry aggregates "
    "computed over training-period transactions, so its features and the training "
    "rows would share information; removing the interval makes the evaluation "
    "boundary clean by construction rather than by assumption."
)

# --------------------------------------------------------------------------
# Entity aggregation windows (Whitrow et al.; Bahnsen et al.)
# --------------------------------------------------------------------------
WINDOWS = {
    "1h": 3_600,
    "24h": 86_400,
    "7d": 604_800,
}
MAX_WINDOW_SECONDS = max(WINDOWS.values())

# --------------------------------------------------------------------------
# Column families
# --------------------------------------------------------------------------
NAMED_NUMERIC = [
    "transactionamt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
]
NAMED_CATEGORICAL = [
    "productcd", "card4", "card6", "p_emaildomain", "r_emaildomain",
]
C_COLS = [f"c{i}" for i in range(1, 15)]
D_COLS = [f"d{i}" for i in range(1, 16)]
M_COLS = [f"m{i}" for i in range(1, 10)]

# D-columns that behave as monotone day counters and are therefore normalised
# against the day index (see temporal.normalise_day_counters).
D_COUNTER_COLS = ["d1", "d2", "d4", "d6", "d10", "d11", "d12", "d13", "d14", "d15"]

IDENTITY_NUMERIC = [f"id_{i:02d}" for i in range(1, 12)]
IDENTITY_CATEGORICAL = [f"id_{i:02d}" for i in range(12, 39)] + [
    "devicetype", "deviceinfo",
]

# Identity columns above this missingness threshold (measured on the training
# period only) are dropped: Chapter 7, Section 7.7 shows nine of them exceed
# 96% missing on identity-linked rows alone, and they are unnameable besides.
IDENTITY_MISSING_DROP_THRESHOLD = 0.95

# Fields whose absence is structural rather than a collection failure
# (Chapter 7, Section 7.7) and therefore get an explicit indicator column.
STRUCTURAL_MISSING_COLS = ["r_emaildomain", "dist1", "dist2", "addr1", "p_emaildomain"]

# --------------------------------------------------------------------------
# Categorical encoding
# --------------------------------------------------------------------------
# Frequency (count) encoding is applied to high-cardinality identifier-like
# fields; ordinal encoding is applied to every remaining object column.
FREQUENCY_ENCODE_COLS = [
    "card1", "card2", "card3", "card5", "addr1",
    "p_emaildomain", "r_emaildomain", "deviceinfo",
    "uid_card", "uid_card_addr", "uid_account",
]
UNSEEN_CATEGORY_CODE = -1

# --------------------------------------------------------------------------
# V-column reduction
# --------------------------------------------------------------------------
V_CORRELATION_THRESHOLD = 0.75
V_REDUCTION_SAMPLE_ROWS = 150_000   # training-period sample used to fit the reduction

# --------------------------------------------------------------------------
# Leakage guards enforced by validate.py
# --------------------------------------------------------------------------
MAX_SINGLE_FEATURE_AUC = 0.95
FORBIDDEN_IN_FEATURE_MATRIX = {TARGET, KEY}
