"""Chapter 12: monitoring configuration.

Paths are environment-overridable by the Chapter 9 mechanism, so the identical
monitoring code runs against a container's volumes or a developer's checkout.
Statistical thresholds are conventional values from the drift literature, stated
as policy rather than tuned - Section 12.3 says so explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa

from src.serving import config as S

# --- paths -----------------------------------------------------------------
ARTIFACTS_DIR = Path(os.getenv("CH12_ARTIFACTS_DIR", S.ARTIFACTS_ROOT / "monitor_v1"))
REPORTS_DIR = Path(os.getenv("CH12_REPORTS_DIR", S.REPORTS_ROOT / "monitor"))
TABLES_DIR = Path(os.getenv("CH12_TABLES_DIR", S.REPORTS_ROOT / "tables" / "ch12"))
FIGURES_DIR = Path(os.getenv("CH12_FIGURES_DIR", S.REPORTS_ROOT / "figures" / "ch12"))
REFERENCE_PATH = ARTIFACTS_DIR / "reference.json"
TRAIN_PARQUET = S.PROJECT_ROOT / "data" / "processed" / "train_v1.parquet"
VALID_PARQUET = S.PROJECT_ROOT / "data" / "processed" / "valid_v1.parquet"

# --- the monitoring window -------------------------------------------------
# Validation is days 127-154; training ends at 119, so every entity aggregate is
# warm throughout. The window stops short of 154 and never touches the test
# partition, which stays sealed until Chapter 13.
WINDOW_START = int(os.getenv("CH12_WINDOW_START", 127))
WINDOW_END = int(os.getenv("CH12_WINDOW_END", 147))  # inclusive
VALID_DAY_MIN, VALID_DAY_MAX = 127, 154

# --- drift scope -----------------------------------------------------------
N_MONITORED = 20          # top-N features by Chapter 10 SHAP importance
MAX_MISSING_SHARE = 0.30  # a feature missing more than this is not monitorable
N_BINS = 10               # reference bin count for PSI

# --- thresholds (conventional; see Section 12.3) ---------------------------
PSI_WARN = 0.10
PSI_ALERT = 0.20
MIN_FEATURES_WARN = 3     # this many features above PSI_WARN is itself a signal
REVIEW_RATE_REL_DEV = 0.25
ATTRIBUTION_SHIFT_PP = 5.0
PERFORMANCE_REL_DROP = 0.10
CONSECUTIVE_DAYS = 2      # drift must persist before it can trigger a retrain
PERF_WINDOW = 7           # days of matured labels pooled before judging the lagging signal
PSI_EXCESS_ALERT = 0.25   # emergent drift: rise above the window's own structural level
SYNTHETIC_DAY_MIN = 148   # injected days are numbered from here and never baseline the monitor

SECONDS_PER_DAY = 86400


def engine() -> sa.Engine:
    """One place that knows how to reach the decision log."""
    return sa.create_engine(
        f"postgresql+psycopg2://{S.PG_USER}:{S.PG_PASSWORD}"
        f"@{S.PG_HOST}:{S.PG_PORT}/{S.PG_DB}"
    )


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, REPORTS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
