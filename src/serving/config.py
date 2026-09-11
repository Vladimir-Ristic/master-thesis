"""Chapter 11 serving configuration.

Constants only: nothing here reads a model, a file or a database. Every path is
overridable by environment variable with the repository layout as the default, so
the identical code runs in the training environment and in the serving container -
the mechanism Chapter 9 section 3.4a established for the feature pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(
    os.environ.get("CH11_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)
load_dotenv(PROJECT_ROOT / ".env")

# --- artifact locations --------------------------------------------------
ARTIFACTS_ROOT = Path(os.environ.get("CH11_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
REPORTS_ROOT = Path(os.environ.get("CH11_REPORTS_DIR", PROJECT_ROOT / "reports"))

MODELS_DIR = ARTIFACTS_ROOT / "models_v1"
EXPLAIN_DIR = ARTIFACTS_ROOT / "explain_v1"
FEATURES_DIR = ARTIFACTS_ROOT / "features_v1"

CONTRACT_PATH = MODELS_DIR / "feature_names_v1.json"
SELECTION_SUMMARY_PATH = MODELS_DIR / "selection_summary.json"
DECISION_RULES_PATH = REPORTS_ROOT / "tables" / "table_9_4_decision_rules.csv"
V_GROUPS_PATH = EXPLAIN_DIR / "v_groups.json"
FEATURE_MAP_PATH = EXPLAIN_DIR / "feature_map.json"
ENCODER_PATH = FEATURES_DIR / "feature_encoder.joblib"

# --- model registry ------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("CH11_MODEL_NAME", "fraud_detector")
MODEL_VERSION = os.environ.get("CH11_MODEL_VERSION", "1")
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# Fork 4: a baked copy of the same version, used only when the registry is
# unreachable at startup. A protocol check asserts it matches the registry.
MODEL_FALLBACK_DIR = Path(
    os.environ.get("CH11_MODEL_FALLBACK_DIR", MODELS_DIR / "registered_v1")
)

# --- what the loaded artifact must be ------------------------------------
EXPECTED_N_FEATURES = 290
EXPECTED_N_TREES = 750
EXPECTED_NUM_LEAVES = 250

# --- decision policy (Chapter 9) -----------------------------------------
DECISION_RULE = "example-dependent, global"
CA_REVIEW = 4.0

# --- explanation policy (Chapter 10, Fork 2) -----------------------------
EXPLAIN_FLAGGED_ONLY = os.environ.get("CH11_EXPLAIN_FLAGGED_ONLY", "1") == "1"
REASON_CODE_TOP_K = 3

# --- database ------------------------------------------------------------
PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
PG_DB = os.environ.get("POSTGRES_DB", "")
PG_USER = os.environ.get("POSTGRES_USER", "")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")