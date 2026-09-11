import os
from pathlib import Path

from src.models.config import SEED  # Chapter 9 seed, src/models/config.py

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = "models:/fraud_detector/1"

# --- paths -------------------------------------------------------------
DATA_DIR     = Path("data/processed")
ARTIFACT_DIR = Path("artifacts/explain_v1")
SHAP_DIR     = ARTIFACT_DIR / "shap"

# --- explanation sets --------------------------------------------------
SELECTION_ROWS = 100_000           # training sample for Step 3.12 only
LIMIT          = int(os.environ.get("CH10_LIMIT", "0")) or None   # smoke-test cap

# --- Chapter 9 inheritance (do not recompute) --------------------------
THRESHOLD  = 0.087041              # Table 9.2, example-dependent global, Ca = $4
CA_REVIEW  = 4.0
TOP_K      = (25, 50, 100, 150)    # Step 3.12 compact models