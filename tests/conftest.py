
"""Shared fixtures: build the pipeline once against the synthetic clone."""

from __future__ import annotations

# >>> ch8 test sandbox (managed by fix_ch8_test_sandbox.py)
# Chapter 8's tests run the real feature pipeline. Without this, they overwrite
# data/interim/ and data/processed/ with a synthetic fixture. The environment
# variables are read by src/features/config.py at import time, which is why they
# are set here, before every src.features import, rather than in a fixture.
import os as _sandbox_os
import pathlib as _sandbox_pathlib
import tempfile as _sandbox_tempfile

CH8_TEST_SANDBOX = _sandbox_pathlib.Path(_sandbox_tempfile.mkdtemp(prefix="ch8_test_"))
(CH8_TEST_SANDBOX / "interim").mkdir(parents=True, exist_ok=True)
(CH8_TEST_SANDBOX / "processed").mkdir(parents=True, exist_ok=True)
_sandbox_os.environ["CH8_INTERIM_DIR"] = str(CH8_TEST_SANDBOX / "interim")
_sandbox_os.environ["CH8_PROCESSED_DIR"] = str(CH8_TEST_SANDBOX / "processed")
_sandbox_os.environ["CH8_ARTIFACTS_DIR"] = str(CH8_TEST_SANDBOX / "artifacts")
_sandbox_os.environ["CH8_REPORTS_DIR"]   = str(CH8_TEST_SANDBOX / "reports")
# < ch8 test sandbox


import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import os, pathlib, tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SYNTH_DIR = ROOT / "data" / "raw_synth"


@pytest.fixture(scope="session")
def synthetic_csvs() -> Path:
    if not (SYNTH_DIR / "train_transaction.csv").exists():
        subprocess.run(
            [sys.executable, "scripts/make_synthetic.py"],
            cwd=ROOT, check=True, capture_output=True,
        )
    return SYNTH_DIR


@pytest.fixture(scope="session")
def built_pipeline(synthetic_csvs):
    """Run extraction and the full build once for the whole test session."""
    from src.features import config as cfg
    from src.features import build_features, extract

    cfg.RAW_TX_CSV = synthetic_csvs / "train_transaction.csv"
    cfg.RAW_ID_CSV = synthetic_csvs / "train_identity.csv"

    base, vdf = extract.extract("csv")
    extract.write_interim(base, vdf)
    metadata = build_features.build(verbose=False)
    return metadata


@pytest.fixture(scope="session")
def partitions(built_pipeline):
    from src.features import validate

    return validate.load_partitions()


@pytest.fixture(scope="session")
def raw_frame(synthetic_csvs) -> pd.DataFrame:
    df = pd.read_csv(synthetic_csvs / "train_transaction.csv")
    df.columns = df.columns.str.lower()
    return df
