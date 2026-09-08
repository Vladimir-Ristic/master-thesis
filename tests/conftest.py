"""Shared fixtures: build the pipeline once against the synthetic clone."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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
