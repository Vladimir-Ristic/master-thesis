"""Shared fixtures. Everything runs against the synthetic clone, so the suite
needs neither PostgreSQL, nor MLflow, nor the real dataset."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("CH9_MLFLOW", "0")
os.environ.setdefault("CH9_PROFILE", "smoke")

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "processed_synth"


@pytest.fixture(scope="session", autouse=True)
def synthetic_partitions():
    if not (SYNTH / "train_v1.parquet").exists():
        subprocess.run(
            [sys.executable, "-m", "scripts.make_synthetic_features", "--out", str(SYNTH)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    from src.models import config as C

    C.DATA_PROCESSED = SYNTH
    return SYNTH


@pytest.fixture(scope="session")
def train(synthetic_partitions):
    from src.models import data

    return data.load_partition("train")


@pytest.fixture(scope="session")
def valid(synthetic_partitions):
    from src.models import data

    return data.load_partition("valid")


@pytest.fixture(scope="session")
def folds(train):
    from src.models import cv

    return cv.make_folds(train.day)
