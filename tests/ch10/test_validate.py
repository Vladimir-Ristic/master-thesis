"""Chapter 10 — tests that the protocol checks and the explainer actually fire."""
import numpy as np
import pandas as pd
import pytest
import shap

from src.explain import config as C
from src.explain.load import load_estimator


def test_validate_fails_loudly_without_shap_arrays(tmp_path, monkeypatch):
    """Run before make ch10-shap, validation must abort, not report PASS."""
    monkeypatch.setattr(C, "SHAP_DIR", tmp_path)
    from src.explain import validate
    with pytest.raises((FileNotFoundError, SystemExit)):
        validate.main()


def test_injected_target_moves_the_attribution():
    """Perturbing a row's top contributor must reduce its attribution."""
    est = load_estimator()
    names = est.booster_.feature_name()
    v = pd.read_parquet(C.DATA_DIR / "valid_v1.parquet")[names]
    tr = pd.read_parquet(C.DATA_DIR / "train_v1.parquet")[names]

    ex = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
    row = v.iloc[[int(np.argmax(est.predict(v, raw_score=True)))]].copy()

    sv0 = np.asarray(ex.shap_values(row)).reshape(-1)
    j = int(np.argmax(sv0))
    assert sv0[j] > 0

    row.iloc[0, j] = tr.iloc[:, j].median()
    sv1 = np.asarray(ex.shap_values(row)).reshape(-1)

    assert abs(sv1[j]) < abs(sv0[j]), (
        f"{names[j]}: attribution did not fall when set to the training median "
        f"({sv0[j]:.4f} -> {sv1[j]:.4f})")
    base = ex.expected_value
    base = float(base[1]) if np.ndim(base) else float(base)
    margin = est.predict(row, raw_score=True)[0]
    assert abs(sv1.sum() + base - margin) < 1e-4, "additivity broke on the perturbed row"