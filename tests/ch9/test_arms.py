"""The twelve arms, and the two properties the chapter claims for them:
resampling is confined to the fit path, and only the resampling arms need a
complete matrix."""

from __future__ import annotations

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models import arms, config as C


def test_positive_weight_by_arm():
    y = np.array([0] * 96 + [1] * 4)  # 4% positive
    assert arms.positive_weight(y, "natural") == 1.0
    assert arms.positive_weight(y, "weighted") == pytest.approx(24.0)
    assert arms.positive_weight(y, "smote") == 1.0
    assert arms.positive_weight(y, "smote_weighted") == pytest.approx(1 / C.SMOTE_RATIO)


def test_smote_weighted_is_milder_than_weighted():
    """Both arms weight, but after resampling the residual imbalance is smaller."""
    y = np.array([0] * 965 + [1] * 35)
    assert arms.positive_weight(y, "smote_weighted") < arms.positive_weight(y, "weighted")


def test_resampled_prior_reflects_the_data_the_model_saw():
    y = np.array([0] * 965 + [1] * 35)
    assert arms.resampled_prior(y, "natural") == pytest.approx(0.035)
    assert arms.resampled_prior(y, "smote") == pytest.approx(
        C.SMOTE_RATIO / (1 + C.SMOTE_RATIO)
    )


def test_only_logreg_and_resampling_arms_need_dense_input():
    assert arms.needs_dense_input("lgbm", "natural") is False
    assert arms.needs_dense_input("xgb", "weighted") is False
    assert arms.needs_dense_input("lgbm", "smote") is True
    assert arms.needs_dense_input("logreg", "natural") is True


def test_resampling_arms_are_imblearn_pipelines_with_a_smote_step(train):
    pipe = arms.build_pipeline("lgbm", "smote", {"n_estimators": 10}, train.y)
    assert isinstance(pipe, ImbPipeline)
    names = [n for n, _ in pipe.steps]
    assert names == ["impute", "scale", "smote", "model"]


def test_non_resampling_tree_arms_are_bare_estimators(train):
    pipe = arms.build_pipeline("lgbm", "weighted", {"n_estimators": 10}, train.y)
    assert [n for n, _ in pipe.steps] == ["model"]
    assert not isinstance(pipe, ImbPipeline)


def test_tree_arms_accept_missing_values_natively(train, valid):
    idx = np.arange(4_000)
    X, y = train.X.iloc[idx], train.y[idx]
    assert X.isna().to_numpy().any(), "fixture should contain missing values"
    pipe = arms.build_pipeline("lgbm", "natural", {"n_estimators": 20}, y)
    pipe.fit(X, y)
    p = pipe.predict_proba(valid.X.iloc[:500])[:, 1]
    assert p.shape == (500,) and np.all((p >= 0) & (p <= 1))


def test_resampling_does_not_change_the_number_of_predicted_rows(train, valid):
    idx = np.arange(6_000)
    X, y = train.X.iloc[idx], train.y[idx]
    pipe = arms.build_pipeline("lgbm", "smote", {"n_estimators": 20}, y)
    pipe.fit(X, y)
    p = pipe.predict_proba(valid.X.iloc[:1_000])[:, 1]
    assert p.shape == (1_000,), "the resampler ran on the prediction path"


def test_resampling_actually_rebalances_the_training_fold(train):
    idx = np.arange(8_000)
    X, y = train.X.iloc[idx], train.y[idx]
    pipe = arms.build_pipeline("lgbm", "smote", {"n_estimators": 5}, y)
    Xt, yt = X, y
    for name, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            Xt, yt = step.fit_resample(Xt, yt)
        else:
            Xt = step.fit_transform(Xt, yt)
    ratio = yt.sum() / (len(yt) - yt.sum())
    assert ratio == pytest.approx(C.SMOTE_RATIO, rel=0.02)
    assert len(yt) > len(y)


def test_search_space_is_within_the_declared_budget():
    class FakeTrial:
        def __init__(self):
            self.seen = {}

        def suggest_int(self, name, lo, hi, step=1, log=False):
            self.seen[name] = (lo, hi)
            return lo

        def suggest_float(self, name, lo, hi, log=False):
            self.seen[name] = (lo, hi)
            return lo

        def suggest_categorical(self, name, choices):
            self.seen[name] = choices
            return choices[0]

    for model in ("lgbm", "xgb"):
        t = FakeTrial()
        params = arms.suggest_params(t, model)
        assert t.seen["n_estimators"] == C.N_ESTIMATORS_RANGE
        assert "learning_rate" in params
    t = FakeTrial()
    assert arms.suggest_params(t, "logreg")["C"] == C.LOGREG_GRID[0]


def test_every_arm_builds(train):
    for model, arm in C.ALL_ARMS:
        pipe = arms.build_pipeline(model, arm, arms.default_params(model), train.y)
        assert pipe.steps[-1][0] == "model"
        assert arms.describe_arm(model, arm)


def test_training_odds_factor_by_arm():
    y = np.array([0] * 965 + [1] * 35)
    true_odds = 35 / 965
    f = {a: arms.training_odds_factor(y, a) for a in C.IMBALANCE_ARMS}

    assert f["natural"] == pytest.approx(1.0)
    assert f["weighted"] == pytest.approx(965 / 35)
    assert f["smote"] == pytest.approx(C.SMOTE_RATIO / true_odds)
    # SMOTE to r then weighting by 1/r lands on the same balance as weighting.
    assert f["smote_weighted"] == pytest.approx(f["weighted"])
    for arm in ("weighted", "smote", "smote_weighted"):
        assert f[arm] > 1.0


def test_odds_factor_requires_both_classes():
    with pytest.raises(ValueError):
        arms.training_odds_factor(np.zeros(10, dtype=np.int8), "weighted")


def test_lightgbm_bagging_is_actually_enabled(train):
    """bagging_fraction is silently ignored unless bagging_freq >= 1."""
    pipe = arms.build_pipeline(
        "lgbm", "natural", {"n_estimators": 10, "bagging_fraction": 0.6}, train.y
    )
    params = pipe.steps[-1][1].get_params()
    assert params["bagging_freq"] >= 1
    assert params["bagging_fraction"] == 0.6
