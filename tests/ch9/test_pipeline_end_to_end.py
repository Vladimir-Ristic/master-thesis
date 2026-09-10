"""End-to-end behaviour, and the safety net that would catch a leak.

A verification step that has never been shown to fire is not a verification
step, so the leakage checks are tested by deliberately injecting the defect
they are supposed to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import arms, config as C, costs, data, metrics, select, tune


def test_a_fold_fit_beats_the_no_skill_line(train, folds):
    p, secs = tune.fit_fold("lgbm", "natural", train, folds[-1], {"n_estimators": 80})
    y = train.y[folds[-1].valid_idx]
    score = metrics.pr_auc(y, p)
    assert score > y.mean() * 1.5, f"PR-AUC {score:.4f} against no-skill {y.mean():.4f}"
    assert score < 0.999, "a near-perfect score on this fixture would mean a leak"


def test_an_injected_target_copy_is_detectable(train, folds):
    """The defect the probe exists to catch, injected on purpose."""
    poisoned = data.Partition(
        name="poisoned",
        X=train.X.assign(leaky_copy_of_target=train.y.astype(np.float32)),
        y=train.y,
        day=train.day,
        amount=train.amount,
        key=train.key,
    )
    p, _ = tune.fit_fold("lgbm", "natural", poisoned, folds[-1], {"n_estimators": 60})
    y = train.y[folds[-1].valid_idx]
    assert metrics.pr_auc(y, p) > 0.95, "an injected label copy went unnoticed"


def test_a_single_feature_cannot_separate_the_classes(train):
    """The Chapter 8 diagnostic, re-run on the matrix Chapter 9 consumes."""
    y = train.y
    worst = 0.0
    for col in train.X.columns:
        v = train.X[col].to_numpy(dtype=np.float64)
        mask = np.isfinite(v)
        if mask.sum() < 1_000 or len(np.unique(y[mask])) < 2:
            continue
        worst = max(worst, metrics.roc_auc(y[mask], v[mask]))
    assert worst < 0.95, f"a single feature reaches ROC-AUC {worst:.3f}"


def test_cross_validation_returns_one_score_per_fold(train, folds):
    result = tune.cross_validate("lgbm", "natural", train, folds, {"n_estimators": 60})
    assert len(result["pr_auc_folds"]) == len(folds)
    assert result["pr_auc_mean"] == pytest.approx(np.mean(result["pr_auc_folds"]))


def test_search_runs_and_returns_a_configuration(train, folds):
    out = tune.run_arm("lgbm", "natural", train, folds, n_trials=2)
    assert out["arm_id"] == "lgbm__natural"
    assert out["best_params"]
    assert out["pr_auc_mean"] > train.prevalence


def test_prior_correction_changes_the_threshold_but_not_the_ranking(train, valid):
    """Why calibration is part of RQ1 and not cosmetic."""
    params = {"n_estimators": 80}
    _, p_raw, p_corr = select.refit_and_score("lgbm", "smote", train, valid, params)

    assert metrics.pr_auc(valid.y, p_corr) == pytest.approx(
        metrics.pr_auc(valid.y, p_raw), abs=1e-9
    )
    assert p_corr.mean() < p_raw.mean()

    t_raw, _ = costs.best_global_threshold(valid.y, p_raw, valid.amount)
    t_corr, _ = costs.best_global_threshold(valid.y, p_corr, valid.amount)
    assert t_corr < t_raw

    # The per-transaction rule compares a probability against ca/amt, so an
    # uncorrected score reviews far too much.
    n_raw = costs.elkan_predictions(p_raw, valid.amount).sum()
    n_corr = costs.elkan_predictions(p_corr, valid.amount).sum()
    assert n_corr < n_raw


def test_every_arm_completes_one_fold(train, folds):
    """Twelve arms, one fold each, smallest budget: the plumbing works."""
    fold = folds[0]
    y = train.y[fold.valid_idx]
    for model, arm in C.ALL_ARMS:
        params = arms.default_params(model)
        p, _ = tune.fit_fold(model, arm, train, fold, params)
        assert p.shape == fold.valid_idx.shape
        assert np.all((p >= 0) & (p <= 1))
        assert metrics.pr_auc(y, p) > y.mean(), f"{model}/{arm} is worse than no skill"


def test_resampling_never_reaches_the_validation_partition(train, valid, monkeypatch):
    """If SMOTE ran on validation data, the row count would change."""
    calls = {"n": 0}
    from imblearn.over_sampling import SMOTE

    original = SMOTE.fit_resample

    def counting(self, X, y):
        calls["n"] += 1
        return original(self, X, y)

    monkeypatch.setattr(SMOTE, "fit_resample", counting)

    pipe = arms.build_pipeline("lgbm", "smote", {"n_estimators": 40}, train.y)
    pipe.fit(train.X.iloc[:8_000], train.y[:8_000])
    assert calls["n"] == 1
    p = pipe.predict_proba(valid.X)[:, 1]
    assert calls["n"] == 1, "the resampler was invoked during prediction"
    assert p.shape[0] == valid.n


def test_class_weighting_is_corrected_like_resampling(train, valid):
    """Weighting inflates probabilities too, so it gets the same treatment."""
    params = {"n_estimators": 80}
    _, p_raw, p_corr = select.refit_and_score("lgbm", "weighted", train, valid, params)
    assert p_corr.mean() < p_raw.mean()
    assert metrics.pr_auc(valid.y, p_corr) == pytest.approx(
        metrics.pr_auc(valid.y, p_raw), abs=1e-9
    )
    assert metrics.brier(valid.y, p_corr) < metrics.brier(valid.y, p_raw)


def test_the_untreated_arm_is_left_alone(train, valid):
    _, p_raw, p_corr = select.refit_and_score(
        "lgbm", "natural", train, valid, {"n_estimators": 60}
    )
    assert np.allclose(p_raw, p_corr)
