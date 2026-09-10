from __future__ import annotations

import numpy as np
import pytest

from src.models import metrics


def test_no_skill_pr_auc_equals_prevalence():
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = rng.random(n)  # no information at all
    assert metrics.pr_auc(y, p) == pytest.approx(y.mean(), abs=0.01)


def test_perfect_ranking_scores_one():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.pr_auc(y, p) == pytest.approx(1.0)
    assert metrics.roc_auc(y, p) == pytest.approx(1.0)


def test_prior_correction_maps_the_resampled_prior_onto_the_true_prior():
    pi_res, pi_true = 0.10 / 1.10, 0.035
    out = metrics.prior_correction(np.array([pi_res]), pi_res, pi_true)
    assert out[0] == pytest.approx(pi_true)


def test_prior_correction_is_monotone_and_deflating():
    pi_res, pi_true = 0.5, 0.035
    p = np.array([0.01, 0.1, 0.3, 0.6, 0.95])
    q = metrics.prior_correction(p, pi_res, pi_true)
    assert np.all(np.diff(q) > 0)
    assert np.all(q < p)


def test_prior_correction_is_invertible():
    pi_res, pi_true = 0.10 / 1.10, 0.035
    p = np.array([0.02, 0.2, 0.5, 0.9])
    q = metrics.prior_correction(p, pi_res, pi_true)
    back = metrics.prior_correction(q, pi_true, pi_res)
    assert np.allclose(back, p, atol=1e-10)


def test_prior_correction_leaves_ranking_unchanged():
    """It must not change PR-AUC, only the probability scale."""
    rng = np.random.default_rng(1)
    n = 5_000
    y = (rng.random(n) < 0.05).astype(np.int8)
    p = np.clip(0.1 + 0.5 * y + rng.normal(0, 0.15, n), 1e-6, 1 - 1e-6)
    q = metrics.prior_correction(p, 0.5, 0.05)
    assert metrics.pr_auc(y, q) == pytest.approx(metrics.pr_auc(y, p), abs=1e-9)
    assert metrics.roc_auc(y, q) == pytest.approx(metrics.roc_auc(y, p), abs=1e-9)


def test_prior_correction_recovers_calibration_it_was_derived_for():
    """Construct the situation the correction addresses, then undo it.

    A well-calibrated probability under the operating prior is inflated to the
    resampled prior — which is what fitting on resampled data does — and the
    correction must recover the original and improve the Brier score.
    """
    rng = np.random.default_rng(11)
    n = 20_000
    pi_true, pi_res = 0.035, 0.10 / 1.10

    p_true = np.clip(rng.beta(0.6, 16.0, n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p_true).astype(np.int8)
    p_inflated = metrics.prior_correction(p_true, pi_true, pi_res)  # shift upwards
    p_recovered = metrics.prior_correction(p_inflated, pi_res, pi_true)

    assert np.allclose(p_recovered, p_true, atol=1e-9)
    assert metrics.brier(y, p_recovered) < metrics.brier(y, p_inflated)


def test_prior_correction_rejects_degenerate_priors():
    with pytest.raises(ValueError):
        metrics.prior_correction(np.array([0.5]), 0.0, 0.035)


def test_recall_at_precision_is_bounded_and_ordered():
    rng = np.random.default_rng(2)
    n = 5_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = np.clip(0.02 + 0.5 * y + rng.normal(0, 0.1, n), 1e-6, 1 - 1e-6)
    r50 = metrics.recall_at_precision(y, p, 0.5)
    r90 = metrics.recall_at_precision(y, p, 0.9)
    assert 0.0 <= r90 <= r50 <= 1.0


def test_reliability_curve_uses_quantile_bins():
    rng = np.random.default_rng(3)
    n = 4_000
    y = (rng.random(n) < 0.035).astype(np.int8)
    p = np.clip(rng.beta(1.2, 30, n), 1e-6, 1 - 1e-6)
    rel = metrics.reliability_curve(y, p, n_bins=10)
    assert rel["count"].sum() == n
    assert rel["count"].min() > n / 40, "equal-width bins would collapse here"


def test_threshold_free_bundle_reports_the_no_skill_line():
    rng = np.random.default_rng(4)
    n = 3_000
    y = (rng.random(n) < 0.04).astype(np.int8)
    p = rng.random(n)
    bundle = metrics.threshold_free_bundle(y, p)
    assert bundle["no_skill_pr_auc"] == pytest.approx(y.mean())
    assert set(bundle) >= {"pr_auc", "roc_auc", "brier", "recall_at_p50"}


def test_odds_correction_matches_the_prior_form():
    pi_true, pi_res = 0.035, 0.10 / 1.10
    factor = (pi_res / (1 - pi_res)) / (pi_true / (1 - pi_true))
    p = np.array([0.01, 0.2, 0.7, 0.99])
    assert np.allclose(
        metrics.odds_correction(p, factor), metrics.prior_correction(p, pi_res, pi_true)
    )


def test_odds_correction_with_unit_factor_is_the_identity():
    p = np.array([0.01, 0.5, 0.99])
    assert np.allclose(metrics.odds_correction(p, 1.0), p)


def test_odds_correction_rejects_a_non_positive_factor():
    with pytest.raises(ValueError):
        metrics.odds_correction(np.array([0.5]), 0.0)
