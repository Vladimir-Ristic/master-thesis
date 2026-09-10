"""The fitted calibration, and the claim that it beats the assumed one."""

from __future__ import annotations

import numpy as np
import pytest

from src.models import arms, calibration, config as C, metrics


def _shifted_sample(seed=0, n=40_000, prior=0.035, inflate=8.0):
    """A well-calibrated score, then inflated — what a treated arm produces."""
    rng = np.random.default_rng(seed)
    p_true = np.clip(rng.beta(0.7, 0.7 / prior * (1 - prior), n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p_true).astype(np.int8)
    odds = p_true / (1 - p_true) * inflate
    return odds / (1 + odds), y, p_true


def test_the_shift_recovers_a_known_inflation():
    p_inflated, y, p_true = _shifted_sample(inflate=8.0)
    cal = calibration.fit("shift", p_inflated, y)
    assert cal.implied_odds_factor == pytest.approx(8.0, rel=0.25)
    assert metrics.brier(y, cal.transform(p_inflated)) < metrics.brier(y, p_inflated)


def test_the_shift_is_the_analytic_correction_with_b_fitted():
    """Same functional form, so the two are directly comparable."""
    p_inflated, y, _ = _shifted_sample(inflate=5.0)
    fitted = calibration.fit("shift", p_inflated, y)
    assumed = calibration.analytic(5.0)
    assert np.allclose(
        assumed.transform(p_inflated),
        metrics.odds_correction(p_inflated, 5.0),
    )
    assert fitted.b == pytest.approx(assumed.b, abs=0.35)


def test_calibration_never_changes_the_ranking():
    p_inflated, y, _ = _shifted_sample()
    for kind in ("identity", "shift", "platt"):
        cal = calibration.fit(kind, p_inflated, y)
        q = cal.transform(p_inflated)
        assert metrics.pr_auc(y, q) == pytest.approx(
            metrics.pr_auc(y, p_inflated), abs=1e-9
        )
        assert metrics.roc_auc(y, q) == pytest.approx(
            metrics.roc_auc(y, p_inflated), abs=1e-9
        )


def test_choose_prefers_a_fitted_map_over_the_wrong_assumption():
    """The point of the exercise: an assumed factor that is too large loses."""
    p_inflated, y, _ = _shifted_sample(inflate=6.0)
    cal, report = calibration.choose(p_inflated, y, odds_factor=30.0)
    by_kind = {r["kind"]: r for r in report}
    assert cal.kind in ("shift", "platt")
    assert by_kind[cal.kind]["log_loss"] < by_kind["analytic"]["log_loss"]
    assert cal.diagnostics["assumed_odds_factor"] == 30.0


def test_an_untreated_score_is_left_alone_by_choice():
    """With no distortion, the identity should be competitive."""
    rng = np.random.default_rng(3)
    n = 40_000
    p_true = np.clip(rng.beta(0.7, 19.0, n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p_true).astype(np.int8)
    cal, report = calibration.choose(p_true, y)
    by_kind = {r["kind"]: r for r in report}
    assert by_kind["identity"]["log_loss"] < by_kind["shift"]["log_loss"] * 1.02


def test_serialisation_round_trips(tmp_path):
    p_inflated, y, _ = _shifted_sample()
    cal, _ = calibration.choose(p_inflated, y, odds_factor=9.0)
    path = tmp_path / "cal.json"
    cal.save(path)
    back = calibration.Calibrator.load(path)
    assert back.kind == cal.kind
    assert np.allclose(back.transform(p_inflated), cal.transform(p_inflated))
    assert back.fitted_on == "oof"


def test_out_of_fold_blocks_are_disjoint_and_inside_training(train, folds):
    p_oof, y_oof, idx, group = calibration.out_of_fold_predictions(
        "lgbm", "natural", train, folds, {"n_estimators": 40}, use_cache=False
    )
    assert set(np.unique(group)) == {f.index for f in folds}
    assert len(np.unique(idx)) == len(idx), "a row was scored twice"
    assert len(idx) < train.n, "out-of-fold cannot cover the whole partition"
    assert idx.max() < train.n
    assert y_oof.shape == p_oof.shape
    assert metrics.pr_auc(y_oof, p_oof) > y_oof.mean()


def test_out_of_fold_days_exclude_the_first_block(train, folds):
    """Fold 1's own training days are never scored, by construction."""
    _, _, idx, _ = calibration.out_of_fold_predictions(
        "lgbm", "natural", train, folds, {"n_estimators": 30}, use_cache=False
    )
    days = train.day[idx]
    assert days.min() >= folds[0].valid_days[0]
    assert days.max() <= folds[-1].valid_days[1]


def test_the_cache_is_reused_and_invalidated(train, folds):
    a = calibration.out_of_fold_predictions(
        "lgbm", "natural", train, folds, {"n_estimators": 30})
    b = calibration.out_of_fold_predictions(
        "lgbm", "natural", train, folds, {"n_estimators": 30})
    assert np.allclose(a[0], b[0])
    c = calibration.out_of_fold_predictions(
        "lgbm", "natural", train, folds, {"n_estimators": 60})
    assert not np.allclose(a[0], c[0]), "different params returned a cached result"


def test_candidates_are_ranked_on_a_held_out_fold():
    """Two free parameters must not win merely by being flexible."""
    p_inflated, y, _ = _shifted_sample(inflate=6.0, n=30_000)
    group = np.repeat([1, 2, 3], 10_000)
    cal, report = calibration.choose(p_inflated, y, odds_factor=30.0, group=group)
    assert all(r["ranked_on"] == "held-out fold 3" for r in report)
    assert cal.diagnostics["n_rank"] == 10_000
    # The comparable one-parameter factor is recorded whichever form wins.
    assert cal.diagnostics["fitted_shift_odds_factor"] == pytest.approx(6.0, rel=0.3)
    assert cal.diagnostics["assumed_odds_factor"] == 30.0


def test_an_already_calibrated_score_is_not_forced_through_platt():
    rng = np.random.default_rng(9)
    n = 60_000
    p_true = np.clip(rng.beta(0.7, 19.0, n), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < p_true).astype(np.int8)
    group = np.repeat([1, 2, 3], n // 3)
    cal, report = calibration.choose(p_true, y, group=group)
    by_kind = {r["kind"]: r["log_loss"] for r in report}
    assert abs(by_kind["identity"] - min(by_kind.values())) < 2e-3
