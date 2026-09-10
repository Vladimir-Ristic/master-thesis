"""Hyperparameter search for one arm, under purged walk-forward CV.

The search objective is PR-AUC averaged over the folds, and nothing else. Cost
is deliberately not the search objective: cost depends on a decision threshold,
so optimising it here would fold the threshold decision into the hyperparameter
decision and make the two impossible to report separately. RQ1 asks about both,
so they are kept apart — the search ranks models by a threshold-free metric,
and `select.py` then chooses the threshold on the validation partition.

Trials are pruned across folds: a configuration whose first-fold PR-AUC is
already below the running median is abandoned rather than fitted twice more.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import optuna

from . import arms, config as C, metrics, tracking
from .cv import Fold
from .data import Partition

optuna.logging.set_verbosity(optuna.logging.WARNING)


def fit_fold(
    model: str,
    arm: str,
    part: Partition,
    fold: Fold,
    params: dict,
    seed: int = C.SEED,
) -> tuple[np.ndarray, float]:
    """Fit one arm on one fold's training block, score its validation block."""
    X_tr = part.X.iloc[fold.train_idx]
    y_tr = part.y[fold.train_idx]
    X_va = part.X.iloc[fold.valid_idx]

    pipe = arms.build_pipeline(model, arm, params, y_tr, seed=seed)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
    return p, time.time() - t0


def cross_validate(
    model: str,
    arm: str,
    part: Partition,
    folds: list[Fold],
    params: dict,
    trial: optuna.Trial | None = None,
    seed: int = C.SEED,
) -> dict:
    """Run every fold, reporting intermediate values so trials can be pruned."""
    per_fold, fit_seconds = [], 0.0
    for step, fold in enumerate(folds):
        p, secs = fit_fold(model, arm, part, fold, params, seed=seed)
        fit_seconds += secs
        y_va = part.y[fold.valid_idx]
        score = metrics.pr_auc(y_va, p)
        per_fold.append(score)

        if trial is not None:
            trial.report(float(np.mean(per_fold)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        "pr_auc_mean": float(np.mean(per_fold)),
        "pr_auc_std": float(np.std(per_fold, ddof=0)),
        "pr_auc_folds": [float(s) for s in per_fold],
        "fit_seconds": round(fit_seconds, 1),
    }


def _sampler(model: str, seed: int):
    """TPE for the boosting families; an exhaustive grid for the baseline.

    The logistic regression baseline has one hyperparameter over a handful of
    orders of magnitude, so a search algorithm would be theatre. Bergstra and
    Bengio's argument for random or model-based search applies to the
    high-dimensional boosting spaces, not to a one-dimensional grid.
    """
    if model == "logreg":
        return optuna.samplers.GridSampler({"C": list(C.LOGREG_GRID)}, seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True)


def n_trials_for(model: str) -> int:
    return len(C.LOGREG_GRID) if model == "logreg" else C.N_TRIALS


def run_arm(
    model: str,
    arm: str,
    part: Partition,
    folds: list[Fold],
    n_trials: int | None = None,
    seed: int = C.SEED,
) -> dict:
    """Search one arm and return its best configuration and CV result."""
    n_trials = n_trials_for(model) if n_trials is None else n_trials
    arm_name = C.arm_id(model, arm)

    study = optuna.create_study(
        direction="maximize",
        sampler=_sampler(model, seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1),
        study_name=arm_name,
    )

    with tracking.run(f"search::{arm_name}", tags={"arm": arm, "model": model,
                                                   "stage": "search"}):
        tracking.log_params(
            {
                "model": model,
                "imbalance_arm": arm,
                "n_folds": len(folds),
                "purge_days": C.PURGE_DAYS,
                "n_trials": n_trials,
                "profile": C.PROFILE,
                "smote_ratio": C.SMOTE_RATIO if arms.uses_resampling(arm) else None,
                "dense_input": arms.needs_dense_input(model, arm),
            }
        )

        def objective(trial: optuna.Trial) -> float:
            params = arms.suggest_params(trial, model)
            result = cross_validate(model, arm, part, folds, params, trial=trial, seed=seed)
            trial.set_user_attr("result", result)
            with tracking.run(f"trial::{arm_name}::{trial.number}", nested=True):
                tracking.log_params({"model": model, "imbalance_arm": arm, **params})
                tracking.log_metrics(
                    {
                        "cv_pr_auc": result["pr_auc_mean"],
                        "cv_pr_auc_std": result["pr_auc_std"],
                        "fit_seconds": result["fit_seconds"],
                        **{f"cv_pr_auc_fold{i+1}": v
                           for i, v in enumerate(result["pr_auc_folds"])},
                    }
                )
            return result["pr_auc_mean"]

        t0 = time.time()
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        wall = time.time() - t0

        best = study.best_trial
        out = {
            "arm_id": arm_name,
            "model": model,
            "imbalance_arm": arm,
            "description": arms.describe_arm(model, arm),
            "best_params": dict(best.params),
            "n_trials_run": len(study.trials),
            "n_pruned": sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ),
            "search_seconds": round(wall, 1),
            **best.user_attrs["result"],
        }
        tracking.log_metrics(
            {
                "best_cv_pr_auc": out["pr_auc_mean"],
                "best_cv_pr_auc_std": out["pr_auc_std"],
                "search_seconds": out["search_seconds"],
                "n_pruned": out["n_pruned"],
            }
        )
        tracking.log_dict(out, f"search/{arm_name}.json")

    return out
