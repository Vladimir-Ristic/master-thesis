"""The twelve experimental arms: three model families x four imbalance treatments.

RQ1 asks what class imbalance handling buys, so the grid is crossed rather than
sequential: every model family is fitted under every treatment, which separates
the effect of the model from the effect of the treatment.

    natural          the class distribution as it occurs
    weighted         cost-sensitive class weighting, no resampling
    smote            SMOTE inside the training fold only, no weighting
    smote_weighted   both

Decision threshold calibration is *not* an arm. It is a decision-layer applied
to all twelve fitted models in `select.py`, because a threshold is chosen after
a model exists and choosing it does not change the model.

Two properties of this module carry the chapter's argument:

1. **Resampling is structural, not procedural.** SMOTE arms are built as an
   imbalanced-learn `Pipeline`, whose resamplers run during `fit` and are
   bypassed during `predict`. The standing rule that resampling never touches
   validation or test data is therefore enforced by the object, not by
   remembering to do it.

2. **SMOTE forces imputation and scaling, and that is a cost of SMOTE.**
   Synthetic minority examples are interpolated between nearest neighbours, so
   the input must be complete and comparably scaled. The SMOTE arms therefore
   give up the native missing-value handling that Chapter 8, Section 8.6 chose
   deliberately. This is not a defect in the implementation; it is a real
   consequence of resampling a feature matrix with 43-58% missingness, and the
   grid is what makes it measurable.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def uses_resampling(arm: str) -> bool:
    return arm.startswith("smote")


def uses_weighting(arm: str) -> bool:
    return arm in ("weighted", "smote_weighted")


def needs_dense_input(model: str, arm: str) -> bool:
    """LogReg cannot accept NaN; SMOTE cannot interpolate over it."""
    return model == "logreg" or uses_resampling(arm)


def positive_weight(y: np.ndarray, arm: str, ratio: float = C.SMOTE_RATIO) -> float:
    """Weight applied to the positive class, on the data the estimator sees.

    Without resampling this is the majority/minority ratio of the training
    fold. After SMOTE at ratio r the minority is already r times the majority,
    so the residual weight needed to reach parity is 1/r — which is why
    `smote_weighted` is a milder treatment than `weighted`, not a stronger one.
    """
    if not uses_weighting(arm):
        return 1.0
    if uses_resampling(arm):
        return 1.0 / ratio
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0:
        raise ValueError("Training fold contains no positive cases.")
    return neg / pos


def resampled_prior(y: np.ndarray, arm: str, ratio: float = C.SMOTE_RATIO) -> float:
    """Positive class prior of the data the estimator was actually fitted on.

    A model fitted after SMOTE at ratio r sees a prior of r / (1 + r), not the
    operating prior.
    """
    if not uses_resampling(arm):
        return float(np.mean(y))
    return float(ratio / (1.0 + ratio))


def training_odds_factor(y: np.ndarray, arm: str, ratio: float = C.SMOTE_RATIO) -> float:
    """Factor by which an arm inflates the predicted positive odds.

    This is the quantity `metrics.odds_correction` divides out, and writing it
    once for all four arms is what makes probabilities comparable across the
    grid. Three of the four arms alter the effective class balance:

        natural          1                       nothing is altered
        weighted         w = n_neg / n_pos       positive odds multiplied by w
        smote            r / (n_pos / n_neg)     the prior is moved to r
        smote_weighted   r / (n_pos / n_neg) * (1 / r) = n_neg / n_pos

    The last line is worth reading twice: SMOTE to ratio r followed by a
    residual weight of 1/r lands on the same fully balanced effective
    distribution as class weighting alone. The two arms differ in *how* they
    get there — synthetic minority examples against reweighted real ones — not
    in the balance they reach, which is precisely what makes their comparison
    informative rather than a difference in degree.
    """
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0 or neg == 0:
        raise ValueError("Training data must contain both classes.")
    true_odds = pos / neg

    factor = 1.0
    if uses_resampling(arm):
        factor *= ratio / true_odds
    if uses_weighting(arm):
        factor *= positive_weight(y, arm, ratio=ratio)
    return float(factor)


# ---------------------------------------------------------------- search spaces


def suggest_params(trial, model: str) -> dict:
    lo, hi = C.N_ESTIMATORS_RANGE
    if model == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", lo, hi, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", lo, hi, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        }
    if model == "logreg":
        return {"C": trial.suggest_categorical("C", C.LOGREG_GRID)}
    raise ValueError(f"Unknown model {model!r}")


def default_params(model: str) -> dict:
    """Untuned reference configuration, used by the smoke profile and tests."""
    lo, _ = C.N_ESTIMATORS_RANGE
    if model == "lgbm":
        return {"n_estimators": lo, "learning_rate": 0.1, "num_leaves": 63}
    if model == "xgb":
        return {"n_estimators": lo, "learning_rate": 0.1, "max_depth": 6}
    if model == "logreg":
        return {"C": C.LOGREG_GRID[0]}
    raise ValueError(f"Unknown model {model!r}")


# ---------------------------------------------------------------- estimators


def _estimator(model: str, params: dict, pos_weight: float, seed: int):
    if model == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            max_bin=C.MAX_BIN,
            n_jobs=C.N_JOBS,
            random_state=seed,
            verbose=-1,
            scale_pos_weight=pos_weight,
            # LightGBM ignores bagging_fraction unless bagging_freq >= 1, so a
            # searched bagging_fraction would silently do nothing without this.
            bagging_freq=1,
            **params,
        )
    if model == "xgb":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            max_bin=C.MAX_BIN,
            n_jobs=C.N_JOBS,
            random_state=seed,
            eval_metric="aucpr",
            scale_pos_weight=pos_weight,
            **params,
        )
    if model == "logreg":
        return LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
            class_weight="balanced" if pos_weight != 1.0 else None,
            **params,
        )
    raise ValueError(f"Unknown model {model!r}")


def build_pipeline(model: str, arm: str, params: dict, y_train: np.ndarray, seed: int = C.SEED):
    """Assemble the fitted-per-fold pipeline for one arm.

    The returned object is an imbalanced-learn Pipeline when the arm resamples
    and a scikit-learn Pipeline otherwise. Either way every fitted parameter —
    imputation medians, scaling statistics, synthetic neighbours, the model
    itself — is estimated from the data passed to `fit` and from nothing else.
    """
    steps = []
    if needs_dense_input(model, arm):
        steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
        steps.append(("scale", StandardScaler()))
    if uses_resampling(arm):
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=C.SMOTE_RATIO,
                    k_neighbors=C.SMOTE_K,
                    random_state=seed,
                ),
            )
        )
    steps.append(("model", _estimator(model, params, positive_weight(y_train, arm), seed)))

    Pipe = ImbPipeline if uses_resampling(arm) else SkPipeline
    return Pipe(steps)


def describe_arm(model: str, arm: str) -> str:
    bits = [model]
    bits.append({"natural": "natural distribution",
                 "weighted": "class weighting",
                 "smote": f"SMOTE r={C.SMOTE_RATIO}",
                 "smote_weighted": f"SMOTE r={C.SMOTE_RATIO} + weighting"}[arm])
    if needs_dense_input(model, arm):
        bits.append("median-imputed, standardised")
    else:
        bits.append("native missing handling")
    return "; ".join(bits)
