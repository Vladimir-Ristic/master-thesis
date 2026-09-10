"""Threshold-free metrics and probability calibration.

PR-AUC is the primary metric throughout, per the project's standing rule and
the argument of Chapter 4, Section 4.5. ROC-AUC is reported alongside for
comparability with the wider literature. Both are threshold-free, which is what
keeps model selection separate from the decision rule chosen in `costs.py`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from . import config as C


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision. The no-skill value equals the positive class rate."""
    return float(average_precision_score(y, p))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def recall_at_precision(y: np.ndarray, p: np.ndarray, min_precision: float) -> float:
    """Highest recall attainable at or above a required precision.

    An alert queue has finite capacity, so this is the operational reading of a
    PR curve: given that an analyst will accept no worse than `min_precision`,
    how much fraud can be caught.
    """
    precision, recall, _ = precision_recall_curve(y, p)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def threshold_free_bundle(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "no_skill_pr_auc": float(np.mean(y)),
        "recall_at_p50": recall_at_precision(y, p, 0.50),
        "recall_at_p90": recall_at_precision(y, p, 0.90),
    }


# ---------------------------------------------------------------- calibration


def odds_correction(p: np.ndarray, factor: float) -> np.ndarray:
    """Divide the predicted odds by `factor`.

    Every treatment in RQ1's grid except the natural one alters the effective
    class balance the model is fitted on, and each does so multiplicatively in
    the odds: resampling to a ratio r moves the prior, and weighting the
    positive class by w multiplies its odds by w. Both therefore inflate the
    predicted probability by a known factor, and both are undone by the same
    operation:

        odds_true = odds_fitted / factor

    Chapter 9 needs this because the example-dependent decision rule compares a
    probability against `ca / amt`. An uncorrected probability is inflated by
    roughly the treatment's factor, so that comparison would send far more
    transactions to review than the cost matrix warrants. Correction leaves the
    ranking — and therefore PR-AUC and ROC-AUC — untouched, since it is a
    strictly increasing map.
    """
    if factor <= 0:
        raise ValueError("The odds factor must be positive.")
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) / factor
    return odds / (1 + odds)


def prior_correction(
    p: np.ndarray, prior_resampled: float, prior_true: float
) -> np.ndarray:
    """Undo a prior shift, in Dal Pozzolo et al.'s (2015) prior form.

        odds_true = odds_resampled * (pi_true / (1 - pi_true))
                                   * ((1 - pi_resampled) / pi_resampled)

    Equivalent to `odds_correction` with the factor written as a ratio of
    priors; kept because it is the form the literature states and the form the
    chapter cites.
    """
    if not (0 < prior_resampled < 1) or not (0 < prior_true < 1):
        raise ValueError("Priors must lie strictly between 0 and 1.")
    factor = (prior_resampled / (1 - prior_resampled)) / (prior_true / (1 - prior_true))
    return odds_correction(p, factor)


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Observed frequency against predicted probability, on quantile bins.

    Equal-width bins are useless at a 3.5% positive rate because almost every
    prediction falls in the first bin, so bins are quantiles of the predicted
    score.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        mean_pred.append(p[m].mean())
        frac_pos.append(y[m].mean())
        counts.append(int(m.sum()))
    return {
        "mean_predicted": np.array(mean_pred),
        "fraction_positive": np.array(frac_pos),
        "count": np.array(counts),
    }
