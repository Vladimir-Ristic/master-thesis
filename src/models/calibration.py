"""Probability calibration fitted out of fold.

Chapter 9's first attempt corrected predicted probabilities by an *assumed*
factor: resampling to a ratio r moves the prior, weighting the positive class by
w multiplies its odds by w, so divide the odds by that factor and be done. That
correction is exact for a model linear in the logit — Dal Pozzolo et al. (2015)
derive it in that setting — and logistic regression obeys it almost exactly. A
regularised gradient boosting ensemble does not: the realised shift is smaller
than the nominal one, because shrinkage and early stopping keep the ensemble
from reaching the weighted optimum, so dividing by the nominal factor
over-corrects.

The remedy is to stop assuming and start measuring, without spending the
evaluation data to do it. The inner purged walk-forward folds already produce
predictions on rows their own model never saw: fold 1 scores days 30-58, fold 2
days 59-87, fold 3 days 88-119. Those blocks are disjoint, so their union is a
single out-of-fold sample covering 90 of the 119 training days, entirely inside
the training partition. A calibrator fitted there costs no validation data and
no test data.

Three transforms are provided, in increasing generality, all monotone and
therefore all leaving PR-AUC and ROC-AUC untouched:

    identity   nothing is changed; the baseline
    shift      one parameter: the log-odds are shifted by b. This is the same
               functional form as the analytic correction, with b fitted instead
               of assumed, which is what makes the two directly comparable — the
               analytic factor corresponds to b = -log(factor).
    platt      two parameters: the log-odds are rescaled and shifted,
               logit(p') = a*logit(p) + b. Niculescu-Mizil and Caruana recommend
               this form for boosted models, whose distortion is not a pure
               shift.

**One limitation to state in the chapter.** The out-of-fold sample covers days
30-119, which excludes the low-fraud opening weeks, so its positive rate is
about 4.0% against 3.39% in the validation partition. A calibrator fitted there
inherits that prior and will predict slightly high. That residual is small
beside the error it replaces, and it is a consequence of the drift Chapter 7,
Section 7.6 documented rather than of the method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from . import config as C

EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


# ---------------------------------------------------------------- calibrator


@dataclass
class Calibrator:
    """A monotone map on predicted probabilities, plus how it was obtained."""

    kind: str
    a: float = 1.0
    b: float = 0.0
    fitted_on: str = ""
    n_fit: int = 0
    prevalence_fit: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.kind == "identity":
            return np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
        return _sigmoid(self.a * _logit(p) + self.b)

    @property
    def implied_odds_factor(self) -> float:
        """The factor this calibrator divides the odds by, when a == 1."""
        return float(np.exp(-self.b)) if self.a == 1.0 else float("nan")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "a": self.a,
            "b": self.b,
            "fitted_on": self.fitted_on,
            "n_fit": self.n_fit,
            "prevalence_fit": self.prevalence_fit,
            "implied_odds_factor": self.implied_odds_factor,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibrator":
        return cls(
            kind=payload["kind"],
            a=payload.get("a", 1.0),
            b=payload.get("b", 0.0),
            fitted_on=payload.get("fitted_on", ""),
            n_fit=payload.get("n_fit", 0),
            prevalence_fit=payload.get("prevalence_fit", float("nan")),
            diagnostics=payload.get("diagnostics", {}),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        return cls.from_dict(json.loads(Path(path).read_text()))


def identity() -> Calibrator:
    return Calibrator(kind="identity", fitted_on="none")


def analytic(odds_factor: float) -> Calibrator:
    """The assumed correction, expressed as a calibrator for comparability."""
    return Calibrator(
        kind="analytic",
        a=1.0,
        b=float(-np.log(odds_factor)),
        fitted_on="assumed (not fitted)",
    )


# ---------------------------------------------------------------- fitting


def fit_shift(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """One-parameter fit: shift the log-odds to minimise log loss.

    The objective is convex in b and its derivative, sum(sigmoid(z + b) - y), is
    strictly increasing, so the root is found by bisection rather than by an
    optimiser — exact to machine precision and with no convergence flags to
    check.
    """
    z = _logit(p)
    y = np.asarray(y, dtype=np.float64)

    def gradient(b: float) -> float:
        return float(np.sum(_sigmoid(z + b) - y))

    lo, hi = -30.0, 30.0
    if gradient(lo) > 0 or gradient(hi) < 0:
        # Degenerate: the shift cannot match the observed rate. Fall back to the
        # boundary rather than raising, and record it.
        b = lo if gradient(lo) > 0 else hi
        return Calibrator(kind="shift", a=1.0, b=b,
                          diagnostics={"warning": "gradient did not bracket zero"})
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gradient(mid) > 0:
            hi = mid
        else:
            lo = mid
    return Calibrator(kind="shift", a=1.0, b=0.5 * (lo + hi))


def fit_platt(p: np.ndarray, y: np.ndarray) -> Calibrator:
    """Two-parameter fit on the logit: logit(p') = a*logit(p) + b."""
    z = _logit(p).reshape(-1, 1)
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000)
    model.fit(z, np.asarray(y, dtype=np.int8))
    return Calibrator(kind="platt", a=float(model.coef_[0][0]),
                      b=float(model.intercept_[0]))


def fit(kind: str, p: np.ndarray, y: np.ndarray, fitted_on: str = "oof") -> Calibrator:
    if kind == "identity":
        cal = identity()
    elif kind == "shift":
        cal = fit_shift(p, y)
    elif kind == "platt":
        cal = fit_platt(p, y)
    else:
        raise ValueError(f"Unknown calibrator kind {kind!r}")
    cal.fitted_on = fitted_on
    cal.n_fit = int(len(y))
    cal.prevalence_fit = float(np.mean(y))
    return cal


def score(cal: Calibrator, p: np.ndarray, y: np.ndarray) -> dict:
    q = cal.transform(p)
    return {
        "brier": float(brier_score_loss(y, q)),
        "log_loss": float(log_loss(y, np.clip(q, 1e-15, 1 - 1e-15))),
        "mean_p": float(np.mean(q)),
    }


def choose(
    p_oof: np.ndarray,
    y_oof: np.ndarray,
    odds_factor: float = 1.0,
    group: np.ndarray | None = None,
    kinds=("identity", "shift", "platt"),
) -> tuple[Calibrator, list[dict]]:
    """Pick a calibrator, ranking the candidates out of sample.

    Fitting all three candidates on the whole out-of-fold sample and then
    ranking them there would always favour the most flexible one, because two
    free parameters cannot do worse than one on the data they were fitted to.
    So the out-of-fold sample is split the same way everything else in this
    chapter is split — by time. `group` carries the fold each row came from;
    candidates are fitted on the earlier folds and ranked on the last one, and
    only the winning form is then refitted on the whole out-of-fold sample.

    The analytic correction is scored on the same ranking slice without being
    fitted at all, which is what turns "the assumption is wrong" into a number
    the chapter can quote.
    """
    report = []
    if group is None or len(np.unique(group)) < 2:
        fit_mask = rank_mask = np.ones(len(y_oof), dtype=bool)
        ranking = "in-sample (single fold available)"
    else:
        last = np.max(group)
        fit_mask = group < last
        rank_mask = group == last
        ranking = f"held-out fold {int(last)}"

    fitted = {}
    for kind in kinds:
        candidate = fit(kind, p_oof[fit_mask], y_oof[fit_mask])
        fitted[kind] = candidate
        report.append({"kind": kind, "a": candidate.a, "b": candidate.b,
                       "implied_odds_factor": candidate.implied_odds_factor,
                       "ranked_on": ranking,
                       **score(candidate, p_oof[rank_mask], y_oof[rank_mask])})

    if odds_factor != 1.0:
        assumed = analytic(odds_factor)
        report.append({"kind": "analytic", "a": assumed.a, "b": assumed.b,
                       "implied_odds_factor": odds_factor, "ranked_on": ranking,
                       **score(assumed, p_oof[rank_mask], y_oof[rank_mask])})

    best_kind = min(
        (r for r in report if r["kind"] in kinds), key=lambda r: r["log_loss"]
    )["kind"]

    # Refit the winning form on every out-of-fold row.
    chosen = fit(best_kind, p_oof, y_oof)

    # The one-parameter fit is always recorded, whichever form wins, because it
    # is the only candidate directly comparable with the assumed factor.
    shift = fit("shift", p_oof, y_oof)
    chosen.diagnostics = dict(
        chosen.diagnostics,
        assumed_odds_factor=float(odds_factor),
        fitted_shift_odds_factor=shift.implied_odds_factor,
        ranked_on=ranking,
        n_rank=int(rank_mask.sum()),
        candidates={r["kind"]: {"log_loss": r["log_loss"], "brier": r["brier"]}
                    for r in report},
    )
    return chosen, report


# ---------------------------------------------------------------- out-of-fold


def oof_dir() -> Path:
    d = C.ARTIFACTS / "oof"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_of_fold_predictions(
    model: str,
    arm: str,
    part,
    folds,
    params: dict,
    seed: int = C.SEED,
    use_cache: bool = True,
):
    """Predictions on rows the fitting model never saw, from the inner folds.

    One fit per fold, so three fits per arm. Cached to disk, because the whole
    point of the exercise is that it can be re-read rather than recomputed while
    the chapter is being written. Returns predictions, labels, row indices and
    the fold each row came from — the last of which lets `choose` rank
    candidates on a held-out fold.
    """
    from . import tune  # imported here to keep this module import-light

    arm_id = C.arm_id(model, arm)
    cache = oof_dir() / f"{arm_id}.npz"
    signature = np.array(
        [len(folds), part.n, part.X.shape[1], seed, hash(json.dumps(params, sort_keys=True)) % (2**31)],
        dtype=np.int64,
    )

    if use_cache and cache.exists():
        stored = np.load(cache)
        if (
            stored["signature"].shape == signature.shape
            and np.array_equal(stored["signature"], signature)
            and "group" in stored
        ):
            return stored["p"], stored["y"], stored["idx"], stored["group"]
        print(f"    out-of-fold cache for {arm_id} is stale; recomputing")

    p_parts, idx_parts, group_parts = [], [], []
    for fold in folds:
        p, _ = tune.fit_fold(model, arm, part, fold, params, seed=seed)
        p_parts.append(p)
        idx_parts.append(fold.valid_idx)
        group_parts.append(np.full(fold.valid_idx.size, fold.index, dtype=np.int16))

    idx = np.concatenate(idx_parts)
    p_oof = np.concatenate(p_parts)
    group = np.concatenate(group_parts)
    y_oof = part.y[idx]

    if len(np.unique(idx)) != len(idx):
        raise AssertionError(
            "Out-of-fold blocks overlap. The calibrator would be fitted on rows "
            "counted twice; check the fold generator."
        )

    np.savez_compressed(cache, p=p_oof, y=y_oof, idx=idx, group=group,
                        signature=signature)
    return p_oof, y_oof, idx, group
