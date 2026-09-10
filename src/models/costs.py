"""The RQ1 business cost function, and the decision rules it implies.

Two cost models are carried side by side.

**Example-dependent** (Elkan's cost-sensitive framing, in the per-transaction
form Bahnsen et al. use for card fraud). A false negative costs the full
transaction amount. Any transaction sent for review costs a fixed amount `ca`,
whether or not it turns out to be fraudulent, because the review is performed
either way. A true negative costs nothing:

    cost_i = ca * yhat_i + amt_i * y_i * (1 - yhat_i)

**Fixed ratio**. An amount-independent matrix with constant costs, reported
alongside as a robustness check against the convention in the imbalance
literature:

    cost_i = c_fn * y_i * (1 - yhat_i) + c_fp * (1 - y_i) * yhat_i

The two disagree about *which* transactions to review, not merely how many, and
that disagreement is a result rather than a nuisance.

Three decision rules are derived:

1. `global_threshold` — one threshold applied to every transaction, chosen to
   minimise total cost on the validation partition.
2. `elkan_rule` — the analytically optimal rule under the example-dependent
   matrix. Reviewing transaction i is worth it when the expected loss avoided
   exceeds the review cost, p_i * amt_i > ca, i.e. p_i > ca / amt_i. The
   threshold is therefore per transaction, not global, and it requires
   probabilities that mean what they say — which is why calibration is part of
   RQ1 rather than an afterthought.
3. `fixed_ratio_threshold` — under a constant matrix the optimal threshold is
   c_fp / (c_fp + c_fn), independent of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


# ---------------------------------------------------------------- cost models


def example_dependent_cost(
    y: np.ndarray, yhat: np.ndarray, amount: np.ndarray, ca: float = C.CA_REVIEW
) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    amount = np.asarray(amount, dtype=np.float64)
    return float(np.sum(ca * yhat + amount * y * (1.0 - yhat)))


def fixed_ratio_cost(
    y: np.ndarray,
    yhat: np.ndarray,
    c_fn: float = C.FIXED_COST_FN,
    c_fp: float = C.FIXED_COST_FP,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    fn = np.sum(y * (1.0 - yhat))
    fp = np.sum((1.0 - y) * yhat)
    return float(c_fn * fn + c_fp * fp)


def baseline_costs(
    y: np.ndarray, amount: np.ndarray, ca: float = C.CA_REVIEW
) -> dict[str, float]:
    """Cost of the two degenerate policies, needed to express savings.

    `none` reviews nothing and absorbs every fraud loss; `all` reviews every
    transaction. Chapter 7's degenerate majority-class classifier is `none`.
    """
    y = np.asarray(y, dtype=np.float64)
    amount = np.asarray(amount, dtype=np.float64)
    n = y.size
    return {
        "ed_none": float(np.sum(amount * y)),
        "ed_all": float(ca * n),
        "fx_none": float(C.FIXED_COST_FN * y.sum()),
        "fx_all": float(C.FIXED_COST_FP * (n - y.sum())),
    }


def savings(cost: float, cost_of_doing_nothing: float) -> float:
    """Bahnsen's savings score: fraction of the do-nothing cost avoided."""
    if cost_of_doing_nothing <= 0:
        return float("nan")
    return float((cost_of_doing_nothing - cost) / cost_of_doing_nothing)


# ---------------------------------------------------------------- decision rules


@dataclass
class Decision:
    rule: str
    threshold: float | None  # None for the per-transaction rule
    tp: int
    fp: int
    fn: int
    tn: int
    review_rate: float
    precision: float
    recall: float
    cost_ed: float
    cost_fx: float
    savings_ed: float
    savings_fx: float

    def as_row(self) -> dict:
        return asdict(self)


def _confusion(y: np.ndarray, yhat: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y).astype(bool)
    yhat = np.asarray(yhat).astype(bool)
    tp = int(np.sum(y & yhat))
    fp = int(np.sum(~y & yhat))
    fn = int(np.sum(y & ~yhat))
    tn = int(np.sum(~y & ~yhat))
    return tp, fp, fn, tn


def evaluate_decision(
    rule: str,
    y: np.ndarray,
    yhat: np.ndarray,
    amount: np.ndarray,
    threshold: float | None,
    ca: float = C.CA_REVIEW,
) -> Decision:
    tp, fp, fn, tn = _confusion(y, yhat)
    base = baseline_costs(y, amount, ca=ca)
    cost_ed = example_dependent_cost(y, yhat, amount, ca=ca)
    cost_fx = fixed_ratio_cost(y, yhat)
    n = len(y)
    return Decision(
        rule=rule,
        threshold=threshold,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        review_rate=float((tp + fp) / n) if n else float("nan"),
        precision=float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        recall=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        cost_ed=cost_ed,
        cost_fx=cost_fx,
        savings_ed=savings(cost_ed, base["ed_none"]),
        savings_fx=savings(cost_fx, base["fx_none"]),
    )


def cost_curve(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    ca: float = C.CA_REVIEW,
    n_grid: int = 400,
) -> dict[str, np.ndarray]:
    """Total cost under both models across a grid of global thresholds."""
    p = np.asarray(p, dtype=np.float64)
    lo, hi = float(np.min(p)), float(np.max(p))
    quantile_grid = np.quantile(p, np.linspace(0.0, 1.0, n_grid))
    grid = np.unique(np.concatenate([quantile_grid, np.linspace(lo, hi, n_grid), [0.5]]))
    ed = np.empty(grid.size)
    fx = np.empty(grid.size)
    for i, t in enumerate(grid):
        yhat = (p >= t).astype(np.int8)
        ed[i] = example_dependent_cost(y, yhat, amount, ca=ca)
        fx[i] = fixed_ratio_cost(y, yhat)
    return {"threshold": grid, "cost_ed": ed, "cost_fx": fx}


def best_global_threshold(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    cost_model: str = "ed",
    ca: float = C.CA_REVIEW,
    n_grid: int = 400,
) -> tuple[float, float]:
    """Threshold minimising total cost, and the cost it attains."""
    curve = cost_curve(y, p, amount, ca=ca, n_grid=n_grid)
    key = "cost_ed" if cost_model == "ed" else "cost_fx"
    j = int(np.argmin(curve[key]))
    return float(curve["threshold"][j]), float(curve[key][j])


def fixed_ratio_threshold(
    c_fn: float = C.FIXED_COST_FN, c_fp: float = C.FIXED_COST_FP
) -> float:
    """Analytic optimum under a constant cost matrix (Elkan)."""
    return float(c_fp / (c_fp + c_fn))


def elkan_predictions(
    p: np.ndarray, amount: np.ndarray, ca: float = C.CA_REVIEW
) -> np.ndarray:
    """Per-transaction decision: review iff expected loss avoided exceeds `ca`.

    Transactions whose amount does not exceed the review cost are never worth
    reviewing under this matrix, whatever their probability, which the rule
    handles without a special case.
    """
    p = np.asarray(p, dtype=np.float64)
    amount = np.asarray(amount, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_row_threshold = np.where(amount > 0, ca / amount, np.inf)
    return (p > per_row_threshold).astype(np.int8)


def all_decision_rules(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    ca: float = C.CA_REVIEW,
) -> list[Decision]:
    """The five rules reported in Table 9.3, in increasing sophistication."""
    out: list[Decision] = []

    out.append(
        evaluate_decision("default 0.50", y, (p >= 0.5).astype(np.int8), amount, 0.5, ca)
    )

    t_fx = fixed_ratio_threshold()
    out.append(
        evaluate_decision(
            f"fixed-ratio analytic ({t_fx:.4f})",
            y,
            (p >= t_fx).astype(np.int8),
            amount,
            t_fx,
            ca,
        )
    )

    t_fx_emp, _ = best_global_threshold(y, p, amount, cost_model="fx", ca=ca)
    out.append(
        evaluate_decision(
            "fixed-ratio cost-minimising",
            y,
            (p >= t_fx_emp).astype(np.int8),
            amount,
            t_fx_emp,
            ca,
        )
    )

    t_ed, _ = best_global_threshold(y, p, amount, cost_model="ed", ca=ca)
    out.append(
        evaluate_decision(
            "example-dependent, global",
            y,
            (p >= t_ed).astype(np.int8),
            amount,
            t_ed,
            ca,
        )
    )

    out.append(
        evaluate_decision(
            "example-dependent, per-transaction (Elkan)",
            y,
            elkan_predictions(p, amount, ca=ca),
            amount,
            None,
            ca,
        )
    )
    return out


def ca_sensitivity(
    y: np.ndarray,
    p: np.ndarray,
    amount: np.ndarray,
    ca_values=C.CA_SENSITIVITY,
) -> list[dict]:
    """How the chosen threshold and the savings move with the review cost.

    `ca` is a policy assumption, not a measurement, so the chapter reports the
    curve rather than a single number.
    """
    rows = []
    for ca in ca_values:
        t, cost = best_global_threshold(y, p, amount, cost_model="ed", ca=ca)
        d_global = evaluate_decision(
            "global", y, (p >= t).astype(np.int8), amount, t, ca
        )
        d_elkan = evaluate_decision(
            "elkan", y, elkan_predictions(p, amount, ca=ca), amount, None, ca
        )
        rows.append(
            {
                "ca": ca,
                "global_threshold": t,
                "global_cost": cost,
                "global_savings": d_global.savings_ed,
                "global_review_rate": d_global.review_rate,
                "global_recall": d_global.recall,
                "elkan_cost": d_elkan.cost_ed,
                "elkan_savings": d_elkan.savings_ed,
                "elkan_review_rate": d_elkan.review_rate,
                "elkan_recall": d_elkan.recall,
            }
        )
    return rows
