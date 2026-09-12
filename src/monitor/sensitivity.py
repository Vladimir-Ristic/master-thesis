"""Chapter 12: how the operating point depends on prevalence and review cost.

The deployed review rate is 18.8 per cent, which no production fraud operation
would staff. That figure is not a property of the model: it follows from a dataset
whose fraud prevalence is an order of magnitude above a real card stream, and from
a review cost of four units. This module quantifies both, offline, on probabilities
already stored, changing nothing in the served path.

The prevalence correction is applied to the odds, which is where a prior shift
belongs - the same mechanism assumed_odds_factor exists for in the calibrator,
and which is set to 1.0 in the deployed path because the selected arm needed no
correction. Rates are then recomposed class-conditionally, so a lower prevalence
changes both the scores the model emits and the mix of transactions it sees,
rather than only one of the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.monitor import config as M
from src.serving.artifacts import load_artifacts
from src.serving.score import score_frame

PREVALENCES = [None, 0.01, 0.003, 0.001]   # None keeps the dataset's own
CA_VALUES = [4.0, 10.0, 25.0]              # the deployed value, then two harder ones


def odds(p):
    return p / (1.0 - p)


def correct(p: np.ndarray, factor: float) -> np.ndarray:
    """Prior correction on the odds: the calibrator's own mechanism, applied offline."""
    if factor == 1.0:
        return p
    o = odds(p) * factor
    return o / (1.0 + o)


def amounts_for(ids: pd.Series) -> np.ndarray:
    with M.engine().connect() as c:
        hist = pd.read_sql(sa.text("select transactionid, transactionamt from tx_history"), c)
    return hist.set_index("transactionid").transactionamt.reindex(ids).to_numpy(dtype="float64")

def cost_minimising_threshold(pc: np.ndarray, y: np.ndarray, amt: np.ndarray,
                              ca: float, w1: float, w0: float) -> float:
    """Chapter 9's rule: one global threshold minimising example-dependent cost.

    Under the Bahnsen cost matrix a reviewed transaction costs ca whether or not it
    is fraudulent, and a missed fraud costs its amount. Class weights recompose the
    population at the assumed prevalence without resampling anything.
    """
    order = np.argsort(-pc)
    ps, ys, ams = pc[order], y[order], np.nan_to_num(amt[order])
    w = np.where(ys == 1, w1, w0)
    fraud_cost = np.where(ys == 1, w1 * ams, 0.0)
    total = fraud_cost.sum()
    cost = ca * np.cumsum(w) + (total - np.cumsum(fraud_cost))
    cost = np.concatenate([[total], cost])
    k = int(np.argmin(cost))
    return np.inf if k == 0 else float(ps[k - 1])


def run() -> pd.DataFrame:
    M.ensure_dirs()
    art = load_artifacts()
    contract = list(art.feature_names)

    valid = pd.read_parquet(M.VALID_PARQUET)
    scored = score_frame(valid[contract], art)
    p = np.array([s.probability for s in scored])
    y = valid["isfraud"].to_numpy().astype(int)
    amt = amounts_for(valid["transactionid"])
    print(f"amounts: {np.isfinite(amt).sum()} of {len(amt)} resolved, "
          f"mean {np.nanmean(amt):.2f}, median {np.nanmedian(amt):.2f}")

    pi_cal = float((y == 1).mean())
    flag = p > art.threshold
    tp = int((flag & (y == 1)).sum())
    print(f"deployed global threshold: {int(flag.sum())} of {len(p)} flagged, "
          f"review {flag.mean():.6f}, precision {tp / max(int(flag.sum()), 1):.6f}, "
          f"recall {tp / max(int((y == 1).sum()), 1):.6f}")
    print(f"dataset prevalence: {pi_cal:.4f}\n")

    rows = []
    for pi in PREVALENCES:
        target = pi_cal if pi is None else pi
        factor = odds(target) / odds(pi_cal)
        pc = correct(p, factor)
        w1 = target / pi_cal
        w0 = (1.0 - target) / (1.0 - pi_cal)
        for ca in CA_VALUES:
            t = cost_minimising_threshold(pc, y, amt, ca, w1, w0)
            f = pc >= t
            tpr = float(f[y == 1].mean())
            fpr = float(f[y == 0].mean())
            fr = y == 1
            amt_f = np.nan_to_num(amt[fr])
            recall_value = float(amt_f[f[fr]].sum() / max(amt_f.sum(), 1e-9))
            rr = target * tpr + (1.0 - target) * fpr
            rows.append({
                "prevalence": round(target, 5),
                "ca_review": ca,
                "odds_factor": round(factor, 5),
                "threshold": t,
                "recall": round(tpr, 4),
                "recall_value": round(recall_value, 4),
                "fpr": round(fpr, 5),
                "review_rate": round(rr, 5),
                "precision": round(target * tpr / rr, 4) if rr > 0 else np.nan,
                "alerts_per_10k": round(10000 * rr, 1),
            })

    out = pd.DataFrame(rows)
    out.to_csv(M.TABLES_DIR / "table_12_5_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nwrote {M.TABLES_DIR / 'table_12_5_sensitivity.csv'}")
    return out


if __name__ == "__main__":
    run()