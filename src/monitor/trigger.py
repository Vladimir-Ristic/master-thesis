"""Chapter 12: from signals to a decision.

Two distinctions do the work here.

Structural against emergent drift. Feature drift is present from the first served
day - the account-frequency encoding was fitted on the training period and the
served population has turned over since - and it does not move the score. A rule
that fires on the level would fire every day and be switched off within a week.
The trigger therefore watches the rise of a feature's PSI above the level the
window itself established, while the frozen reference still supplies the reported
number.

Leading against lagging. Drift is observable immediately; performance is not. The
lagging signal is pooled over a rolling window because a single day carries only
about fifty matured fraud labels, which is roughly five percent relative noise on
recall - enough to fire a naive daily rule on a healthy system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitor import config as M
from src.monitor.load import reference


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    drift = pd.read_csv(M.TABLES_DIR / "drift_daily.csv")
    signals = pd.read_csv(M.TABLES_DIR / "signals_daily.csv")
    psi_long = pd.read_csv(M.TABLES_DIR / "psi_by_feature.csv")
    d = drift.merge(signals, on="tx_day", how="left").sort_values("tx_day")
    return d.reset_index(drop=True), psi_long


def emergent(psi_long: pd.DataFrame) -> pd.DataFrame:
    """PSI above the structural level the baseline days established."""
    base = psi_long[psi_long.tx_day < M.SYNTHETIC_DAY_MIN]
    level = base.groupby("feature").psi.median()
    psi_long = psi_long.assign(excess=psi_long.psi - psi_long.feature.map(level))
    idx = psi_long.groupby("tx_day").excess.idxmax()
    top = psi_long.loc[idx].set_index("tx_day")
    return pd.DataFrame({"psi_excess": top.excess, "psi_excess_feature": top.feature})


def evaluate(d: pd.DataFrame, psi_long: pd.DataFrame) -> pd.DataFrame:
    op = reference()["operating_point"]
    d = d.join(emergent(psi_long), on="tx_day")

    tp = d.recall * d.n_fraud
    roll_recall = (tp.rolling(M.PERF_WINDOW, min_periods=3).sum()
                   / d.n_fraud.rolling(M.PERF_WINDOW, min_periods=3).sum())
    d["roll_recall"] = roll_recall
    d["roll_recall_rel_drop"] = (op["recall"] - roll_recall) / op["recall"]

    d["sig_feature"] = d.psi_excess > M.PSI_EXCESS_ALERT
    d["sig_score"] = d.score_psi > M.PSI_ALERT
    d["sig_review"] = d.review_rate_rel_dev.abs() > M.REVIEW_RATE_REL_DEV
    d["sig_attribution"] = d.named_share_shift_pp.abs() > M.ATTRIBUTION_SHIFT_PP
    d["sig_performance"] = d.roll_recall_rel_drop > M.PERFORMANCE_REL_DROP

    sig_cols = ["sig_feature", "sig_score", "sig_review", "sig_attribution", "sig_performance"]
    d["n_signals"] = d[sig_cols].sum(axis=1)

    both = (d.sig_feature & d.sig_review).to_numpy()
    persistent = np.zeros(len(d), bool)
    for i in range(len(d)):
        if i + 1 >= M.CONSECUTIVE_DAYS:
            persistent[i] = both[i - M.CONSECUTIVE_DAYS + 1: i + 1].all()

    d["decision"] = np.where(
        d.sig_performance.to_numpy() | persistent, "retrain",
        np.where(d.n_signals.to_numpy() > 0, "investigate", "no_action"),
    )
    d["reason"] = [
        ", ".join(c.replace("sig_", "") for c in sig_cols if row[c]) or "-"
        for _, row in d.iterrows()
    ]
    return d


def run() -> pd.DataFrame:
    M.ensure_dirs()
    d, psi_long = load_frames()
    out = evaluate(d, psi_long)

    cols = ["tx_day", "n", "review_rate", "score_psi", "max_feature_psi",
            "psi_excess", "psi_excess_feature", "named_share_shift_pp",
            "roll_recall", "roll_recall_rel_drop", "n_signals", "decision", "reason"]
    out[cols].to_csv(M.TABLES_DIR / "decisions_daily.csv", index=False)

    show = out[cols].copy()
    for c in ["review_rate", "score_psi", "max_feature_psi", "psi_excess",
              "named_share_shift_pp", "roll_recall", "roll_recall_rel_drop"]:
        show[c] = show[c].round(4)
    pd.set_option("display.width", 220)
    print(show.to_string(index=False))
    print("\ndecisions:", out.decision.value_counts().to_dict())
    print(f"wrote {M.TABLES_DIR / 'decisions_daily.csv'}")
    return out


if __name__ == "__main__":
    run()