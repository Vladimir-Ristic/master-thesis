"""Chapter 12: the signals the decision log carries beyond feature drift.

Attribution drift asks whether the model is still explaining its decisions the
same way. A reason code is the top-three positive contributions under the Section
10.6 definition, with anonymised features rendered as their V-group, so the daily
named share is computed over flagged rows only and is baselined on the pooled
window rather than on Chapter 10's whole-sample SHAP mass, which is a different
quantity.

Delayed performance is the lagging signal. In production these labels arrive days
to weeks after the decision; the validation partition supplies them here, which is
what lets Section 12.4 say how much the leading signals were actually standing in for.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.monitor import config as M
from src.monitor.load import decision_log, labels, reference


def as_pct(x) -> float:
    x = float(x)
    return x * 100.0 if x <= 1.0 else x


def block_of(feature: str) -> str:
    return "anonymised" if str(feature).startswith("V-group") else "named"


def attribution(rows: pd.DataFrame) -> dict:
    flagged = rows[rows.flagged]
    mass = {"named": 0.0, "anonymised": 0.0}
    with_named, n = 0, 0
    for codes in flagged.reason_codes:
        if not codes:
            continue
        n += 1
        has_named = False
        for c in codes:
            b = block_of(c["feature"])
            mass[b] += abs(float(c["contribution"]))
            has_named |= b == "named"
        with_named += int(has_named)
    total = mass["named"] + mass["anonymised"]
    return {
        "n_flagged": int(len(flagged)),
        "named_share": 100.0 * mass["named"] / total if total else np.nan,
        "named_coverage": 100.0 * with_named / n if n else np.nan,
    }


def performance(rows: pd.DataFrame, y: pd.Series) -> dict:
    lab = y.reindex(rows.transactionid)
    known = lab.notna().to_numpy()
    if not known.any():
        return {"precision": np.nan, "recall": np.nan, "pr_auc": np.nan,
                "n_fraud": 0, "n_labelled": 0}
    yy = lab.to_numpy()[known].astype(int)
    flagged = rows.flagged.to_numpy()[known]
    prob = rows.probability.to_numpy()[known]
    tp = int((flagged & (yy == 1)).sum())
    return {
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int((yy == 1).sum()), 1),
        "pr_auc": float(average_precision_score(yy, prob)) if yy.sum() else np.nan,
        "n_fraud": int((yy == 1).sum()),
        "n_labelled": int(known.sum()),
    }


def run(day_min: int | None = None, day_max: int | None = None) -> pd.DataFrame:
    M.ensure_dirs()
    ref = reference()
    op = ref["operating_point"]
    ch10 = ref["attribution"]

    log = decision_log(day_min, day_max)
    if log.empty:
        raise SystemExit("decision log is empty")
    y = labels()

    pooled = attribution(log[log.tx_day < M.SYNTHETIC_DAY_MIN])
    print(f"pooled window attribution: named {pooled['named_share']:.1f}% / "
          f"anonymised {100 - pooled['named_share']:.1f}%, "
          f"named coverage {pooled['named_coverage']:.1f}%")
    print(f"  Chapter 10 whole-sample SHAP mass was "
          f"{as_pct(ch10['named_share']):.1f}% named, coverage "
          f"{as_pct(ch10['named_reason_coverage']):.1f}% (different quantity; consistency check only)")

    rows = []
    for d in sorted(log.tx_day.unique()):
        cur = log[log.tx_day == d]
        r = {"tx_day": int(d)} | attribution(cur) | performance(cur, y)
        r["named_share_shift_pp"] = r["named_share"] - pooled["named_share"]
        r["recall_rel_drop"] = (op["recall"] - r["recall"]) / op["recall"]
        r["precision_rel_drop"] = (op["precision"] - r["precision"]) / op["precision"]
        rows.append(r)
        print(f"  day {d}: named={r['named_share']:.1f}% ({r['named_share_shift_pp']:+.1f}pp) "
              f"cov={r['named_coverage']:.1f}% prec={r['precision']:.4f} "
              f"rec={r['recall']:.4f} ({r['recall_rel_drop']:+.1%}) pr_auc={r['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    out.attrs["pooled_named_share"] = pooled["named_share"]
    out.to_csv(M.TABLES_DIR / "signals_daily.csv", index=False)
    print(f"\nwrote {M.TABLES_DIR / 'signals_daily.csv'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-min", type=int, default=None)
    ap.add_argument("--day-max", type=int, default=None)
    a = ap.parse_args()
    run(a.day_min, a.day_max)


if __name__ == "__main__":
    main()