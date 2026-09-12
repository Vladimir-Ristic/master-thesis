"""Chapter 12: drift detection.

PSI is the reported statistic, computed against bin edges frozen in the reference.
Binning the current window on its own quantiles would drive PSI to zero every day
regardless of what the data did, which is the standard way a drift monitor comes to
report nothing at all.

Evidently AI produces the per-day report retained in the repository and an
independent count of drifted columns. The two are deliberately not the same
calculation: the reported number is ours and reproducible from the reference file.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.monitor import config as M
from src.monitor.load import decision_log, monitored_features, reference

EPS = 1e-6


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    r = np.clip(np.asarray(ref, float), EPS, None)
    c = np.clip(np.asarray(cur, float), EPS, None)
    r, c = r / r.sum(), c / c.sum()
    return float(((c - r) * np.log(c / r)).sum())


def _props(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin proportions with an explicit final bucket for missingness."""
    v = np.asarray(values, float)
    miss = float(np.isnan(v).mean()) if v.size else 0.0
    finite = v[np.isfinite(v)]
    counts = np.histogram(finite, bins=np.asarray(edges, float))[0].astype(float)
    p = counts / max(counts.sum(), 1) * (1.0 - miss)
    return np.append(p, miss)


def _ref_props(spec: dict) -> np.ndarray:
    miss = spec["missing_share"]
    return np.append(np.asarray(spec["props"], float) * (1.0 - miss), miss)


def evidently_report(cur: pd.DataFrame, ref: pd.DataFrame, feats: list[str],
                     day: int, write_html: bool = False) -> dict:
    """Evidently's independent read: Wasserstein distance, not PSI.

    Retained because it is a different statistic against the same reference, so
    agreement is evidence rather than arithmetic. The reported number stays ours.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except Exception:
        return {"evidently_drifted": np.nan, "evidently_share": np.nan}

    dd = DataDefinition(numerical_columns=feats)
    snap = Report(metrics=[DataDriftPreset(columns=feats)]).run(
        Dataset.from_pandas(cur[feats], data_definition=dd),
        Dataset.from_pandas(ref[feats], data_definition=dd),
    )
    if write_html:
        snap.save_html(str(M.REPORTS_DIR / f"drift_day_{day}.html"))

    count = share = np.nan
    for m in snap.dict().get("metrics", []):
        if str(m.get("metric_name", "")).startswith("DriftedColumnsCount"):
            v = m.get("value")
            if isinstance(v, dict):
                count, share = v.get("count", np.nan), v.get("share", np.nan)
            elif isinstance(v, (int, float)):
                count = v
            break
    return {"evidently_drifted": count, "evidently_share": share}


def run(day_min: int | None = None, day_max: int | None = None,
        with_evidently: bool = True) -> pd.DataFrame:
    M.ensure_dirs()
    ref = reference()
    feats = monitored_features()
    op = ref["operating_point"]

    log = decision_log(day_min, day_max)
    if log.empty:
        raise SystemExit("decision log is empty for the requested window")
    days = sorted(log.tx_day.unique())
    print(f"days in log: {days[0]}-{days[-1]} ({len(days)} days, {len(log)} rows)")

    html_days = {int(days[0]), int(days[-1])}


    ref_train = pd.read_parquet(M.TRAIN_PARQUET, columns=feats)
    ref_sample = ref_train.sample(n=min(50_000, len(ref_train)), random_state=42)

    rows, per_feature = [], []
    for d in days:
        cur = log[log.tx_day == d]
        psis = {}
        for f in feats:
            psis[f] = psi(_ref_props(ref["features"][f]),
                          _props(cur[f].to_numpy(), ref["features"][f]["edges"]))
            per_feature.append({"tx_day": d, "feature": f, "psi": psis[f]})
        s = pd.Series(psis).sort_values(ascending=False)

        score_psi = psi(np.asarray(ref["score"]["props"], float),
                        np.histogram(cur.probability.to_numpy(),
                                     bins=np.asarray(ref["score"]["edges"], float))[0])
        rr = float(cur.flagged.mean())

        row = {
            "tx_day": int(d),
            "n": int(len(cur)),
            "review_rate": rr,
            "review_rate_rel_dev": (rr - op["review_rate"]) / op["review_rate"],
            "score_psi": score_psi,
            "max_feature_psi": float(s.iloc[0]),
            "top_drift_feature": s.index[0],
            "n_psi_gt_warn": int((s > M.PSI_WARN).sum()),
            "n_psi_gt_alert": int((s > M.PSI_ALERT).sum()),
        }
        if with_evidently:
            row |= evidently_report(cur, ref_sample, feats, int(d),
                                    write_html=int(d) in html_days)
        rows.append(row)
        print(f"  day {d}: n={row['n']} rr={rr:.4f} ({row['review_rate_rel_dev']:+.1%}) "
              f"score_psi={score_psi:.4f} max_psi={row['max_feature_psi']:.4f} "
              f"({row['top_drift_feature']}) warn={row['n_psi_gt_warn']}")

    out = pd.DataFrame(rows)
    out.to_csv(M.TABLES_DIR / "drift_daily.csv", index=False)
    pd.DataFrame(per_feature).to_csv(M.TABLES_DIR / "psi_by_feature.csv", index=False)
    print(f"\nwrote {M.TABLES_DIR / 'drift_daily.csv'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-min", type=int, default=None)
    ap.add_argument("--day-max", type=int, default=None)
    ap.add_argument("--no-evidently", action="store_true")
    a = ap.parse_args()
    run(a.day_min, a.day_max, with_evidently=not a.no_evidently)


if __name__ == "__main__":
    main()