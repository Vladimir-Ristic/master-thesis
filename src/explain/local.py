"""Chapter 10, Step 3.11 — review-queue reason codes, coverage, worked cases, latency."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from src.explain import config as C
from src.explain.aggregate import load_shap
from src.explain.load import load_estimator

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*LightGBM binary classifier with TreeExplainer.*")

TABLES = Path("reports/tables/ch10")
ANON   = "Anonymised V (retained)"
N_LATENCY = 1000


def reason_label(feat, family, vgroup):
    """Anonymised features are rendered as their group, never as a V column."""
    return f"{vgroup[feat]} (anonymised group)" if family[feat] == ANON else feat


def main():
    sv, base, names = load_shap("valid")
    fmap = json.loads((C.ARTIFACT_DIR / "feature_map.json").read_text())
    family, vgroup = fmap["feature_family"], fmap["v_group"]
    cal = json.load(open("artifacts/models_v1/selection_summary.json"))["calibrator"]

    est = load_estimator()
    v = pd.read_parquet(C.DATA_DIR / "valid_v1.parquet")
    X = v[names]
    assert len(X) == sv.shape[0], "SHAP array and parquet are not row-aligned"

    margin = est.predict(X, raw_score=True)
    p = expit(cal["a"] * margin + cal["b"])
    y = v.isfraud.values.astype(bool)
    flagged = p >= C.THRESHOLD

    tp = int((flagged & y).sum())
    print(f"=== review queue, threshold {C.THRESHOLD} ===")
    print(f"flagged transactions        {flagged.sum():>7,}  ({flagged.mean():.6f} of validation)")
    print(f"  of which true positives   {tp:>7,}  (precision {tp/flagged.sum():.3f})")
    print(f"  Chapter 9 targets:         14,629  (0.187980, precision 0.156)")

    # --- reason codes on the flagged set --------------------------------
    svf = sv[flagged]
    top3 = np.argsort(-svf, axis=1)[:, :3]
    pos  = np.take_along_axis(svf, top3, axis=1) > 0
    is_named = np.array([family[n] != ANON for n in names])
    covered = (is_named[top3] & pos).any(1)
    cov = float(covered.mean())
    print(f"named-reason coverage       {cov:>7.1%}  (>=1 named/derived in top-3 positive)")
    print(f"anonymised-only cases       {int((~covered).sum()):>7,}  ({1-cov:.1%})")

    # --- three worked cases, selected by rule ---------------------------
    idx = np.arange(len(v))
    cases = {
        "highest-p true positive (flagged)":
            idx[flagged & y][np.argmax(p[flagged & y])],
        "highest-amount false positive (flagged)":
            idx[flagged & ~y][np.argmax(v.transactionamt.values[flagged & ~y])],
        "highest-amount false negative (below threshold)":
            idx[~flagged & y][np.argmax(v.transactionamt.values[~flagged & y])],
    }
    rows = []
    for label, i in cases.items():
        order = np.argsort(-sv[i])[:3]
        for rank, j in enumerate(order, 1):
            rows.append({"case": label, "rank": rank,
                         "p_calibrated": round(float(p[i]), 6),
                         "amount": float(v.transactionamt.values[i]),
                         "is_fraud": int(y[i]),
                         "reason": reason_label(names[j], family, vgroup),
                         "value": float(X.iloc[i, j]),
                         "contribution_logodds": round(float(sv[i, j]), 4)})
    t3 = pd.DataFrame(rows)
    print("\n=== worked cases ===")
    print(t3.to_string(index=False))

    # --- single-row serving latency -------------------------------------
    ex = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
    rng = np.random.default_rng(C.SEED)
    pick = rng.choice(len(X), size=N_LATENCY, replace=False)
    ms = []
    for i in pick:
        t0 = time.perf_counter()
        ex.shap_values(X.iloc[[i]])
        ms.append((time.perf_counter() - t0) * 1000)
    med, p95 = float(np.median(ms)), float(np.percentile(ms, 95))
    print(f"\nsingle-row TreeSHAP: median {med:.2f} ms, p95 {p95:.2f} ms over {N_LATENCY:,}")

    TABLES.mkdir(parents=True, exist_ok=True)
    t3.to_csv(TABLES / "table_10_3_local_cases.csv", index=False)
    s = json.loads((C.ARTIFACT_DIR / "summary.json").read_text())
    s.update({"flagged": int(flagged.sum()), "review_rate": float(flagged.mean()),
              "precision": tp / int(flagged.sum()), "named_reason_coverage": cov,
              "latency_median_ms": med, "latency_p95_ms": p95})
    (C.ARTIFACT_DIR / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"wrote {TABLES}/table_10_3_local_cases.csv")


if __name__ == "__main__":
    main()