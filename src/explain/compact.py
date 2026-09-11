"""Chapter 10, Step 3.12 — the cost of an interpretability constraint."""
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.explain import config as C
from src.explain.aggregate import load_shap
from src.models import costs, data, select
from src.explain.load import load_model, load_estimator

TABLES = Path("reports/tables/ch10")


def main():
    # ranking comes from TRAINING rows only — protocol check 8
    sv, _, names = load_shap("train_sample")
    ordered = [names[i] for i in np.argsort(-np.abs(sv).mean(0))]

    train, valid = data.load_partition("train"), data.load_partition("valid")
    assert not set(np.asarray(train.key)) & set(np.asarray(valid.key)), \
        "ABORT: selection sample overlaps validation"

    reg_pipe, reg_est = load_model(), load_estimator()

    rows, baseline = [], None
    for k in [len(names)] + sorted(C.TOP_K, reverse=True):
        cols = ordered[:k]
        t0 = time.time()
        if k == len(names):
            # the registered model rescored -- never a refit (guide 3.4)
            p = reg_pipe.predict_proba(valid.X[reg_est.booster_.feature_name()])[:, 1]
        else:
            _, _, p = select.refit_and_score(
                "lgbm", "natural",
                replace(train, X=train.X[cols]), replace(valid, X=valid.X[cols]))
        fit_s = time.time() - t0

        thr, _ = costs.best_global_threshold(valid.y, p, valid.amount, ca=C.CA_REVIEW)
        d = costs.evaluate_decision("global", valid.y, (p >= thr).astype(int),
                                    valid.amount, thr, ca=C.CA_REVIEW)
        pr = average_precision_score(valid.y, p)
        baseline = pr if baseline is None else baseline
        rows.append({"k": k, "pr_auc": pr, "roc_auc": roc_auc_score(valid.y, p),
                     "d_pr_auc": pr - baseline, "threshold": thr,
                     "review_rate": d.review_rate, "precision": d.precision,
                     "recall": d.recall, "cost_ed": d.cost_ed,
                     "savings": d.savings_ed, "fit_s": fit_s})
        print(f"k={k:>4}  pr_auc {pr:.4f}  d {pr-baseline:+.4f}  "
              f"cost {d.cost_ed:>10,.0f}  savings {d.savings_ed:.3f}  "
              f"{fit_s:.0f}s", flush=True)

    t4 = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    t4.to_csv(TABLES / "table_10_4_compact.csv", index=False)
    print("\n=== compact models, top-k by training-set mean |SHAP| ===")
    print(t4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {TABLES}/table_10_4_compact.csv")


if __name__ == "__main__":
    main()