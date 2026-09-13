"""Chapter 13, Section 13.2: test-partition performance through the deployed path.

Probabilities are read from the decision log - the values the running service
persisted during the run of 2026-09-12 - and joined to labels and amounts from
the test parquet. Nothing is rescored here. A number produced by reloading the
model and predicting again would be a claim about the model; this is a claim
about the system, which is what Section 11.8 committed to.

Labels enter the chapter for the first time at this point, after every decision
was written. Cost and savings come from Chapter 9's own costs.evaluate_decision,
so the savings figure is comparable with Table 9.2's 0.7352 rather than merely
similar to it.

PR-AUC is reported with its own no-skill baseline beside it. The two partitions
have different prevalence, so the quantity that compares across them is lift over
baseline, not raw PR-AUC.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.eval import config as C
from src.eval import seal
from src.features import config as fcfg
from src.models import costs
from dataclasses import asdict
from src.serving import config as S

AMOUNT = "transactionamt"


def load_scored() -> pd.DataFrame:
    """The decision log joined to labels. One row per scored test transaction."""
    seal.assert_opened_once()
    sql = sa.text(
        """
        SELECT p.transactionid, p.probability, p.probability_raw, p.decision,
               p.threshold, p.latency_ms, (h.transactiondt / 86400)::int AS tx_day
        FROM predictions p JOIN tx_history h USING (transactionid)
        WHERE p.run_label = :label
        ORDER BY tx_day, p.transactionid
        """
    )
    with C.engine().begin() as cx:
        log = pd.read_sql(sql, cx, params={"label": C.RUN_LABEL_TEST})

    truth = pd.read_parquet(
        C.TEST_PARQUET, columns=[fcfg.KEY, fcfg.TARGET, AMOUNT]
    ).rename(columns={fcfg.KEY: "transactionid"})

    df = log.merge(truth, on="transactionid", how="inner", validate="1:1")
    if len(df) != C.EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"joined {len(df)} rows, expected {C.EXPECTED_TEST_ROWS}"
        )
    if df["threshold"].nunique() != 1:
        raise RuntimeError("the service applied more than one threshold")
    return df


def point_metrics(df: pd.DataFrame, ca: float = C.CA_REVIEW) -> dict:
    y = df[fcfg.TARGET].to_numpy(dtype=np.int8)
    p = df["probability"].to_numpy(dtype=np.float64)
    amount = df[AMOUNT].to_numpy(dtype=np.float64)
    yhat = (df["decision"] == C.DECISION_REVIEW).to_numpy(dtype=np.int8)

    threshold = float(df["threshold"].iloc[0])
    if threshold != C.CH9_THRESHOLD:
        raise RuntimeError(f"served threshold {threshold!r} is not Chapter 9's")
    # The decision is the service's, not recomputed from p. They must agree.
    drift = int((yhat != (p >= threshold).astype(np.int8)).sum())
    if drift:
        raise RuntimeError(f"{drift} decisions disagree with the served threshold")

    no_skill = float(y.mean())
    pr_auc = float(average_precision_score(y, p))
    ev = costs.evaluate_decision(
        S.DECISION_RULE, y, yhat, amount, threshold, ca=ca
    )


    return {
        "n": int(len(y)),
        "prevalence": no_skill,
        "fraud_count": int(y.sum()),
        "pr_auc": pr_auc,
        "no_skill_pr_auc": no_skill,
        "lift_over_no_skill": pr_auc / no_skill,
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_p": float(p.mean()),
        "threshold": threshold,
        **{k: v for k, v in asdict(ev).items() if k not in ("rule", "threshold")},
        "latency_ms_median": float(df["latency_ms"].median()),
    }


def day_block_bootstrap(df: pd.DataFrame, n: int = C.N_BOOTSTRAP,
                        seed: int = C.SEED) -> pd.DataFrame:
    """Resample days, not rows.

    Transactions within a day share entities and a common history state, so an
    i.i.d. row bootstrap would treat dependent observations as independent and
    return an interval that is too narrow. The day is the exchangeable unit.
    """
    rng = np.random.default_rng(seed)
    days = df["tx_day"].unique()
    groups = {d: g for d, g in df.groupby("tx_day", sort=True)}
    out = []
    for _ in range(n):
        draw = pd.concat([groups[d] for d in rng.choice(days, len(days), replace=True)])
        y = draw[fcfg.TARGET].to_numpy(dtype=np.int8)
        if y.sum() == 0:
            continue
        p = draw["probability"].to_numpy(dtype=np.float64)
        yhat = (draw["decision"] == C.DECISION_REVIEW).to_numpy(dtype=np.int8)
        tp = int(((yhat == 1) & (y == 1)).sum())
        out.append({
            "pr_auc": float(average_precision_score(y, p)),
            "precision": tp / max(int(yhat.sum()), 1),
            "recall": tp / max(int(y.sum()), 1),
        })
    return pd.DataFrame(out)


def main() -> None:
    C.ensure_dirs()
    df = load_scored()
    m = point_metrics(df)
    spec = json.loads(C.SPEC_PATH.read_text())
    v = spec["validation_comparators"]

    boot = day_block_bootstrap(df)
    ci = {c: (float(boot[c].quantile(0.025)), float(boot[c].quantile(0.975)))
          for c in boot.columns}

    v = dict(v)
    v["prevalence"] = v["no_skill_pr_auc"]
    v["lift_over_no_skill"] = v["pr_auc"] / v["no_skill_pr_auc"]
    
    rows = []
    for key, label in [
        ("n", "Transactions"), ("prevalence", "Fraud prevalence"),
        ("pr_auc", "PR-AUC"), ("no_skill_pr_auc", "No-skill PR-AUC"),
        ("lift_over_no_skill", "Lift over no-skill"), ("roc_auc", "ROC-AUC"),
        ("review_rate", "Review rate"), ("precision", "Precision"),
        ("recall", "Recall"), ("cost_ed", "Cost (example-dependent)"),
        ("savings_ed", "Savings"),
    ]:
        rows.append({
            "metric": label,
            "validation": v.get(key, v.get("n") if key == "n" else None),
            "test": m.get(key),
            "test_ci_low": ci.get(key, (None, None))[0],
            "test_ci_high": ci.get(key, (None, None))[1],
        })

    frame = pd.DataFrame(rows)
    out = C.TABLES_DIR / "table_13_1_operating_point.csv"
    frame.to_csv(out, index=False)
    boot.to_csv(C.TABLES_DIR / "bootstrap_test.csv", index=False)
    (C.ARTIFACTS_DIR / "test_metrics.json").write_text(
        json.dumps(m, indent=2, sort_keys=True) + "\n"
    )

    print(frame.to_string(index=False))
    print(f"\nbootstrap resamples: {len(boot)} of {C.N_BOOTSTRAP}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()