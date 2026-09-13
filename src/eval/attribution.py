"""Chapter 13, Section 13.4: RQ2 on the test partition.

Three questions, three bases, kept apart because Chapter 12 already had to warn
that two of them are different quantities (signals.py:94):

  1. Whole-sample attribution structure. TreeSHAP over the test partition,
     aggregated on Chapter 10's headline basis - two-block sum-then-absolute -
     against its 77.638% named share. Chapter 10 computed that on the full
     77,822-row validation partition, not a sample.
  2. The same figure on the 5,000-row seed-42 sample the measurement
     specification actually commits to. DEVIATION: the spec assumed Chapter 10's
     basis was a sample. It was not. Both are computed and both are reported;
     neither is chosen on the basis of which is more flattering.
  3. Reason codes as served. What the running service actually emitted for
     flagged test transactions - the deployed explanation policy, Chapter 10
     Fork 2, top-3 codes on flagged rows only.

TreeSHAP only. feature_perturbation="tree_path_dependent", identical to
Chapter 10, because the setting changes the attribution basis.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
import sqlalchemy as sa

from src.eval import config as C
from src.eval import seal
from src.explain import config as EC
from src.explain.load import load_estimator
from src.explain.shap_values import explain
from src.features import config as fcfg

ANON = "Anonymised V (retained)"


def build_cache() -> None:
    """Compute and cache TreeSHAP for the test partition, Chapter 10's way."""
    seal.assert_opened_once()
    EC.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    est = load_estimator()
    names = est.booster_.feature_name()
    ex = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
    X = pd.read_parquet(C.TEST_PARQUET)[names]
    explain("test", X, est, ex)


def two_block(sv: np.ndarray, names: list[str], family: dict) -> dict:
    """Chapter 10's headline basis, arithmetic unchanged."""
    col = {f: i for i, f in enumerate(names)}
    mean_abs = np.abs(sv).mean(0)
    ia = [col[f] for f in names if family[f] == ANON]
    inn = [col[f] for f in names if family[f] != ANON]
    ma = float(np.abs(sv[:, ia].sum(1)).mean())
    mn = float(np.abs(sv[:, inn].sum(1)).mean())
    ta = float(mean_abs[ia].sum())
    tn = float(mean_abs[inn].sum())
    return {
        "n_rows": int(sv.shape[0]),
        "n_named": len(inn),
        "n_anonymised": len(ia),
        "named_share_two_block": mn / (mn + ma),
        "anon_share_two_block": ma / (mn + ma),
        "named_share_abs_then_sum": tn / (tn + ta),
        "anon_share_abs_then_sum": ta / (tn + ta),
        "cancellation_named": mn / tn,
        "cancellation_anon": ma / ta,
    }


def served_reason_codes() -> dict:
    """What the service emitted. Not the same quantity as the SHAP shares."""
    sql = sa.text(
        """
        SELECT p.transactionid, p.reason_codes
        FROM predictions p
        WHERE p.run_label = :label AND p.decision = :review
        """
    )
    with C.engine().begin() as cx:
        rows = pd.read_sql(sql, cx, params={"label": C.RUN_LABEL_TEST,
                                            "review": C.DECISION_REVIEW})
    fmap = json.loads((EC.ARTIFACT_DIR / "feature_map.json").read_text())
    family = fmap["feature_family"]

    n_codes = n_named = 0
    with_named = 0
    for codes in rows["reason_codes"]:
        codes = codes if isinstance(codes, list) else json.loads(codes or "[]")
        named_here = 0
        for c in codes:
            feat = c.get("feature")
            n_codes += 1
            if family.get(feat, ANON) != ANON:
                n_named += 1
                named_here += 1
        with_named += int(named_here > 0)
    return {
        "flagged_rows": int(len(rows)),
        "reason_codes_emitted": n_codes,
        "reason_code_named_share": n_named / n_codes if n_codes else float("nan"),
        "named_reason_coverage": with_named / len(rows) if len(rows) else float("nan"),
    }


def main() -> None:
    C.ensure_dirs()
    seal.assert_opened_once()

    path = EC.SHAP_DIR / "test.npz"
    if not path.exists():
        print("no test SHAP cache; computing (10-20 min)")
        build_cache()

    z = np.load(path, allow_pickle=False)
    sv = z["shap"]
    names = [str(f) for f in z["feature_names"]]
    family = json.loads((EC.ARTIFACT_DIR / "feature_map.json").read_text())["feature_family"]
    ch10 = json.loads((EC.ARTIFACT_DIR / "summary.json").read_text())

    full = two_block(sv, names, family)
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(sv.shape[0], size=C.SHAP_SAMPLE_N, replace=False)
    sample = two_block(sv[idx], names, family)
    served = served_reason_codes()

    out = {
        "basis": "two-block sum-then-absolute, Chapter 10 headline",
        "chapter10_validation": {
            "n_rows": 77822,
            "named_share_two_block": ch10["named_share_two_block"],
            "named_share_abs_then_sum": ch10["named_share_abs_then_sum"],
            "named_reason_coverage": ch10["named_reason_coverage"],
        },
        "test_full_partition": full,
        "test_sample_5000": sample,
        "test_served": served,
        "spec_deviation": (
            "The specification named a 5,000-row sample for RQ2, assuming Chapter "
            "10's figure was sampled. Chapter 10 used the full validation "
            "partition. Both are computed; the full-partition figure is the "
            "like-for-like comparison."
        ),
    }
    (C.ARTIFACTS_DIR / "attribution_test.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame([
        {"basis": "Ch10 validation (77,822)", **out["chapter10_validation"]},
        {"basis": "Test, full partition (78,542)", **full},
        {"basis": "Test, 5,000 sample", **sample},
    ]).to_csv(C.TABLES_DIR / "attribution_test.csv", index=False)

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()