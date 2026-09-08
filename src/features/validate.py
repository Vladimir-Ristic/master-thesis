"""
Post-build verification of the feature sets.

A feature pipeline that leaks does not fail loudly; it produces excellent numbers
that do not survive deployment. The checks below are therefore run as a build
step rather than left to inspection, and their results are reported in the
chapter so the evaluation in Chapter 9 rests on something audited.

Two categories of check are performed. Structural checks are assertions about the
partitions themselves -- disjointness, ordering, an empty embargo, a stable column
contract. Statistical checks look for the signatures leakage leaves behind: a
single feature that separates the classes almost perfectly, or an end-to-end
score high enough to be implausible for this problem.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as cfg

META_COLS = [cfg.KEY, cfg.TARGET, "tx_day"]


def load_partitions() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "valid", "test"):
        out[name] = pd.read_parquet(
            cfg.DATA_PROCESSED / f"{name}_{cfg.FEATURE_VERSION}.parquet"
        )
    return out


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


# ------------------------------------------------------------------ checks
def check_structure(parts: dict[str, pd.DataFrame]) -> list[dict]:
    results = []

    def record(name, passed, detail):
        results.append({"Check": name, "Result": "PASS" if passed else "FAIL",
                        "Detail": detail})

    cols = {k: _feature_cols(v) for k, v in parts.items()}
    same = cols["train"] == cols["valid"] == cols["test"]
    record("Identical feature contract across partitions", same,
           f"{len(cols['train'])} features, order preserved" if same
           else "column sets or order differ")

    forbidden = cfg.FORBIDDEN_IN_FEATURE_MATRIX & set(cols["train"])
    record("Target and key absent from feature matrix", not forbidden,
           "clean" if not forbidden else f"found {sorted(forbidden)}")

    ids = {k: set(v[cfg.KEY]) for k, v in parts.items()}
    overlaps = {
        "train/valid": ids["train"] & ids["valid"],
        "train/test": ids["train"] & ids["test"],
        "valid/test": ids["valid"] & ids["test"],
    }
    worst = max(len(v) for v in overlaps.values())
    record("No transaction appears in two partitions", worst == 0,
           "disjoint" if worst == 0 else f"{worst} shared transaction ids")

    ranges = {k: (int(v["tx_day"].min()), int(v["tx_day"].max())) for k, v in parts.items()}
    ordered = ranges["train"][1] < ranges["valid"][0] < ranges["valid"][1] < ranges["test"][0]
    record("Partitions are strictly chronological", ordered,
           " -> ".join(f"{k} {ranges[k][0]}-{ranges[k][1]}" for k in ("train", "valid", "test")))

    gap = ranges["valid"][0] - ranges["train"][1] - 1
    required = cfg.MAX_WINDOW_SECONDS // cfg.SECONDS_PER_DAY
    record("Embargo interval is empty and wide enough", gap >= required,
           f"{gap} day gap, widest feature window is {required} days")

    nunique = parts["train"][cols["train"]].nunique(dropna=False)
    constant = nunique[nunique <= 1].index.tolist()
    record("No constant features in the training partition", not constant,
           "none" if not constant else f"{len(constant)} constant: {constant[:5]}")

    return results


def check_single_feature_separation(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Rank features by how well each separates the classes on its own.

    A legitimate fraud feature is weakly informative in isolation; a feature that
    reaches near-perfect separation alone is almost always carrying the label in
    disguise. The ranking is reported, and the highest value is asserted against
    a threshold.
    """
    train = parts["train"]
    # One ROC-AUC per feature over 400k rows is minutes of work for a diagnostic;
    # a fixed sample gives the same ranking at a fraction of the cost.
    if len(train) > 200_000:
        train = train.sample(n=200_000, random_state=42)
    y = train[cfg.TARGET].to_numpy()
    rows = []
    for col in _feature_cols(train):
        x = train[col].to_numpy(dtype="float64")
        mask = np.isfinite(x)
        if mask.sum() < 100 or len(np.unique(y[mask])) < 2:
            continue
        auc = roc_auc_score(y[mask], x[mask])
        rows.append({"feature": col, "auc": max(auc, 1 - auc),
                     "coverage": float(mask.mean())})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    return out


def check_unseen_categories(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of rows in each evaluation partition hitting the unseen-category code."""
    from .encoders import FeatureEncoder

    encoder = FeatureEncoder.load(cfg.ARTIFACT_DIR)
    encoded = [c for c in encoder.ordinal_maps_ if c in parts["train"].columns]
    rows = []
    for name in ("valid", "test"):
        df = parts[name]
        for col in encoded:
            share = float((df[col] == cfg.UNSEEN_CATEGORY_CODE).mean())
            if share > 0:
                rows.append({"partition": name, "feature": col, "unseen_share": share})
    out = pd.DataFrame(rows, columns=["partition", "feature", "unseen_share"])
    return out.sort_values("unseen_share", ascending=False)


def check_missingness_drift(parts: dict[str, pd.DataFrame], threshold: float = 0.25) -> pd.DataFrame:
    """Features whose missing rate moves sharply between training and test."""
    cols = _feature_cols(parts["train"])
    tr = parts["train"][cols].isna().mean()
    te = parts["test"][cols].isna().mean()
    delta = (te - tr).abs().sort_values(ascending=False)
    out = pd.DataFrame({"train_missing": tr, "test_missing": te, "abs_delta": delta})
    return out[out["abs_delta"] > threshold].sort_values("abs_delta", ascending=False)


def check_end_to_end(parts: dict[str, pd.DataFrame]) -> dict:
    """
    Fit a small, untuned LightGBM as a leakage probe -- not as a result.

    This is deliberately a throwaway model with default-ish settings. Its only
    purpose is to establish that the feature set behaves plausibly: a validation
    PR-AUC close to 1.0 on a 3.5% positive rate would indicate leakage rather
    than success. All modelling proper belongs to Chapter 9.
    """
    import lightgbm as lgb

    cols = _feature_cols(parts["train"])
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(parts["train"][cols], parts["train"][cfg.TARGET])

    scores = {}
    for name in ("train", "valid", "test"):
        p = model.predict_proba(parts[name][cols])[:, 1]
        y = parts[name][cfg.TARGET]
        scores[name] = {
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "positive_rate": float(y.mean()),
        }

    importance = (
        pd.Series(model.feature_importances_, index=cols)
        .sort_values(ascending=False)
        .head(25)
    )
    return {"scores": scores, "top_features": importance}


# -------------------------------------------------------------------- main
def run(verbose: bool = True) -> dict:
    parts = load_partitions()
    cfg.REPORT_TABLES.mkdir(parents=True, exist_ok=True)

    structure = pd.DataFrame(check_structure(parts))
    separation = check_single_feature_separation(parts)
    unseen = check_unseen_categories(parts)
    drift = check_missingness_drift(parts)
    probe = check_end_to_end(parts)

    max_auc = float(separation["auc"].max()) if len(separation) else np.nan
    structure.loc[len(structure)] = {
        "Check": "No single feature separates the classes",
        "Result": "PASS" if max_auc <= cfg.MAX_SINGLE_FEATURE_AUC else "FAIL",
        "Detail": f"highest single-feature ROC-AUC {max_auc:.3f} "
                  f"({separation.iloc[0]['feature'] if len(separation) else 'n/a'})",
    }
    plausible = probe["scores"]["valid"]["pr_auc"] < 0.99
    structure.loc[len(structure)] = {
        "Check": "End-to-end probe score is plausible",
        "Result": "PASS" if plausible else "FAIL",
        "Detail": f"validation PR-AUC {probe['scores']['valid']['pr_auc']:.4f}",
    }

    structure.to_csv(cfg.REPORT_TABLES / "table_8_3_leakage_checks.csv", index=False)
    separation.head(30).to_csv(cfg.REPORT_TABLES / "table_8_4_single_feature_auc.csv", index=False)
    (cfg.ARTIFACT_DIR / "validation_report.json").write_text(
        json.dumps(
            {
                "structure": structure.to_dict(orient="records"),
                "max_single_feature_auc": max_auc,
                "probe_scores": probe["scores"],
                "unseen_category_rows": unseen.to_dict(orient="records")[:20],
                "missingness_drift": drift.reset_index(names="feature").to_dict(orient="records")[:20],
            },
            indent=2,
        )
    )

    if verbose:
        print(structure.to_string(index=False))
        print("\nTop single-feature ROC-AUC (training partition):")
        print(separation.head(10).to_string(index=False))
        print("\nLeakage probe (untuned LightGBM, not a Chapter 9 result):")
        print(pd.DataFrame(probe["scores"]).T.to_string())
        print("\nTop 15 probe feature importances:")
        print(probe["top_features"].head(15).to_string())
        if len(unseen):
            print("\nUnseen-category exposure:")
            print(unseen.head(10).to_string(index=False))
        if len(drift):
            print("\nMissingness drift train -> test:")
            print(drift.head(10).to_string())

    failures = structure.loc[structure["Result"] == "FAIL", "Check"].tolist()
    if failures:
        raise AssertionError(f"validation failed: {failures}")
    return {"structure": structure, "separation": separation, "probe": probe}


def main() -> None:
    run(verbose=True)


if __name__ == "__main__":
    main()
