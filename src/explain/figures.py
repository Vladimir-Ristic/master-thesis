"""Chapter 10, Step 3.14 — Figures 10.1, 10.2 and 10.3."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.lines import Line2D
from scipy.special import expit

from src.explain import config as C
from src.explain.aggregate import load_shap, TABLES
from src.explain.load import load_estimator

FIG  = Path("reports/figures/ch10")
ANON = "Anonymised V (retained)"
NAMED_C, ANON_C = "#1f77b4", "#d62728"


def fig_10_1(sv, names, X, family):
    order = np.argsort(-np.abs(sv).mean(0))[:20]
    pick  = np.random.default_rng(C.SEED).choice(len(X), 5000, replace=False)
    plt.figure()
    shap.summary_plot(sv[np.ix_(pick, order)], X.iloc[pick, order],
                      feature_names=[names[j] for j in order], show=False,
                      plot_size=(8, 7), max_display=20)

    ax = plt.gca()
    ax.set_yticklabels([l.get_text() for l in ax.get_yticklabels()], color="black")


    plt.tight_layout()
    plt.savefig(FIG / "fig_10_1_beeswarm.png", dpi=150)
    plt.close()


def fig_10_2():
    t2 = pd.read_csv(TABLES / "table_10_2_v_groups.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(t2.missing_share, t2.share_of_anon,
               s=t2.n_columns * 12, color=ANON_C, alpha=0.75, edgecolor="white")
    for _, r in t2.iterrows():
        ax.annotate(r.group, (r.missing_share, r.share_of_anon),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xlabel("missing share (training sample)")
    ax.set_ylabel("share of anonymised attribution mass")
    plt.tight_layout()
    plt.savefig(FIG / "fig_10_2_group_attribution.png", dpi=150)
    plt.close()


def fig_10_3(sv, base, names, X, valid):
    est = load_estimator()
    cal = json.load(open("artifacts/models_v1/selection_summary.json"))["calibrator"]
    p = expit(cal["a"] * est.predict(X, raw_score=True) + cal["b"])
    y = valid.isfraud.values.astype(bool)
    flagged = p >= C.THRESHOLD
    i = np.arange(len(valid))[flagged & y][np.argmax(p[flagged & y])]   # same rule as Table 10.2
    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(values=sv[i], base_values=base,
                         data=X.iloc[i].values, feature_names=names),
        max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(FIG / "fig_10_3_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    sns.set_theme(style="whitegrid")
    FIG.mkdir(parents=True, exist_ok=True)
    sv, base, names = load_shap("valid")
    family = json.loads((C.ARTIFACT_DIR / "feature_map.json").read_text())["feature_family"]
    valid = pd.read_parquet(C.DATA_DIR / "valid_v1.parquet")
    X = valid[names]

    fig_10_1(sv, names, X, family)
    fig_10_2()
    fig_10_3(sv, base, names, X, valid)
    for f in sorted(FIG.glob("*.png")):
        print(f"wrote {f} ({f.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()