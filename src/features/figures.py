"""
Chapter 8 figures.

Styling follows the conventions already established for the Chapter 7 figures:
seaborn whitegrid, no in-figure titles (captions are set in the Word document),
150 dpi for structural plots and 300 dpi where fine detail matters.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

from . import config as cfg

BAND_COLOURS = {
    "Train": "#1f77b4",
    "Embargo": "#7f7f7f",
    "Validation": "#ff7f0e",
    "Test": "#2ca02c",
}


def _setup():
    sns.set_theme(style="whitegrid")
    cfg.REPORT_FIGURES.mkdir(parents=True, exist_ok=True)


def figure_8_1_split_timeline() -> None:
    """
    Daily volume and fraud rate with the four partitions shaded.

    This is the figure that carries the chapter's central design decision: it
    shows the partition boundaries against the same drifting fraud rate that
    Figure 7.3 established, and makes the discarded embargo interval visible
    rather than asserted.
    """
    _setup()
    # Read the interim frame rather than the partition outputs, so the embargo
    # interval is drawn with its real data and can be seen being discarded.
    base = pd.read_parquet(cfg.DATA_INTERIM / "base.parquet", columns=[cfg.TIME, cfg.TARGET])
    base["tx_day"] = base[cfg.TIME] // cfg.SECONDS_PER_DAY
    daily = base.groupby("tx_day")[cfg.TARGET].agg(["count", "mean"]).reset_index()
    daily["fraud_pct"] = daily["mean"] * 100
    daily["count_7d"] = daily["count"].rolling(7, min_periods=1).mean()
    daily["fraud_pct_7d"] = daily["fraud_pct"].rolling(7, min_periods=1).mean()

    bands = [
        ("Train", cfg.TRAIN_DAYS),
        ("Embargo", cfg.EMBARGO_DAYS),
        ("Validation", cfg.VALID_DAYS),
        ("Test", cfg.TEST_DAYS),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax in (ax1, ax2):
        for label, (lo, hi) in bands:
            ax.axvspan(lo, hi, color=BAND_COLOURS[label], alpha=0.10, lw=0)
            ax.axvline(lo, color="grey", lw=0.8, ls="--", alpha=0.7)

    ax1.plot(daily["tx_day"], daily["count"], alpha=0.3, color="tab:blue")
    ax1.plot(daily["tx_day"], daily["count_7d"], color="tab:blue", lw=2,
             label="7-day average")
    ax1.set_ylabel("Transaction Volume")
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax1.legend(loc="upper right")

    ax2.plot(daily["tx_day"], daily["fraud_pct"], alpha=0.3, color="tab:red")
    ax2.plot(daily["tx_day"], daily["fraud_pct_7d"], color="tab:red", lw=2,
             label="7-day average")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_xlabel("Time (Days)")
    ax2.legend(loc="upper right")

    # Band labels sit along the bottom of the upper panel, clear of the legend.
    lo_y, hi_y = ax1.get_ylim()
    for label, (lo, hi) in bands:
        ax1.text(
            (lo + hi) / 2, lo_y + (hi_y - lo_y) * 0.04, label,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color=BAND_COLOURS[label],
            rotation=90 if (hi - lo) < 12 else 0,
        )
    ax1.set_ylim(lo_y, hi_y)

    plt.tight_layout()
    plt.savefig(cfg.REPORT_FIGURES / "fig_8_1_split_timeline.png", dpi=150)
    plt.close()


def figure_8_2_v_reduction() -> None:
    """Columns per missingness block, before and after correlation pruning."""
    _setup()
    summary = json.loads((cfg.ARTIFACT_DIR / "encoder_summary.json").read_text())
    kept = set(summary["v_columns_kept"])

    vdf = pd.read_parquet(cfg.DATA_INTERIM / "vcols.parquet")
    v_cols = [c for c in vdf.columns if c != cfg.KEY]
    nan_counts = vdf[v_cols].isna().sum()
    share = (nan_counts / len(vdf)).round(3)

    rows = []
    for block, cols in pd.Series(v_cols).groupby(share.reindex(v_cols).to_numpy()):
        cols = list(cols)
        rows.append({"missing_share": block, "before": len(cols),
                     "after": sum(c in kept for c in cols)})
    table = pd.DataFrame(rows).sort_values("missing_share")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["before"], width=0.4, label="Before reduction", color="#9ecae1")
    ax.bar(x + 0.2, table["after"], width=0.4, label="Retained", color="#08519c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0%}" for v in table["missing_share"]], rotation=45, ha="right")
    ax.set_xlabel("Missingness block (share of rows missing)")
    ax.set_ylabel("V-columns")
    ax.legend()
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(cfg.REPORT_FIGURES / "fig_8_2_v_reduction.png", dpi=150)
    plt.close()

    table.to_csv(cfg.REPORT_TABLES / "table_8_5_v_reduction.csv", index=False)


def figure_8_3_feature_families() -> None:
    """Composition of the final feature matrix by family."""
    _setup()
    table = pd.read_csv(cfg.REPORT_TABLES / "table_8_2_feature_families.csv")
    table = table.sort_values("Count")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(table["Feature family"], table["Count"], color="#1f77b4")
    for y, v in enumerate(table["Count"]):
        ax.text(v + 0.5, y, str(v), va="center", fontsize=9)
    ax.set_xlabel("Number of features")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, table["Count"].max() * 1.12)
    plt.tight_layout()
    plt.savefig(cfg.REPORT_FIGURES / "fig_8_3_feature_families.png", dpi=150)
    plt.close()


def figure_8_4_history_coverage() -> None:
    """
    Share of transactions with usable account history, by day.

    History-based features are undefined for an account's first transaction, so
    their coverage is lowest at the start of the observation window and rises as
    accounts accumulate history. Documenting this matters: it is a genuine
    property of the feature set, it is the reason two entity features show a
    training-to-test shift in the validation report, and Chapter 9 has to know
    that early training rows carry systematically weaker features.
    """
    _setup()
    cols = ["tx_day", "acct_cnt_hist", "acct_amt_mean_hist"]
    frames = [
        pd.read_parquet(cfg.DATA_PROCESSED / f"{n}_{cfg.FEATURE_VERSION}.parquet", columns=cols)
        for n in ("train", "valid", "test")
    ]
    df = pd.concat(frames)
    daily = df.groupby("tx_day").agg(
        has_history=("acct_cnt_hist", lambda s: float((s > 0).mean())),
        mean_history=("acct_cnt_hist", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily["tx_day"], daily["has_history"] * 100, color="#1f77b4", lw=2,
            label="Transactions with prior account history (%)")
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Time (Days)")
    ax.set_ylim(0, 100)

    ax2 = ax.twinx()
    ax2.plot(daily["tx_day"], daily["mean_history"], color="#d62728", lw=1.5, alpha=0.8,
             label="Mean prior transactions per account")
    ax2.set_ylabel("Mean prior transactions")
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=9)

    for lo in (cfg.EMBARGO_DAYS[0], cfg.VALID_DAYS[0], cfg.TEST_DAYS[0]):
        ax.axvline(lo, color="grey", lw=0.8, ls="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(cfg.REPORT_FIGURES / "fig_8_4_history_coverage.png", dpi=150)
    plt.close()


def main() -> None:
    figure_8_1_split_timeline()
    figure_8_2_v_reduction()
    figure_8_3_feature_families()
    figure_8_4_history_coverage()
    print(f"figures written to {cfg.REPORT_FIGURES}")


if __name__ == "__main__":
    main()
