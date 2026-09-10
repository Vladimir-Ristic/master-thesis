"""Chapter 9 figures.

Three figures only. The Chapter 7-8 compression pass established the rule: a
chart that restates the numbers in the sentence above it becomes the sentence.
What survives here is what a table cannot carry — a curve's shape, a minimum's
location, and a systematic departure from a diagonal.

Styling matches Chapters 7-8: seaborn whitegrid, no in-figure titles (captions
live in the Word document), 150 dpi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from . import config as C, costs, metrics

sns.set_theme(style="whitegrid")
DPI = 150

ARM_LABEL = {
    "natural": "Natural distribution",
    "weighted": "Class weighting",
    "smote": "SMOTE",
    "smote_weighted": "SMOTE + weighting",
}


def _scores_dir():
    return C.ARTIFACTS / "scores"


def _load_scores():
    y = np.load(_scores_dir() / "valid_y.npy")
    amount = np.load(_scores_dir() / "valid_amount.npy")
    scores = {
        p.stem: np.load(p)
        for p in sorted(_scores_dir().glob("*.npy"))
        if p.stem not in ("valid_y", "valid_amount")
    }
    if not scores:
        raise FileNotFoundError("No validation scores found; run `make ch9-select` first.")
    return y, amount, scores


def _winner(scores, y) -> str:
    return max(scores, key=lambda k: metrics.pr_auc(y, scores[k]))


def fig_pr_curves(y, scores, family: str | None = None):
    """Figure 9.1 — precision-recall curves for one family's four arms."""
    family = family or _winner(scores, y).split("__")[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for arm in C.IMBALANCE_ARMS:
        key = C.arm_id(family, arm)
        if key not in scores:
            continue
        precision, recall, _ = precision_recall_curve(y, scores[key])
        ax.plot(recall, precision, lw=1.8,
                label=f"{ARM_LABEL[arm]} (PR-AUC {metrics.pr_auc(y, scores[key]):.3f})")

    ax.axhline(float(y.mean()), ls=":", c="0.35", lw=1.2,
               label=f"No skill ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    out = C.FIGURES / "fig_9_1_pr_curves.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_cost_threshold(y, amount, p):
    """Figure 9.2 — total cost against the decision threshold, both cost models.

    Costs are plotted on separate axes because they are in different units
    (dollars and abstract cost points). The point of the figure is that the two
    minima sit at different thresholds.
    """
    curve = costs.cost_curve(y, p, amount)
    t_ed, c_ed = costs.best_global_threshold(y, p, amount, cost_model="ed")
    t_fx, c_fx = costs.best_global_threshold(y, p, amount, cost_model="fx")
    elkan = costs.evaluate_decision(
        "elkan", y, costs.elkan_predictions(p, amount), amount, None
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax2 = ax.twinx()
    ax2.grid(False)

    ax.plot(curve["threshold"], curve["cost_ed"], lw=1.8, color="C0",
            label="Example-dependent cost (USD)")
    ax2.plot(curve["threshold"], curve["cost_fx"], lw=1.4, color="C1", ls="--",
             label="Fixed-ratio cost (points)")

    ax.axvline(t_ed, color="C0", ls=":", lw=1.2)
    ax2.axvline(t_fx, color="C1", ls=":", lw=1.2)
    ax.axhline(elkan.cost_ed, color="C2", ls="-.", lw=1.4,
               label=f"Per-transaction rule (${elkan.cost_ed:,.0f})")

    ax.annotate(
        f"global optimum\nt={t_ed:.4f}, ${c_ed:,.0f}",
        xy=(t_ed, c_ed),
        xycoords="data",
        xytext=(0.42, 0.30),
        textcoords="axes fraction",
        fontsize=9,
        color="C0",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.0),
    )

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost, example-dependent (USD)")
    ax2.set_ylabel("Total cost, fixed ratio (points)")
    ax.set_xscale("log")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left", fontsize=9)
    fig.tight_layout()
    out = C.FIGURES / "fig_9_2_cost_threshold.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def fig_calibration(y, scores, family: str | None = None, arm: str = "smote"):
    """Figure 9.3 — reliability of a treated arm, before and after correction.

    Only meaningful for an arm that altered the class balance, since undoing
    that alteration is the point.
    """
    family = family or _winner(scores, y).split("__")[0]
    key = C.arm_id(family, arm)
    raw_path = C.ARTIFACTS / "scores_raw" / f"{key}.npy"
    if key not in scores or not raw_path.exists():
        return None

    p_calibrated = scores[key]
    p_raw = np.load(raw_path)

    series = [("Uncalibrated", p_raw, "o--"), ("Fitted out of fold", p_calibrated, "s-")]

    cal_path = C.ARTIFACTS / "calibrators" / f"{key}.json"
    if cal_path.exists():
        import json

        from . import metrics as M

        payload = json.loads(cal_path.read_text())
        assumed = payload.get("diagnostics", {}).get("assumed_odds_factor")
        if assumed:
            series.insert(1, ("Assumed correction",
                              M.odds_correction(p_raw, assumed), "^:"))

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for label, p, style in series:
        rel = metrics.reliability_curve(y, p, n_bins=10)
        ax.plot(rel["mean_predicted"], rel["fraction_positive"], style, lw=1.6,
                ms=4, label=f"{label} (Brier {metrics.brier(y, p):.4f})")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls=":", c="0.35", lw=1.2, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = C.FIGURES / "fig_9_3_calibration.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def synthesis_table(y, amount, scores) -> pd.DataFrame:
    """One row per arm: ranking quality and cost side by side."""
    rows = []
    for arm_id, p in scores.items():
        model, arm = arm_id.split("__")
        t, cost = costs.best_global_threshold(y, p, amount, cost_model="ed")
        base = costs.baseline_costs(y, amount)
        elkan = costs.evaluate_decision(
            "elkan", y, costs.elkan_predictions(p, amount), amount, None
        )
        rows.append(
            {
                "model": model,
                "imbalance_arm": arm,
                "pr_auc": metrics.pr_auc(y, p),
                "roc_auc": metrics.roc_auc(y, p),
                "brier": metrics.brier(y, p),
                "global_threshold": t,
                "cost_global": cost,
                "savings_global": costs.savings(cost, base["ed_none"]),
                "cost_per_transaction_rule": elkan.cost_ed,
                "savings_per_transaction_rule": elkan.savings_ed,
                "recall_at_p50": metrics.recall_at_precision(y, p, 0.5),
            }
        )
    frame = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    frame.to_csv(C.TABLES / "table_9_8_synthesis.csv", index=False)
    return frame


def main() -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)
    y, amount, scores = _load_scores()
    win = _winner(scores, y)
    family = win.split("__")[0]
    print(f"winning arm {win}; drawing figures for the {family} family")

    print(" ", fig_pr_curves(y, scores, family))
    print(" ", fig_cost_threshold(y, amount, scores[win]))
    cal = fig_calibration(y, scores, family)
    if cal:
        print(" ", cal)

    frame = synthesis_table(y, amount, scores)
    print("\n" + frame.to_string(index=False))


if __name__ == "__main__":
    main()
