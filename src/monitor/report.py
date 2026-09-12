"""Chapter 12: the figure and the two tables that reach the draft.

Figure 12.1 carries the chapter's argument in one object: the operational signal
is flat while structural drift sits high, and both break together on the injected
days. Anything the figure carries is kept out of the tables.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.monitor import config as M
from src.monitor.load import reference

STYLE = {
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.1,
    "figure.dpi": 300,
}


def figure(dec: pd.DataFrame, ref: dict) -> None:
    op = ref["operating_point"]
    base = dec[dec.tx_day <= M.WINDOW_END]
    inj = dec[dec.tx_day > M.WINDOW_END]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(4.3, 5.4), sharex=True)
        x = dec.tx_day

        def shade(ax):
            if len(inj):
                ax.axvspan(inj.tx_day.min() - 0.5, inj.tx_day.max() + 0.5,
                           color="0.85", zorder=0)

        ax = axes[0]
        shade(ax)
        ax.plot(x, dec.review_rate, color="black", marker="o", markersize=2.5)
        ax.axhline(op["review_rate"], color="0.35", linestyle="--", linewidth=0.9)
        for s in (1 + M.REVIEW_RATE_REL_DEV, 1 - M.REVIEW_RATE_REL_DEV):
            ax.axhline(op["review_rate"] * s, color="0.55", linestyle=":", linewidth=0.8)
        ax.set_ylabel("review rate")
        ax.set_title("Operational signal, structural drift and matured labels", pad=4)
        ax.text(x.min(), op["review_rate"] * (1 + M.REVIEW_RATE_REL_DEV) * 1.01,
                f"commissioning {op['review_rate']:.4f}  ±{M.REVIEW_RATE_REL_DEV:.0%}",
                fontsize=6, color="0.35", va="bottom")

        ax = axes[1]
        shade(ax)
        ax.plot(x, dec.max_feature_psi, color="0.55", linestyle=":",
                marker="s", markersize=2, label="max feature PSI (structural)")
        ax.plot(x, dec.psi_excess, color="black", marker="o", markersize=2.5,
                label="PSI excess (emergent)")
        ax.axhline(M.PSI_EXCESS_ALERT, color="0.35", linestyle="--", linewidth=0.9)
        ax.set_ylabel("PSI")
        ax.legend(loc="upper left", frameon=False)

        ax = axes[2]
        shade(ax)
        ax.plot(x, dec.roll_recall, color="black", marker="o", markersize=2.5)
        ax.axhline(op["recall"], color="0.35", linestyle="--", linewidth=0.9)
        ax.axhline(op["recall"] * (1 - M.PERFORMANCE_REL_DROP), color="0.55",
                   linestyle=":", linewidth=0.8)
        ax.set_ylabel(f"recall, {M.PERF_WINDOW}-day pool")
        ax.set_xlabel("transaction day")

        retrain = dec[dec.decision == "retrain"]
        for _, r in retrain.iterrows():
            axes[0].annotate("retrain", xy=(r.tx_day, r.review_rate),
                             xytext=(r.tx_day - 5.5, r.review_rate * 1.06),
                             fontsize=6.5, color="black",
                             arrowprops=dict(arrowstyle="->", linewidth=0.7, color="black"))

        fig.align_ylabels(axes)
        fig.tight_layout(pad=0.6)
        out = M.FIGURES_DIR / "fig_12_1_monitoring.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def table_signals(ref: dict, pooled_named: float) -> pd.DataFrame:
    op = ref["operating_point"]
    rows = [
        ("Feature drift", "predictions.features, 20 monitored columns",
         "train_v1, days 1-119", "PSI on frozen bins, excess over window level",
         f"excess > {M.PSI_EXCESS_ALERT:.2f}", "investigate"),
        ("Score drift", "predictions.probability",
         "validation commissioning distribution", "PSI on frozen deciles",
         f"> {M.PSI_ALERT:.2f}", "investigate"),
        ("Review-rate drift", "predictions.decision",
         f"{op['review_rate']:.6f}", "relative deviation",
         f"> ±{M.REVIEW_RATE_REL_DEV:.0%}", "investigate"),
        ("Attribution drift", "predictions.reason_codes",
         f"pooled window named share {pooled_named:.1f}%",
         "shift in named share", f"> ±{M.ATTRIBUTION_SHIFT_PP:.0f} pp", "investigate"),
        ("Delayed performance", f"matured labels, {M.PERF_WINDOW}-day pool",
         f"recall {op['recall']:.6f}", "relative drop",
         f"> {M.PERFORMANCE_REL_DROP:.0%}", "retrain"),
        ("Combined rule", "the five signals above", "-",
         "performance breach, or feature and review-rate breach together",
         f"persisting {M.CONSECUTIVE_DAYS} days", "retrain"),
    ]
    df = pd.DataFrame(rows, columns=["Signal", "Source", "Reference", "Statistic",
                                     "Threshold", "Action"])
    df.to_csv(M.TABLES_DIR / "table_12_1_signals.csv", index=False)
    print(f"wrote {M.TABLES_DIR / 'table_12_1_signals.csv'}")
    return df


def table_summary(dec: pd.DataFrame, ref: dict) -> pd.DataFrame:
    op = ref["operating_point"]
    base = dec[dec.tx_day <= M.WINDOW_END]
    inj = dec[dec.tx_day > M.WINDOW_END]

    def span(s):
        return f"{s.min():.4f} to {s.max():.4f}"

    rows = [
        ("Days observed", f"{len(base)} ({base.tx_day.min()}-{base.tx_day.max()})",
         f"{len(inj)} ({inj.tx_day.min()}-{inj.tx_day.max()})", "-"),
        ("Transactions scored", f"{int(base.n.sum()):,}", f"{int(inj.n.sum()):,}", "-"),
        ("Review rate", span(base.review_rate), span(inj.review_rate),
         f"{op['review_rate']:.4f} ±{M.REVIEW_RATE_REL_DEV:.0%}"),
        ("Score PSI", span(base.score_psi), span(inj.score_psi), f"{M.PSI_ALERT:.2f}"),
        ("Max feature PSI (structural)", span(base.max_feature_psi),
         span(inj.max_feature_psi), "reported, not alerted"),
        ("PSI excess (emergent)", span(base.psi_excess), span(inj.psi_excess),
         f"{M.PSI_EXCESS_ALERT:.2f}"),
        ("Named attribution shift, pp",
         f"{base.named_share_shift_pp.min():+.1f} to {base.named_share_shift_pp.max():+.1f}",
         f"{inj.named_share_shift_pp.min():+.1f} to {inj.named_share_shift_pp.max():+.1f}",
         f"±{M.ATTRIBUTION_SHIFT_PP:.0f}"),
        (f"Recall, {M.PERF_WINDOW}-day pool", span(base.roll_recall.dropna()),
         span(inj.roll_recall), f"{op['recall']:.4f} -{M.PERFORMANCE_REL_DROP:.0%}"),
        ("Verdicts", ", ".join(f"{k} {v}" for k, v in base.decision.value_counts().items()),
         ", ".join(f"{k} {v}" for k, v in inj.decision.value_counts().items()), "-"),
    ]
    df = pd.DataFrame(rows, columns=["Quantity", "Monitoring window",
                                     "Injected days", "Threshold"])
    df.to_csv(M.TABLES_DIR / "table_12_2_window_summary.csv", index=False)
    print(f"wrote {M.TABLES_DIR / 'table_12_2_window_summary.csv'}")
    return df


def run() -> None:
    M.ensure_dirs()
    ref = reference()
    dec = pd.read_csv(M.TABLES_DIR / "decisions_daily.csv").sort_values("tx_day")
    sig = pd.read_csv(M.TABLES_DIR / "signals_daily.csv")
    pooled_named = float(
        (sig[sig.tx_day <= M.WINDOW_END].named_share
         - sig[sig.tx_day <= M.WINDOW_END].named_share_shift_pp).mean()
    )
    figure(dec, ref)
    t1 = table_signals(ref, pooled_named)
    t2 = table_summary(dec, ref)
    pd.set_option("display.width", 200, "display.max_colwidth", 46)
    print("\n== Table 12.1 ==")
    print(t1.to_string(index=False))
    print("\n== Table 12.2 ==")
    print(t2.to_string(index=False))


if __name__ == "__main__":
    run()