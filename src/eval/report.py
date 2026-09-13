"""Chapter 13: the two figures that reach the draft.

Figure 13.1 exists because a table of two PR-AUC values invites the misreading the
whole chapter guards against. The partitions have different prevalence, so each
curve is drawn against its own no-skill floor. Those floors turn out to be
near-coincident - 0.0339 against 0.0353 - which is itself the finding: the gap
between the curves cannot be explained away as a base-rate effect, because there
is barely any base-rate difference to appeal to. Validation probabilities are the
calibrated scores Chapter 9 evaluated, cached at the time; test probabilities are
the ones the service persisted during the run of 2026-09-12.

Figure 13.2 inherits Figure 12.1's line vocabulary exactly - black solid with round
markers for the emergent series, grey dotted with squares for structural, grey
dashed for thresholds - because the two figures are read against each other and a
second vocabulary would obscure the comparison. Days 148-154 are absent by
construction: 148-149 carried the synthetic transactions of Section 12.5 and are
excluded from a figure about real out-of-sample behaviour, and 150-154 were never
scored, the commissioning window having ended at day 147. The vertical rule marks
that discontinuity.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.eval import config as C
from src.eval import seal
from src.eval.metrics import load_scored
from src.features import config as fcfg
from src.models import config as MC
from src.monitor import config as M
from src.monitor.report import STYLE


def fig_13_1() -> None:
    y_v = np.load(MC.ARTIFACTS / "scores" / "valid_y.npy")
    p_v = np.load(MC.ARTIFACTS / "scores" / f"{C.DEPLOYED_ARM}.npy")
    df = load_scored()
    y_t = df[fcfg.TARGET].to_numpy()
    p_t = df["probability"].to_numpy()

    m = json.loads((C.ARTIFACTS_DIR / "test_metrics.json").read_text())
    spec = json.loads(C.SPEC_PATH.read_text())["validation_comparators"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.3, 3.2))

        for y, p, color, style, label, auc, ns in [
            (y_v, p_v, "0.55", ":", "validation", spec["pr_auc"],
             spec["no_skill_pr_auc"]),
            (y_t, p_t, "black", "-", "test", m["pr_auc"], m["no_skill_pr_auc"]),
        ]:
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, color=color, linestyle=style, linewidth=1.1,
                    label=f"{label}  PR-AUC {auc:.4f}")
            ax.axhline(ns, color=color, linestyle="--", linewidth=0.7)

        ax.plot(m["recall"], m["precision"], marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=1.1, linestyle="none", zorder=5,
                label=f"deployed, test ({m['recall']:.3f}, "
                      f"{m['precision']:.3f})")
        ax.plot(spec["recall"], spec["precision"], marker="s", markersize=4,
                markerfacecolor="white", markeredgecolor="0.55",
                markeredgewidth=1.0, linestyle="none", zorder=5,
                label=f"commissioned, validation ({spec['recall']:.3f}, "
                      f"{spec['precision']:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Precision-Recall, Validation Against Test", pad=4)
        ax.legend(loc="lower left", frameon=False,
                  bbox_to_anchor=(-0.02, 0.05))
        ax.text(0.30, 0.955,
                f"no-skill: validation {spec['no_skill_pr_auc']:.4f}, "
                f"test {m['no_skill_pr_auc']:.4f} (near-identical)",
                fontsize=6, color="0.35")

        fig.tight_layout(pad=0.6)
        out = C.FIGURES_DIR / "fig_13_1_pr_curves.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def fig_13_2() -> None:
    ch12 = pd.read_csv(M.TABLES_DIR / "decisions_daily.csv")
    ch12 = ch12[ch12.tx_day <= M.WINDOW_END]
    test = pd.read_csv(C.TABLES_DIR / "test_drift_with_variants.csv")
    op = json.loads(M.REFERENCE_PATH.read_text())["operating_point"]
    boundary = (M.WINDOW_END + C.TEST_DAY_MIN) / 2

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(4.3, 3.8), sharex=True)

        ax = axes[0]
        for d, color, style, lab in [
            (ch12, "0.55", ":", "commissioning window (Ch. 12)"),
            (test, "black", "-", "test partition"),
        ]:
            ax.plot(d.tx_day, d.review_rate, color=color, linestyle=style,
                    marker="o", markersize=2.5, label=lab)
        ax.axhline(op["review_rate"], color="0.35", linestyle="--", linewidth=0.9)
        for s in (1 + M.REVIEW_RATE_REL_DEV, 1 - M.REVIEW_RATE_REL_DEV):
            ax.axhline(op["review_rate"] * s, color="0.55", linestyle=":",
                       linewidth=0.8)
        ax.axvline(boundary, color="0.35", linewidth=0.8)
        ax.legend(loc="lower left", frameon=False, ncol=2)
        ax.annotate(
            f"commissioning {op['review_rate']:.4f}  "
            f"±{M.REVIEW_RATE_REL_DEV:.0%}",
            xy=(ch12.tx_day.min() + 1, op["review_rate"]),
            xytext=(ch12.tx_day.min() + 1, op["review_rate"] * 1.34),
            fontsize=6, color="0.35", va="bottom",
            arrowprops=dict(arrowstyle="->", linewidth=0.7, color="0.35"),
        )
        ax.set_ylabel("Review rate")
        ax.set_title("Commissioning Window Against the Test Partition", pad=4)

        ax = axes[1]
        ax.plot(ch12.tx_day, ch12.max_feature_psi, color="0.55", linestyle=":",
                marker="s", markersize=2)
        ax.plot(test.tx_day, test.max_feature_psi, color="0.55", linestyle=":",
                marker="s", markersize=2, label="max feature PSI (structural)")
        ax.plot(ch12.tx_day, ch12.psi_excess, color="black", marker="o",
                markersize=2.5)
        ax.plot(test.tx_day, test.psi_excess_carried_forward, color="black",
                marker="o", markersize=2.5, label="PSI excess (emergent)")
        ax.axhline(M.PSI_EXCESS_ALERT, color="0.35", linestyle="--",
                   linewidth=0.9)
        ax.axvline(boundary, color="0.35", linewidth=0.8)
        ax.set_ylabel("PSI")
        ax.set_xlabel("Transaction Day")
        ax.legend(loc="upper left", frameon=False)

        fig.align_ylabels(axes)
        fig.tight_layout(pad=0.6)
        out = C.FIGURES_DIR / "fig_13_2_test_monitoring.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    C.ensure_dirs()
    seal.assert_opened_once()
    fig_13_1()
    fig_13_2()


if __name__ == "__main__":
    main()