"""Chapter 13, RQ1: class imbalance handling on the test partition.

The model family is held fixed at LightGBM and only the imbalance
treatment varies, because that is what we ask in RQ1. Chapter 9's own refit_and_score
is the fit path - not an equivalent reimplementation - so the comparison runs
through the code that produced Table 9.1, with build_pipeline applying the
resampler inside the fit and nowhere else.

Two asymmetries are recorded rather than smoothed over:

  1. The deployed arm's headline test numbers come from the REGISTERED model,
     scored through the service (run_test.py). The figures this module produces
     for lgbm__natural are a refit, and Section 10.7 measured what that costs:
     0.5986 against the registered 0.598894. One order of magnitude below the
     arm spread it might disturb, and reported alongside so a reader can see it.
  2. Each arm loads its own calibrator. The Platt coefficients differ materially
     between arms and the implied odds factors differ by up to 27.5x; sharing one
     calibrator would compare score scales rather than models.

No selection is made here. The deployed model does not change whatever this
produces, and the comparison was named in the measurement specification before
the seal was opened.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.eval import config as C
from src.eval import seal
from src.models import config as MC
from src.models import data, select
from src.models.calibration import Calibrator


def _calibrator(arm_id: str) -> Calibrator:
    path = MC.ARTIFACTS / "calibrators" / f"{arm_id}.json"
    return Calibrator.from_dict(json.loads(path.read_text()))


def _split(arm_id: str) -> tuple[str, str]:
    model, arm = arm_id.split("__", 1)
    return model, arm


def _score_row(arm_id: str, partition: str, y, p, elapsed: float) -> dict:
    no_skill = float(y.mean())
    pr_auc = float(average_precision_score(y, p))
    t = C.operating_threshold()
    flagged = p >= t
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return {
        "arm_id": arm_id,
        "partition": partition,
        "n": int(len(y)),
        "prevalence": no_skill,
        "pr_auc": pr_auc,
        "no_skill_pr_auc": no_skill,
        "lift_over_no_skill": pr_auc / no_skill,
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_p": float(p.mean()),
        "flagged": int(flagged.sum()),
        "review_rate": float(flagged.mean()),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fit_seconds": round(elapsed, 1),
    }


def run() -> pd.DataFrame:
    # The Chapter 8 flag and the Chapter 13 ledger are the same commitment at two
    # layers. The flag flips only because the ledger says the seal is open.
    led = seal.assert_opened_once()
    MC.ALLOW_TEST_READ = True
    data.C.ALLOW_TEST_READ = True
    print(f"seal opened at {led['opened_at']}; ALLOW_TEST_READ enabled")

    train = data.load_partition("train")
    valid = data.load_partition("valid")
    test = data.load_partition("test")
    data.assert_same_contract(train, test)
    data.assert_disjoint(train, test)
    print(f"{train!r}\n{valid!r}\n{test!r}")

    rows = []
    for arm_id in C.RQ1_ARMS:
        model, arm = _split(arm_id)
        cal = _calibrator(arm_id)
        for partition, part in (("valid", valid), ("test", test)):
            t0 = time.perf_counter()
            _, _, p = select.refit_and_score(
                model, arm, train, part, seed=C.SEED, calibrator=cal
            )
            rows.append(_score_row(arm_id, partition, part.y, np.asarray(p),
                                   time.perf_counter() - t0))
            print(f"  {arm_id:24s} {partition:5s} "
                  f"PR-AUC {rows[-1]['pr_auc']:.6f}  "
                  f"lift {rows[-1]['lift_over_no_skill']:.2f}")

    frame = pd.DataFrame(rows)
    C.ensure_dirs()
    out = C.TABLES_DIR / "table_13_2_rq1.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return frame


def main() -> None:
    frame = run()
    cols = ["arm_id", "partition", "pr_auc", "no_skill_pr_auc",
            "lift_over_no_skill", "roc_auc", "review_rate", "recall"]
    print(frame[cols].to_string(index=False))


if __name__ == "__main__":
    main()