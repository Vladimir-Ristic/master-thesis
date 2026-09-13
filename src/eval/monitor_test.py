"""Chapter 13, Section 13.6: the monitor over the test run.

Chapter 12 built a monitor and froze a snapshot of what "normal"
looked like, before the test data had ever been scored. This module points that
same monitor at the 28 test days and asks whether it would have raised an alarm.
Nothing about the monitor is adjusted.
Chapter 12's drift code runs unchanged against the unchanged frozen reference.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd

from src.eval import config as C
from src.eval import seal
from src.monitor import config as M
from src.monitor import drift, trigger

CH12_OUTPUTS = ["drift_daily.csv", "psi_by_feature.csv"]


def _md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def run_drift() -> pd.DataFrame:
    """Run Chapter 12's drift over the test window without disturbing its tables."""
    seal.assert_opened_once()
    ref_md5 = _md5(M.REFERENCE_PATH)
    if ref_md5 != C.EXPECTED_MD5["artifacts/monitor_v1/reference.json"]:
        raise RuntimeError("the frozen reference has changed")

    snapshot = {}
    for name in CH12_OUTPUTS:
        src = M.TABLES_DIR / name
        if src.exists():
            dst = C.ARTIFACTS_DIR / f"ch12_snapshot_{name}"
            shutil.copy2(src, dst)
            snapshot[name] = (dst, _md5(src))

    try:
        out = drift.run(C.TEST_DAY_MIN, C.TEST_DAY_MAX, with_evidently=True)
        for name in CH12_OUTPUTS:
            produced = M.TABLES_DIR / name
            if produced.exists():
                shutil.move(str(produced), C.TABLES_DIR / f"test_{name}")
    finally:
        for name, (dst, _) in snapshot.items():
            shutil.copy2(dst, M.TABLES_DIR / name)

    for name, (_, want) in snapshot.items():
        got = _md5(M.TABLES_DIR / name)
        if got != want:
            raise RuntimeError(f"Chapter 12 artifact {name} was not restored intact")
    if _md5(M.REFERENCE_PATH) != ref_md5:
        raise RuntimeError("the reference changed during the run")
    print(f"Chapter 12 artifacts restored intact ({len(snapshot)} files)")
    return out


def emergent_variants(test_psi: pd.DataFrame,
                      ch12_psi: pd.DataFrame) -> dict[str, pd.DataFrame]:
    carried = trigger.emergent(pd.concat([ch12_psi, test_psi], ignore_index=True))
    carried = carried.loc[carried.index >= C.TEST_DAY_MIN]

    level = test_psi.groupby("feature").psi.median()
    frame = test_psi.assign(excess=test_psi.psi - test_psi.feature.map(level))
    top = frame.loc[frame.groupby("tx_day").excess.idxmax()].set_index("tx_day")
    self_ref = pd.DataFrame({"psi_excess": top.excess,
                             "psi_excess_feature": top.feature})
    return {"carried_forward": carried, "self_referential": self_ref}


def main() -> None:
    C.ensure_dirs()
    run_drift()

    test_psi = pd.read_csv(C.TABLES_DIR / "test_psi_by_feature.csv")
    ch12_psi = pd.read_csv(M.TABLES_DIR / "psi_by_feature.csv")
    daily = pd.read_csv(C.TABLES_DIR / "test_drift_daily.csv")

    variants = emergent_variants(test_psi, ch12_psi)
    for name, frame in variants.items():
        daily = daily.join(
            frame.rename(columns={"psi_excess": f"psi_excess_{name}",
                                  "psi_excess_feature": f"psi_excess_feature_{name}"}),
            on="tx_day",
        )
    daily["sig_feature_carried_forward"] = daily.psi_excess_carried_forward > M.PSI_EXCESS_ALERT
    daily["sig_feature_self_referential"] = daily.psi_excess_self_referential > M.PSI_EXCESS_ALERT
    daily["sig_score"] = daily.score_psi > M.PSI_ALERT
    daily["sig_review"] = daily.review_rate_rel_dev.abs() > M.REVIEW_RATE_REL_DEV

    out = C.TABLES_DIR / "test_drift_with_variants.csv"
    daily.to_csv(out, index=False)

    ch12_daily = pd.read_csv(M.TABLES_DIR / "drift_daily.csv")
    ch12_win = ch12_daily[ch12_daily.tx_day <= M.WINDOW_END]
    summary = []
    for q in ["review_rate", "score_psi", "max_feature_psi"]:
        summary.append({
            "quantity": q,
            "ch12_window_min": ch12_win[q].min(), "ch12_window_max": ch12_win[q].max(),
            "test_min": daily[q].min(), "test_max": daily[q].max(),
        })
    sf = pd.DataFrame(summary)
    sf.to_csv(C.TABLES_DIR / "table_13_3_monitoring.csv", index=False)

    print(sf.to_string(index=False))
    print("\nsignals fired on the test window:")
    for c in ["sig_feature_carried_forward", "sig_feature_self_referential",
              "sig_score", "sig_review"]:
        print(f"  {c:34s} {int(daily[c].sum()):2d} of {len(daily)} days")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()