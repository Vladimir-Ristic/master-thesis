"""Chapter 12: protocol checks.

Nine assertions that fail loudly. Every one of them guards a way the monitoring
results could be quietly wrong rather than obviously broken: a moved reference, a
rebinned distribution, a threshold rounded somewhere, a duplicated day, or the test
partition leaking into a window it has no business being in.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.monitor import config as M
from src.monitor.drift import psi
from src.monitor.load import reference
from src.serving.artifacts import load_artifacts

CH9_THRESHOLD = 0.08704082880739572
CH9_FLAGGED = 14629
CH9_REVIEW_RATE = 0.187980


def md5_of(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> int:
    M.ensure_dirs()
    ref = reference()
    art = load_artifacts()
    rows: list[dict] = []
    failed = 0

    def check(name: str, ok, detail: str) -> None:
        nonlocal failed
        ok = bool(ok)
        if not ok:
            failed += 1
        rows.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    # 1 - the seal
    test_ids = set(pd.read_parquet(
        M.TRAIN_PARQUET.parent / "test_v1.parquet", columns=["transactionid"]
    ).transactionid.tolist())
    with M.engine().connect() as c:
        pred_ids = {r[0] for r in c.execute(sa.text("select transactionid from predictions"))}
        max_day = c.execute(sa.text(
            "select max(h.transactiondt / :spd) from predictions p "
            "join tx_history h using (transactionid)"), {"spd": M.SECONDS_PER_DAY}).scalar()
        n_dup = c.execute(sa.text(
            "select count(*) from (select transactionid from predictions "
            "group by 1 having count(*) > 1) t")).scalar()
        n_rows = c.execute(sa.text("select count(*) from predictions")).scalar()
    leaked = pred_ids & test_ids
    check("test partition sealed", not leaked,
          f"{len(leaked)} test-partition rows in the decision log")

    # 2 - the window
    check("window inside validation", M.VALID_DAY_MIN <= M.WINDOW_START and max_day <= M.VALID_DAY_MAX,
          f"days {M.WINDOW_START}-{max_day}, validation is {M.VALID_DAY_MIN}-{M.VALID_DAY_MAX}")

    # 3 - the reference has not moved
    same = (md5_of(M.TRAIN_PARQUET) == ref["sources"]["train_parquet_md5"]
            and md5_of(M.VALID_PARQUET) == ref["sources"]["valid_parquet_md5"])
    check("reference sources unchanged", same,
          "train and valid parquet md5 match the frozen reference")

    # 4 - full precision throughout
    exact = float(art.threshold) == CH9_THRESHOLD == float(ref["threshold"])
    check("threshold at full precision", exact, f"{ref['threshold']!r}")

    # 5 - the monitored set
    contract = set(art.feature_names)
    feats = ref["monitored_features"]
    ok5 = (len(feats) == M.N_MONITORED and set(feats) <= contract
           and all(ref["features"][f]["missing_share"] <= M.MAX_MISSING_SHARE for f in feats))
    check("monitored set well formed", ok5,
          f"{len(feats)} features, all in the 290-column contract, all under "
          f"{M.MAX_MISSING_SHARE:.0%} missing")

    # 6 - binning integrity: the reference re-binned through its own edges is itself
    train = pd.read_parquet(M.TRAIN_PARQUET, columns=feats)
    worst, worst_f = 0.0, ""
    for f in feats:
        spec = ref["features"][f]
        v = train[f].to_numpy(dtype="float64")
        miss = float(np.isnan(v).mean())
        counts = np.histogram(v[np.isfinite(v)], bins=np.asarray(spec["edges"], float))[0]
        cur = np.append(counts / max(counts.sum(), 1) * (1 - miss), miss)
        p = psi(np.append(np.asarray(spec["props"], float) * (1 - spec["missing_share"]),
                          spec["missing_share"]), cur)
        if p > worst:
            worst, worst_f = p, f
    check("binning reproduces the reference", worst < 1e-9,
          f"worst self-PSI {worst:.2e} ({worst_f})")

    # 7 - the commissioning point
    op = ref["operating_point"]
    ok7 = op["flagged"] == CH9_FLAGGED and abs(op["review_rate"] - CH9_REVIEW_RATE) < 5e-7
    check("commissioning point reproduces Table 9.2", ok7,
          f"{op['flagged']} of {op['n']} flagged, review rate {op['review_rate']:.6f}")

    # 8 - one row per decision
    check("decision log has no duplicates", n_dup == 0,
          f"{n_rows} rows, {n_dup} duplicated transactionids")

    # 9 - the positive control
    dec = pd.read_csv(M.TABLES_DIR / "decisions_daily.csv")
    base = dec[dec.tx_day <= M.WINDOW_END]
    n_retrain = int((base.decision == "retrain").sum())
    check("baseline window triggers no retrain", n_retrain == 0,
          f"days {M.WINDOW_START}-{M.WINDOW_END}: "
          f"{base.decision.value_counts().to_dict()}")

    out = pd.DataFrame(rows)
    out.to_csv(M.TABLES_DIR / "table_12_4_protocol_checks.csv", index=False)
    if failed:
        print(f"\n{failed} of {len(rows)} protocol checks FAILED")
    else:
        print(f"\nall {len(rows)} protocol checks passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(run())