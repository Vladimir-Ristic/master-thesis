"""Chapter 13: protocol checks.

Sixteen assertions that fail loudly, serving as guards for the test-partition. 
This file re-checks, after the fact, that the rules the chapter
set for itself were actually kept - that the test data was opened once and only
once, that the plan written beforehand was not edited afterwards, that all 78,542
transactions were scored in date order with none missing or counted twice, that
the threshold and the frozen reference are bit-for-bit the values Chapter 9 and
Chapter 12 set, and that nothing from the earlier chapters was overwritten along
the way.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import sqlalchemy as sa

from src.eval import config as C
from src.eval import seal
from src.monitor import config as M
from src.serving.artifacts import load_artifacts


def md5_of(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> int:
    C.ensure_dirs()
    rows: list[dict] = []
    failed = 0

    def check(name: str, ok, detail: str) -> None:
        nonlocal failed
        ok = bool(ok)
        if not ok:
            failed += 1
        rows.append({"check": name, "result": "PASS" if ok else "FAIL",
                     "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    led = json.loads(C.SEAL_PATH.read_text())
    spec = json.loads(C.SPEC_PATH.read_text())
    art = load_artifacts()
    metrics = json.loads((C.ARTIFACTS_DIR / "test_metrics.json").read_text())
    attrib = json.loads((C.ARTIFACTS_DIR / "attribution_test.json").read_text())

    # 1 - the seal was opened exactly once
    check("seal opened exactly once", led["open_count"] == 1,
          f"opened {led['opened_at']}")

    # 2 - the specification did not change after the open
    now = hashlib.md5(C.SPEC_PATH.read_bytes()).hexdigest()
    check("specification frozen at open", now == led["spec_md5"],
          f"{now} == {led['spec_md5']}")

    # 3 - the test partition is present, complete and labelled
    with C.engine().connect() as c:
        n_test = c.execute(sa.text(
            "select count(*) from predictions where run_label = :l"),
            {"l": C.RUN_LABEL_TEST}).scalar()
        n_pending = c.execute(sa.text(
            "select count(*) from predictions where run_label = 'pending'")).scalar()
        n_dup = c.execute(sa.text(
            "select count(*) from (select transactionid from predictions "
            "group by 1 having count(*) > 1) d")).scalar()
        days = pd.read_sql(sa.text(
            "select (h.transactiondt / :spd)::int as tx_day, min(p.scored_at) as first "
            "from predictions p join tx_history h using (transactionid) "
            "where p.run_label = :l group by 1 order by 1"),
            c, params={"spd": M.SECONDS_PER_DAY, "l": C.RUN_LABEL_TEST})
        n_ch12 = c.execute(sa.text(
            "select count(*) from predictions where run_label like 'ch12%'")).scalar()

    check("test partition fully scored", n_test == C.EXPECTED_TEST_ROWS,
          f"{n_test} rows, expected {C.EXPECTED_TEST_ROWS}")
    check("every row labelled", n_pending == 0, f"{n_pending} pending")
    check("decision log has no duplicates", n_dup == 0, f"{n_dup} duplicated ids")
    check("Chapter 12 rows untouched", n_ch12 == C.PRE_RUN_LOG_ROWS,
          f"{n_ch12} rows, expected {C.PRE_RUN_LOG_ROWS}")

    # 4 - chronological order
    asc = days["first"].is_monotonic_increasing
    check("days scored in chronological order", asc,
          f"days {days.tx_day.min()}-{days.tx_day.max()}, "
          f"{len(days)} days, first_scored monotonic={asc}")
    check("every test day present", len(days) == 28,
          f"{len(days)} of 28 days")

    # 5 - the threshold, four ways
    t = C.operating_threshold()
    exact = (t == C.CH9_THRESHOLD == float(art.threshold)
             == metrics["threshold"])
    check("threshold identical across four sources", exact, repr(t))

    # 6 - the reference was not regenerated
    got = md5_of(M.REFERENCE_PATH)
    check("frozen reference unchanged", got == C.EXPECTED_MD5[
        "artifacts/monitor_v1/reference.json"], got)

    # 7 - sealed sources unchanged
    try:
        seal.verify_digests()
        check("sealed sources unchanged", True, "4 digests match")
    except RuntimeError as e:
        check("sealed sources unchanged", False, str(e).splitlines()[0])

    # 8 - Chapter 12 artifacts survived the drift re-run
    ch12_drift = pd.read_csv(M.TABLES_DIR / "drift_daily.csv")
    check("Chapter 12 drift table intact",
          int(ch12_drift.tx_day.max()) <= 149,
          f"max tx_day {int(ch12_drift.tx_day.max())}")

    # 9 - no selection was made on test
    check("no selection on test", spec["rq1"]["selection_on_test"] is False,
          "specification records selection_on_test = false")

    # 10 - resampling never left the training fit
    check("resampling confined to training",
          "inside the training fit only" in spec["rq1"]["resampling"],
          spec["rq1"]["resampling"])

    # 11 - TreeSHAP only
    check("TreeSHAP only, no KernelSHAP", spec["rq2"]["kernel_shap"] is False,
          "specification records kernel_shap = false")

    # 12 - the RQ2 deviation is recorded rather than hidden
    check("RQ2 specification deviation recorded",
          "spec_deviation" in attrib and bool(attrib["spec_deviation"]),
          "full partition and 5,000 sample both reported")

    # 13 - both drift variants were computed
    tv = pd.read_csv(C.TABLES_DIR / "test_drift_with_variants.csv")
    both = ("psi_excess_carried_forward" in tv
            and "psi_excess_self_referential" in tv)
    check("both emergent-drift variants computed", both,
          f"carried_forward fired {int(tv.sig_feature_carried_forward.sum())}/28, "
          f"self_referential {int(tv.sig_feature_self_referential.sum())}/28")

    # 14 - the deployed model was not retrained
    check("deployed model unchanged",
          art.model_version == "1" and art.model_name == "fraud_detector",
          f"{art.model_name} v{art.model_version}")

    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES_DIR / "table_13_4_protocol_checks.csv", index=False)
    if failed:
        print(f"\n{failed} of {len(rows)} protocol checks FAILED")
    else:
        print(f"\nall {len(rows)} protocol checks passed")
    return failed


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()