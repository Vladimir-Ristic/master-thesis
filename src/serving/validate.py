"""Chapter 11 protocol checks.

Each is written to fail loudly rather than to reassure. Three assert facts about
the training run rather than about the service, because a serving layer that agrees
with itself and disagrees with Chapter 9 is precisely the failure this chapter
exists to prevent.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import config as fcfg
from src.serving import config as C
from src.serving.artifacts import load_artifacts
from src.serving.db import connect
from src.serving.history import aggregates_for

SEAL_DT = 155 * fcfg.SECONDS_PER_DAY


def run() -> int:
    art = load_artifacts()
    r: list[tuple[str, bool, str]] = []

    def check(name: str, ok, detail: str) -> None:
        r.append((name, bool(ok), detail))

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT max(transactiondt), count(*) FROM tx_history")
        max_dt, n_hist = cur.fetchone()
    check("1. history store holds no test-partition rows", max_dt < SEAL_DT,
          f"max dt {max_dt:,} < {SEAL_DT:,} over {n_hist:,} rows")

    names = list(art.booster.feature_name())
    check("2. registered model matches the pinned contract",
          names == art.feature_names and len(names) == C.EXPECTED_N_FEATURES,
          f"{len(names)} features, {art.booster.num_trees()} trees")

    check("3. fitted encoder produces the contract",
          list(art.encoder.feature_names_) == art.feature_names,
          f"{len(art.encoder.feature_names_)} encoder features")

    check("4. Chapter 10 maps cover the contract",
          set(art.family_of) == set(art.feature_names)
          and len(art.group_of) == 143
          and len(set(art.group_of.values())) == 13,
          f"{len(set(art.family_of.values()))} families, "
          f"{len(set(art.group_of.values()))} groups over {len(art.group_of)} columns")

    valid = pd.read_parquet(fcfg.DATA_PROCESSED / "valid_v1.parquet")
    p = art.calibrator.transform(art.pipeline.predict_proba(valid[art.feature_names])[:, 1])
    flagged = int((p >= art.threshold).sum())
    check("5. threshold reproduces Chapter 9's review queue",
          flagged == 14629 and art.threshold != round(art.threshold, 6),
          f"{flagged:,} flagged at full precision {art.threshold!r}")

    probe = np.linspace(1e-6, 1 - 1e-6, 10_000)
    check("6. calibrator strictly monotone",
          bool(np.all(np.diff(art.calibrator.transform(probe)) > 0)),
          f"a = {art.calibrator.a:.6f} > 0")

    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT transactiondt, transactionamt, uid_card, uid_card_addr, uid_account"
            " FROM tx_history ORDER BY transactiondt DESC LIMIT 1"
        )
        t, amt, uc, uca, ua = cur.fetchone()
        got = aggregates_for(
            cur, {"uid_card": uc, "uid_card_addr": uca, "uid_account": ua}, t, float(amt)
        )["acct_cnt_hist"]
        cur.execute(
            "SELECT count(*) FILTER (WHERE transactiondt < %s),"
            "       count(*) FILTER (WHERE transactiondt <= %s)"
            " FROM tx_history WHERE uid_account = %s", (t, t, ua)
        )
        before, inclusive = cur.fetchone()
    check("7. a transaction never enters its own aggregates",
          got == before and inclusive > before,
          f"cnt_hist {got:.0f} == {before} strictly before, {inclusive} inclusive")

    par = C.REPORTS_ROOT / "tables" / "ch11" / "table_11_1_parity.csv"
    if par.exists():
        t8 = pd.read_csv(par).set_index("scope")
        flips = int(t8.loc["DECISION at threshold", "n_differing"])
        cells = int(t8.loc["FEATURES (all 290)", "n_differing"])
        n_cells = int(t8.loc["FEATURES (all 290)", "n_compared"])
        check("8. serving path changes no decision", flips == 0,
              f"{cells} of {n_cells:,} cells differ ({cells / n_cells:.5%}), "
              f"{flips} decisions changed")
    else:
        check("8. serving path changes no decision", False,
              "parity table missing - run make ch11-parity")

    import mlflow
    from mlflow.artifacts import download_artifacts

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    recorded = {}
    for line in (Path(download_artifacts(artifact_uri=C.MODEL_URI)) / "requirements.txt").read_text().splitlines():
        if "==" in line:
            k, v = line.split("==", 1)
            recorded[k.strip()] = v.strip()
    watch = ["lightgbm", "numpy", "scikit-learn", "scipy", "pandas", "skops", "mlflow"]
    bad = []
    for pkg in watch:
        if pkg in recorded:
            try:
                installed = md.version(pkg)
            except md.PackageNotFoundError:
                installed = "absent"
            if installed != recorded[pkg]:
                bad.append(f"{pkg} {installed} != {recorded[pkg]}")
    check("9. serving environment matches the registered model's", not bad,
          "; ".join(bad) or f"{len(watch)} packages match the registry")

    width = max(len(n) for n, _, _ in r)
    for name, ok, detail in r:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    n_fail = sum(1 for _, ok, _ in r if not ok)
    print(f"\n{f'{n_fail} of {len(r)} protocol checks FAILED' if n_fail else f'all {len(r)} protocol checks passed'}")

    out = C.REPORTS_ROOT / "tables" / "ch11"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r, columns=["check", "passed", "detail"]).to_csv(
        out / "table_11_3_protocol_checks.csv", index=False
    )
    return n_fail


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()