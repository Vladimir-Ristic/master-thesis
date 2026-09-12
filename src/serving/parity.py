"""Chapter 11 section 11.7: training-serving parity.

Replays real validation transactions through the serving feature path and compares
them - feature by feature, then probability by probability, then decision by
decision - against the Chapter 8 matrix and the Chapter 9 model. This is the
measurement that makes the chapter's central claim checkable rather than asserted.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.features import config as fcfg
from src.serving import config as C
from src.serving.artifacts import load_artifacts
from src.serving.db import connect
from src.serving.featurize import NAMED_FIELDS, featurize
from src.serving.schemas import RawTransaction

RTOL, ATOL = 1e-4, 1e-5


def _payloads(ids: list[int]) -> list[RawTransaction]:
    base = pd.read_parquet(fcfg.DATA_INTERIM / "base.parquet")
    vdf = pd.read_parquet(fcfg.DATA_INTERIM / "vcols.parquet")
    base = base.set_index(fcfg.KEY).loc[ids]
    vdf = vdf.set_index(fcfg.KEY).loc[ids]
    vcols = list(vdf.columns)

    out = []
    for tid in ids:
        b, v = base.loc[tid], vdf.loc[tid]
        payload = {f: (None if pd.isna(b[f]) else b[f]) for f in NAMED_FIELDS if f != fcfg.KEY}
        payload[fcfg.KEY] = int(tid)
        payload["v"] = {c: (None if pd.isna(v[c]) else float(v[c])) for c in vcols}
        out.append(RawTransaction(**payload))
    return out


def run(n: int = 5000, chunk: int = 500, seed: int = 42) -> pd.DataFrame:
    art = load_artifacts()
    valid = pd.read_parquet(fcfg.DATA_PROCESSED / "valid_v1.parquet")
    sample = valid.sample(n, random_state=seed).sort_values("transactionid")
    ids = [int(i) for i in sample.transactionid]

    served = []
    with connect() as conn, conn.cursor() as cur:
        for i in range(0, len(ids), chunk):
            batch = ids[i : i + chunk]
            served.append(featurize(_payloads(batch), art, cur)[0])
            print(f"  featurized {min(i + chunk, len(ids)):>6,}/{len(ids):,}", flush=True)
    got = pd.concat(served).astype("float32")

    exp = sample.set_index("transactionid").reindex(got.index)[art.feature_names].astype("float32")

    rows, cells_bad, rows_bad = [], 0, np.zeros(len(got), dtype=bool)
    for c in art.feature_names:
        a, b = got[c].to_numpy(), exp[c].to_numpy()
        ok = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        if not ok.all():
            d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            rows.append({"scope": c, "n_compared": len(a),
                         "n_differing": int((~ok).sum()), "max_abs_diff": float(d.max())})
            cells_bad += int((~ok).sum())
            rows_bad |= ~ok

    p_got = art.calibrator.transform(art.pipeline.predict_proba(got[art.feature_names])[:, 1])
    p_exp = art.calibrator.transform(art.pipeline.predict_proba(exp[art.feature_names])[:, 1])
    d_got, d_exp = p_got >= art.threshold, p_exp >= art.threshold
    flips = int((d_got != d_exp).sum())

    rows += [
        {"scope": "FEATURES (all 290)", "n_compared": len(got) * 290,
         "n_differing": cells_bad, "max_abs_diff": np.nan},
        {"scope": "ROWS with any differing feature", "n_compared": len(got),
         "n_differing": int(rows_bad.sum()), "max_abs_diff": np.nan},
        {"scope": "calibrated probability", "n_compared": len(got),
         "n_differing": int((~np.isclose(p_got, p_exp, rtol=1e-6, atol=1e-9)).sum()),
         "max_abs_diff": float(np.abs(p_got - p_exp).max())},
        {"scope": "DECISION at threshold", "n_compared": len(got),
         "n_differing": flips, "max_abs_diff": np.nan},
    ]
    table = pd.DataFrame(rows)

    out_dir = C.REPORTS_ROOT / "tables" / "ch11"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table_11_1_parity.csv", index=False)

    print("\n" + table.to_string(index=False))
    print(f"\nwrote {out_dir / 'table_11_1_parity.csv'}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()
    run(args.n, args.chunk)


if __name__ == "__main__":
    main()