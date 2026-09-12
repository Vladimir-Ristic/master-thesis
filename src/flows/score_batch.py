"""Chapter 11: the batch scoring flow - the offline half of the architecture.

It scores a day of transactions through the same HTTP path a caller uses, so the
batch and online routes cannot diverge, and asks the service to store the served
feature row, which is what Chapter 12 reads to measure drift against a recorded
reference rather than a remembered one.

Scheduling, drift detection and retraining belong to Chapter 12. This flow is
deliberately one task chain with no schedule attached.
"""

from __future__ import annotations

import argparse

import httpx
import pandas as pd
from prefect import flow, get_run_logger, task

from src.features import config as fcfg
from src.serving.bench import _clean
from src.serving.featurize import NAMED_FIELDS

API_URL = "http://127.0.0.1:8000"


@task(retries=2, retry_delay_seconds=10)
def load_day(day: int) -> list[dict]:
    base = pd.read_parquet(fcfg.DATA_INTERIM / "base.parquet")
    base = base.loc[(base[fcfg.TIME] // fcfg.SECONDS_PER_DAY) == day]
    ids = [int(i) for i in base[fcfg.KEY]]
    vdf = pd.read_parquet(fcfg.DATA_INTERIM / "vcols.parquet").set_index(fcfg.KEY).loc[ids]
    base = base.set_index(fcfg.KEY)
    vcols = list(vdf.columns)

    out = []
    for tid in ids:
        b, v = base.loc[tid], vdf.loc[tid]
        p = {f: _clean(b[f]) for f in NAMED_FIELDS if f != fcfg.KEY}
        p[fcfg.KEY] = int(tid)
        p["transactiondt"] = int(b[fcfg.TIME])
        p["v"] = {c: _clean(v[c]) for c in vcols}
        out.append(p)
    return out


@task(retries=3, retry_delay_seconds=15)
def score_chunk(payloads: list[dict], url: str) -> int:
    r = httpx.post(
        f"{url}/score/batch", json=payloads,
        params={"store_features": True}, timeout=900,
    )
    r.raise_for_status()
    return sum(1 for x in r.json() if x["decision"] == "review")


@flow(name="score-batch")
def score_day(day: int = 127, batch_size: int = 200, url: str = API_URL) -> dict:
    log = get_run_logger()
    payloads = load_day(day)
    log.info("day %d: %d transactions", day, len(payloads))

    flagged = 0
    for i in range(0, len(payloads), batch_size):
        flagged += score_chunk(payloads[i : i + batch_size], url)
        log.info("  scored %d/%d", min(i + batch_size, len(payloads)), len(payloads))

    rate = flagged / max(len(payloads), 1)
    log.info("day %d complete: %d scored, %d flagged (%.4f)", day, len(payloads), flagged, rate)
    return {"day": day, "n": len(payloads), "flagged": flagged, "review_rate": rate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, default=127)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--url", default=API_URL)
    args = ap.parse_args()
    score_day(args.day, args.batch_size, args.url)


if __name__ == "__main__":
    main()