"""Chapter 12: backfill the monitoring window through the Chapter 11 service.

The window is the decision log Chapter 12 measures drift over. Every day is scored
through the same HTTP endpoint the Prefect flow uses, so the batch and online routes
cannot diverge. Prefect's orchestration layer is deliberately bypassed: this is a
one-off backfill of a historical window, not a recurring run. The scheduled
deployments that do use Prefect are built in Section 12.6.
"""

from __future__ import annotations

import argparse
import time

from src.flows.score_batch import API_URL, load_day, score_chunk


def score_day_direct(day: int, batch_size: int = 200, url: str = API_URL) -> dict:
    payloads = load_day.fn(day)
    flagged = 0
    for i in range(0, len(payloads), batch_size):
        flagged += score_chunk.fn(payloads[i : i + batch_size], url)
    n = len(payloads)
    return {"day": day, "n": n, "flagged": flagged, "review_rate": flagged / max(n, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=127)
    ap.add_argument("--end", type=int, default=147, help="inclusive")
    ap.add_argument("--batch-size", type=int, default=200)
    a = ap.parse_args()

    t0 = time.time()
    for d in range(a.start, a.end + 1):
        s = time.time()
        r = score_day_direct(d, batch_size=a.batch_size)
        print(
            f"DAY {d} n={r['n']} flagged={r['flagged']} rate={r['review_rate']:.4f} "
            f"secs={time.time() - s:.0f} elapsed={time.time() - t0:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
