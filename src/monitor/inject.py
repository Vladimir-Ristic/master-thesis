"""Chapter 12: the injected drift day.

The demonstration is worth nothing if the drifted rows are written straight into
the decision log. The perturbation is applied to the raw payload and scored through
the running service, so the whole chain - featurisation from history, calibration,
threshold, reason codes, persistence - executes exactly as it does for real traffic.

Days 148 and 149 are used because they lie inside the validation partition but
outside the monitored window, so the injected rows cannot contaminate the baseline
the monitor established. Two consecutive days are injected because the retraining
rule deliberately requires persistence.
"""

from __future__ import annotations

import argparse

from src.flows.score_batch import API_URL, load_day, score_chunk


def perturb(payloads: list[dict], factor: float) -> list[dict]:
    """A shift in the transacted amount, the second-ranked feature by SHAP."""
    out = []
    for p in payloads:
        q = dict(p)
        amt = q.get("transactionamt")
        if amt is not None:
            q["transactionamt"] = float(amt) * factor
        out.append(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=[148, 149])
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=200)
    a = ap.parse_args()

    for d in a.days:
        payloads = perturb(load_day.fn(d), a.factor)
        flagged = 0
        for i in range(0, len(payloads), a.batch_size):
            flagged += score_chunk.fn(payloads[i: i + a.batch_size], API_URL)
        n = len(payloads)
        print(f"INJECTED day {d} factor={a.factor}: n={n} flagged={flagged} "
              f"rate={flagged / max(n, 1):.4f}", flush=True)


if __name__ == "__main__":
    main()