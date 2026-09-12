"""Chapter 11 section 11.7: latency and throughput.

Payloads are built once and held in memory so the numbers measure the service and
not the client. Latency here is wall clock on one machine under stated conditions;
unlike PR-AUC it does not reproduce exactly, and the chapter reports it that way.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
import json

import httpx
import pandas as pd

from src.features import config as fcfg
from src.serving import config as C
from src.serving.featurize import NAMED_FIELDS


def _clean(x):
    if pd.isna(x):
        return None
    return str(x) if isinstance(x, str) else float(x)


def build_payloads(n: int, seed: int = 7) -> list[dict]:
    valid = pd.read_parquet(fcfg.DATA_PROCESSED / "valid_v1.parquet")
    ids = [int(i) for i in valid.sample(n, random_state=seed).transactionid]
    base = pd.read_parquet(fcfg.DATA_INTERIM / "base.parquet").set_index(fcfg.KEY).loc[ids]
    vdf = pd.read_parquet(fcfg.DATA_INTERIM / "vcols.parquet").set_index(fcfg.KEY).loc[ids]
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


def _pct(xs: list[float], q: float) -> float:
    return statistics.quantiles(xs, n=100)[int(q) - 1] if len(xs) > 2 else max(xs)


def run(url: str, n_single: int, batch_sizes: list[int], n_payloads: int) -> pd.DataFrame:
    payloads = build_payloads(n_payloads)
    client = httpx.Client(timeout=300)

    client.post(f"{url}/score", json=payloads[0])  # warm up

    single: dict[str, list[float]] = {"review": [], "accept": []}
    for p in payloads[:n_single]:
        t0 = time.perf_counter()
        r = client.post(f"{url}/score", json=p)
        dt = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        single[r.json()["decision"]].append(dt)
        print(f"  single {len(single['review']) + len(single['accept']):>4}/{n_single}", end="\r")

    rows = []
    for label, xs in (("single, review", single["review"]), ("single, accept", single["accept"])):
        if xs:
            rows.append({
                "mode": label, "batch_size": 1, "n_requests": len(xs), "n_transactions": len(xs),
                "p50_ms_per_tx": statistics.median(xs),
                "p95_ms_per_tx": _pct(xs, 95), "p99_ms_per_tx": _pct(xs, 99),
                "throughput_tx_per_s": 1000.0 / statistics.median(xs),
            })

    for size in batch_sizes:
        per_tx = []
        for i in range(0, min(len(payloads), size * 5), size):
            batch = payloads[i : i + size]
            if len(batch) < size:
                break
            t0 = time.perf_counter()
            r = client.post(f"{url}/score/batch", json=batch)
            dt = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            per_tx.append(dt / size)
        if per_tx:
            rows.append({
                "mode": "batch", "batch_size": size, "n_requests": len(per_tx),
                "n_transactions": len(per_tx) * size,
                "p50_ms_per_tx": statistics.median(per_tx),
                "p95_ms_per_tx": _pct(per_tx, 95), "p99_ms_per_tx": _pct(per_tx, 99),
                "throughput_tx_per_s": 1000.0 / statistics.median(per_tx),
            })

    table = pd.DataFrame(rows)
    out_dir = C.REPORTS_ROOT / "tables" / "ch11"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table_11_2_latency.csv", index=False)

    health = client.get(f"{url}/health").json()
    conditions = {
        "server": health,
        "client_platform": platform.platform(),
        "client_cpu_count": os.cpu_count(),
        "n_single": n_single,
        "batch_sizes": batch_sizes,
    }
    (out_dir / "bench_conditions.json").write_text(json.dumps(conditions, indent=2))

    print("\nserver conditions:", json.dumps(health, indent=2))
    print("\n" + table.to_string(index=False))
    print(f"\nwrote {out_dir / 'table_11_2_latency.csv'} and bench_conditions.json")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n-single", type=int, default=60)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[10, 50, 200])
    ap.add_argument("--n-payloads", type=int, default=1000)
    args = ap.parse_args()
    run(args.url, args.n_single, args.batch_sizes, args.n_payloads)


if __name__ == "__main__":
    main()