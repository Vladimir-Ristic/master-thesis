"""Generate a synthetic clone of the Chapter 8 feature matrix.

The clone reproduces the *structure* the Chapter 9 code depends on — three
chronological partitions, an integer day index, an unmodified amount column,
heavy and blockwise missingness, a ~3.5% positive rate and a weak nonlinear
signal — at a size that runs in seconds. It is not a simulation of the IEEE-CIS
data and no result from it belongs in the thesis.

    python -m scripts.make_synthetic_features --out data/processed_synth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROWS = {"train": 30_000, "valid": 6_000, "test": 6_000}
DAY_RANGES = {"train": (1, 119), "valid": (127, 154), "test": (155, 182)}
N_NAMED = 24
N_V = 36


def _build(name: str, n: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    d0, d1 = DAY_RANGES[name]
    day = np.sort(rng.integers(d0, d1 + 1, size=n)).astype(np.int32)

    named = rng.standard_normal((n, N_NAMED)).astype(np.float32)
    v = rng.standard_normal((n, N_V)).astype(np.float32)

    # A weak nonlinear signal with an interaction, so a tree can beat a line.
    logit = (
        1.10 * named[:, 0]
        + 0.85 * np.abs(named[:, 1])
        + 0.70 * (named[:, 2] > 0.4)
        + 0.55 * named[:, 3] * named[:, 4]
        + 0.45 * v[:, 0]
        + 0.30 * (day - day.mean()) / max(day.std(), 1.0)
    )
    logit = logit - 5.20  # tunes prevalence to roughly 3.5%
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.int8)

    # Amount: lognormal, with fraud shifted slightly higher, as in Section 7.4.
    amount = np.exp(rng.normal(4.1 + 0.18 * y, 1.05)).round(3).astype(np.float64)

    frame = pd.DataFrame(
        {
            "transactionid": np.arange(start_id, start_id + n, dtype=np.int64),
            "isfraud": y,
            "tx_day": day,
            "transactionamt": amount,
        }
    )
    for j in range(N_NAMED):
        frame[f"named_{j:02d}"] = named[:, j]
    for j in range(N_V):
        frame[f"v{j+1}"] = v[:, j]

    # Blockwise missingness: three groups sharing an identical pattern, plus
    # scattered missingness in the named block.
    for block, cols in enumerate(np.array_split([f"v{j+1}" for j in range(N_V)], 3)):
        mask = rng.random(n) < (0.20 + 0.15 * block)
        frame.loc[mask, list(cols)] = np.nan
    for j in range(4, N_NAMED):
        mask = rng.random(n) < 0.12
        frame.loc[mask, f"named_{j:02d}"] = np.nan

    # Entity-history features are undefined for an account's first transaction.
    frame["acct_cnt_7d"] = np.where(rng.random(n) < 0.18, np.nan, rng.poisson(3, n))
    frame["acct_amt_std_hist"] = np.where(
        rng.random(n) < 0.35, np.nan, np.abs(rng.normal(40, 25, n))
    )
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed_synth")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply row counts (0.2 for a very fast fixture)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    start = 1
    for name in ("train", "valid", "test"):
        n = max(int(DEFAULT_ROWS[name] * args.scale), 2_000)
        frame = _build(name, n, rng, start)
        start += n
        path = out / f"{name}_v1.parquet"
        frame.to_parquet(path, index=False)
        print(f"{name}: {len(frame):,} rows x {frame.shape[1]} cols, "
              f"days {frame.tx_day.min()}-{frame.tx_day.max()}, "
              f"prevalence {frame.isfraud.mean():.3%} -> {path}")


if __name__ == "__main__":
    main()
