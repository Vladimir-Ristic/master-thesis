"""
Generate a small synthetic clone of the IEEE-CIS Fraud Detection schema.

This is a TEST FIXTURE ONLY. It reproduces the *structure* of the real data
(column names, dtypes, entity behaviour, D1 semantics, missingness rates,
temporal fraud drift) at ~1/20th the scale so the Chapter 8 pipeline can be
executed end to end quickly. It is not used for any thesis result.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260906)

N_ROWS = 30_000
N_DAYS = 183
N_V = 60          # real data has 339; 60 is enough to exercise the reduction logic
N_ACCOUNTS = 2_500
OUT_DIR = "data/raw_synth"


def _daily_volume() -> np.ndarray:
    """Front-loaded volume profile, mirroring Figure 7.3."""
    day = np.arange(N_DAYS)
    base = np.where(day < 25, 1.8 - 0.02 * day, 1.0)
    base = base + 0.12 * np.sin(2 * np.pi * day / 7)      # weekly seasonality
    return np.clip(base, 0.4, None)


def build_transactions() -> pd.DataFrame:
    weights = _daily_volume()
    weights = weights / weights.sum()
    day = RNG.choice(np.arange(N_DAYS), size=N_ROWS, p=weights)
    sec_in_day = RNG.integers(0, 86_400, size=N_ROWS)
    transactiondt = day.astype(np.int64) * 86_400 + sec_in_day
    order = np.argsort(transactiondt, kind="mergesort")
    transactiondt = transactiondt[order]
    day = day[order]

    # --- pseudo-accounts: each account has a stable card1/addr1 and a
    # registration day, so that (day - D1) recovers the registration day.
    acct = RNG.integers(0, N_ACCOUNTS, size=N_ROWS)
    acct_card1 = RNG.integers(1000, 19000, size=N_ACCOUNTS)
    acct_addr1 = RNG.integers(100, 540, size=N_ACCOUNTS)
    acct_reg_day = RNG.integers(-400, 1, size=N_ACCOUNTS)      # registered before window
    acct_amt_mu = RNG.lognormal(mean=4.1, sigma=0.55, size=N_ACCOUNTS)

    card1 = acct_card1[acct]
    addr1 = acct_addr1[acct]
    d1 = day - acct_reg_day[acct]                              # days since first tx

    productcd = RNG.choice(
        ["W", "C", "R", "H", "S"], size=N_ROWS, p=[0.745, 0.116, 0.064, 0.056, 0.019]
    )
    card4 = RNG.choice(
        ["visa", "mastercard", "american express", "discover"],
        size=N_ROWS, p=[0.652, 0.320, 0.017, 0.011],
    )
    card6 = RNG.choice(["debit", "credit"], size=N_ROWS, p=[0.747, 0.253])

    amt = np.round(acct_amt_mu[acct] * RNG.lognormal(0.0, 0.7, size=N_ROWS), 3)
    amt = np.clip(amt, 0.251, 31_937.391)

    # --- fraud generation: drifting base rate + subgroup effects
    drift = 0.02 + 0.03 * (day / N_DAYS) + 0.01 * np.sin(2 * np.pi * day / 60)
    logit = np.log(drift / (1 - drift))
    logit += np.where(productcd == "C", 1.5, 0.0)
    logit += np.where(productcd == "W", -0.4, 0.0)
    logit += np.where(card6 == "credit", 0.8, 0.0)
    logit += 0.25 * (np.log(amt) - np.log(amt).mean())
    p = 1 / (1 + np.exp(-logit))
    isfraud = (RNG.random(N_ROWS) < p).astype(np.int8)

    df = pd.DataFrame(
        {
            "transactionid": np.arange(2_987_000, 2_987_000 + N_ROWS, dtype=np.int64),
            "isfraud": isfraud,
            "transactiondt": transactiondt,
            "transactionamt": amt,
            "productcd": productcd,
            "card1": card1,
            "card2": RNG.integers(100, 600, size=N_ROWS).astype(float),
            "card3": RNG.choice([150.0, 185.0], size=N_ROWS, p=[0.88, 0.12]),
            "card4": card4,
            "card5": RNG.integers(100, 240, size=N_ROWS).astype(float),
            "card6": card6,
            "addr1": addr1.astype(float),
            "addr2": RNG.choice([87.0, 60.0], size=N_ROWS, p=[0.95, 0.05]),
            "dist1": RNG.exponential(50, size=N_ROWS).round(0),
            "dist2": RNG.exponential(120, size=N_ROWS).round(0),
            "p_emaildomain": RNG.choice(
                ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com",
                 "aol.com", "outlook.com", "gmail", "yahoo.com.mx"],
                size=N_ROWS, p=[0.40, 0.20, 0.12, 0.10, 0.06, 0.06, 0.03, 0.03],
            ),
            "r_emaildomain": RNG.choice(
                ["gmail.com", "hotmail.com", "anonymous.com", "yahoo.com"],
                size=N_ROWS, p=[0.45, 0.25, 0.20, 0.10],
            ),
        }
    )

    # C1-C14 counting features: complete (0% missing), correlated with entity activity
    for i in range(1, 15):
        df[f"c{i}"] = RNG.poisson(1.5 + 0.4 * (i % 4), size=N_ROWS).astype(float)

    # D1-D15 timedeltas
    df["d1"] = d1.astype(float)
    for i in range(2, 16):
        col = np.where(RNG.random(N_ROWS) < 0.55, np.nan,
                       RNG.integers(0, 640, size=N_ROWS).astype(float))
        df[f"d{i}"] = col

    # M1-M9 binary match flags
    for i in range(1, 10):
        vals = RNG.choice(["T", "F"], size=N_ROWS, p=[0.7, 0.3]).astype(object)
        vals[RNG.random(N_ROWS) < 0.50] = np.nan
        df[f"m{i}"] = vals
    df["m4"] = RNG.choice(["M0", "M1", "M2"], size=N_ROWS)   # m4 is 3-level in real data

    # V-columns, generated in correlated blocks with shared missingness patterns
    n_blocks = 6
    per_block = N_V // n_blocks
    for b in range(n_blocks):
        miss_rate = [0.0, 0.13, 0.28, 0.47, 0.62, 0.87][b]
        block_mask = RNG.random(N_ROWS) < miss_rate
        latent = RNG.normal(size=N_ROWS)
        for j in range(per_block):
            idx = b * per_block + j + 1
            noise = RNG.normal(scale=0.15 if j % 3 else 0.9, size=N_ROWS)
            vals = (latent + noise).astype(np.float32)
            vals = np.where(block_mask, np.nan, vals)
            df[f"v{idx}"] = vals

    # structural missingness in named fields
    df.loc[RNG.random(N_ROWS) < 0.76, "r_emaildomain"] = np.nan
    df.loc[RNG.random(N_ROWS) < 0.60, "dist1"] = np.nan
    df.loc[RNG.random(N_ROWS) < 0.94, "dist2"] = np.nan
    df.loc[RNG.random(N_ROWS) < 0.003, "card4"] = np.nan
    df.loc[RNG.random(N_ROWS) < 0.003, "card6"] = np.nan
    df.loc[RNG.random(N_ROWS) < 0.16, "p_emaildomain"] = np.nan
    df.loc[RNG.random(N_ROWS) < 0.11, "addr1"] = np.nan

    return df


def build_identity(tx: pd.DataFrame) -> pd.DataFrame:
    # identity coverage is risk-correlated (7.85% vs 2.09% fraud rate in Ch. 7)
    base = 0.19 + 0.45 * tx["isfraud"].to_numpy()
    linked = RNG.random(len(tx)) < base
    ids = tx.loc[linked, "transactionid"].to_numpy()
    n = len(ids)

    idn = pd.DataFrame({"transactionid": ids})
    for i in range(1, 12):                              # id_01..id_11 numeric
        col = RNG.normal(size=n).round(3)
        idn[f"id_{i:02d}"] = np.where(RNG.random(n) < (0.05 if i < 3 else 0.85),
                                      np.nan, col)
    for i in range(12, 39):                             # id_12..id_38 categorical
        col = RNG.choice(["Found", "NotFound", "New"], size=n).astype(object)
        col[RNG.random(n) < (0.30 if i < 20 else 0.75)] = np.nan
        idn[f"id_{i:02d}"] = col

    dev = RNG.choice(["mobile", "desktop"], size=n, p=[0.395, 0.605]).astype(object)
    dev[RNG.random(n) < 0.024] = np.nan
    idn["devicetype"] = dev
    idn["deviceinfo"] = RNG.choice(
        ["Windows", "iOS Device", "MacOS", "Trident/7.0", "SAMSUNG SM-G892A"], size=n
    )
    return idn


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    tx = build_transactions()
    idn = build_identity(tx)
    tx.to_csv(f"{OUT_DIR}/train_transaction.csv", index=False)
    idn.to_csv(f"{OUT_DIR}/train_identity.csv", index=False)
    print(f"transactions: {tx.shape}, identity: {idn.shape}")
    print(f"fraud rate: {tx['isfraud'].mean():.4%}")
    print(f"identity coverage: {len(idn) / len(tx):.2%}")
    print(f"day span: {tx['transactiondt'].max() / 86400:.1f}")
