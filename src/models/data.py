"""Partition loading and the feature contract.

Chapter 8 produced three parquet partitions with an audited, identical feature
contract. Chapter 9 must not re-engineer features, so this module only reads,
separates target/key/day from the feature matrix, and asserts the contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Partition:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    day: np.ndarray
    amount: np.ndarray
    key: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<Partition {self.name}: {self.n:,} rows x {self.X.shape[1]} features, "
            f"days {self.day.min()}-{self.day.max()}, "
            f"prevalence {self.prevalence:.4%}>"
        )


def _resolve_day_column(columns) -> str:
    for candidate in C.DAY_COL_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(
        "No day index column found. Expected one of "
        f"{C.DAY_COL_CANDIDATES}. Chapter 9 needs an integer day index to build "
        "purged folds; add it to the Chapter 8 output or extend "
        "config.DAY_COL_CANDIDATES."
    )


def load_partition(name: str, path=None) -> Partition:
    """Read one partition into a Partition object.

    Reading the test partition raises unless config.ALLOW_TEST_READ is set.
    That guard is the code-level form of the Chapter 8 commitment that the test
    partition is read once, in Chapter 13.
    """
    if name == "test" and not C.ALLOW_TEST_READ:
        raise PermissionError(
            "The test partition is reserved for Chapter 13. Set "
            "config.ALLOW_TEST_READ = True deliberately, and record why."
        )

    path = path or (C.DATA_PROCESSED / C.PARTITION_FILES[name])
    frame = pd.read_parquet(path)
    part = from_frame(name, frame)
    if C.STRICT_SHAPE_CHECK and _is_canonical_directory():
        assert_matches_chapter8(part, path)
    return part


def _is_canonical_directory() -> bool:
    """Only the real feature directory is held to Table 8.1.

    The synthetic clone is deliberately smaller, so the check is skipped
    whenever the partitions are being read from anywhere else — which covers
    the test suite, the smoke target, and any ad-hoc experiment on a copy.
    """
    return Path(C.DATA_PROCESSED).resolve() == (C.ROOT / "data" / "processed").resolve()


def assert_matches_chapter8(part: Partition, path=None) -> None:
    """Refuse a partition that is not the one Chapter 8 recorded.

    This exists because the failure mode is silent. A leftover synthetic or
    partially built parquet in `data/processed/` loads without complaint, folds
    correctly, trains happily and produces a table of numbers that mean
    nothing. Every figure in Chapter 9 would then be unattributable to the
    dataset the thesis describes. Table 8.1 is the contract; this is the check.
    """
    expected = C.EXPECTED_PARTITIONS.get(part.name)
    if expected is None:
        return

    problems = []
    if part.n != expected["rows"]:
        problems.append(f"{part.n:,} rows, Table 8.1 says {expected['rows']:,}")
    observed_days = (int(part.day.min()), int(part.day.max()))
    if observed_days != tuple(expected["days"]):
        problems.append(f"days {observed_days[0]}-{observed_days[1]}, "
                        f"Table 8.1 says {expected['days'][0]}-{expected['days'][1]}")
    if abs(part.prevalence - expected["prevalence"]) > 1e-3:
        problems.append(f"prevalence {part.prevalence:.4%}, "
                        f"Table 8.1 says {expected['prevalence']:.4%}")

    if problems:
        raise AssertionError(
            f"The {part.name} partition at {path or C.DATA_PROCESSED} does not match "
            f"Chapter 8: " + "; ".join(problems) + ".\n"
            "This is almost always a leftover synthetic or partially built parquet "
            "sitting in data/processed/. Re-run the Chapter 8 build (`make ch8-build`) "
            "and confirm Table 8.1 before continuing.\n"
            "If Chapter 8 was deliberately rebuilt with different boundaries, update "
            "config.EXPECTED_PARTITIONS and Table 8.1 together, or set "
            "CH9_STRICT_SHAPE=0 to bypass this check for one run."
        )


def from_frame(name: str, frame: pd.DataFrame) -> Partition:
    day_col = _resolve_day_column(frame.columns)

    if C.TARGET not in frame.columns:
        raise KeyError(f"Target column {C.TARGET!r} missing from {name} partition.")
    if C.AMOUNT not in frame.columns:
        raise KeyError(
            f"{C.AMOUNT!r} missing from {name} partition. Chapter 8, Section 8.10 "
            "commits it unmodified as the basis of the cost function."
        )

    y = frame[C.TARGET].to_numpy(dtype=np.int8)
    day = frame[day_col].to_numpy(dtype=np.int32)
    amount = frame[C.AMOUNT].to_numpy(dtype=np.float64)
    key = (
        frame[C.KEY].to_numpy()
        if C.KEY in frame.columns
        else np.arange(len(frame), dtype=np.int64)
    )

    drop = [C.TARGET]
    if C.KEY in frame.columns:
        drop.append(C.KEY)
    if C.DROP_ABSOLUTE_TIME_FROM_FEATURES:
        drop += [c for c in C.DAY_COL_CANDIDATES if c in frame.columns]

    X = frame.drop(columns=list(dict.fromkeys(drop)))
    X = X.astype(np.float32, copy=False)

    if not np.isfinite(amount).all() or (amount < 0).any():
        raise ValueError("TransactionAmt contains non-finite or negative values.")

    return Partition(name=name, X=X, y=y, day=day, amount=amount, key=key)


def assert_same_contract(a: Partition, b: Partition) -> None:
    """The feature contract must be identical, in the same order."""
    if list(a.X.columns) != list(b.X.columns):
        only_a = sorted(set(a.X.columns) - set(b.X.columns))
        only_b = sorted(set(b.X.columns) - set(a.X.columns))
        raise AssertionError(
            f"Feature contract differs between {a.name} and {b.name}. "
            f"Only in {a.name}: {only_a[:10]}. Only in {b.name}: {only_b[:10]}."
        )


def assert_disjoint(a: Partition, b: Partition) -> None:
    if a.key.dtype != object and b.key.dtype != object:
        overlap = np.intersect1d(a.key, b.key)
        if overlap.size:
            raise AssertionError(
                f"{overlap.size} transactions appear in both {a.name} and {b.name}."
            )
    if a.day.max() >= b.day.min():
        raise AssertionError(
            f"{a.name} (to day {a.day.max()}) is not strictly before "
            f"{b.name} (from day {b.day.min()})."
        )


def subsample(part: Partition, n: int, seed: int = C.SEED) -> Partition:
    """Contiguous tail subsample, used only by the smoke profile.

    The tail is taken rather than a random sample so the day ordering the fold
    generator relies on is preserved.
    """
    if n >= part.n:
        return part
    idx = np.arange(part.n - n, part.n)
    return Partition(
        name=f"{part.name}[tail{n}]",
        X=part.X.iloc[idx].reset_index(drop=True),
        y=part.y[idx],
        day=part.day[idx],
        amount=part.amount[idx],
        key=part.key[idx],
    )
