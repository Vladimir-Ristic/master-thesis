"""Chapter 13: the read-once ledger.

Chapters 8 through 12 each carried a protocol check asserting the test partition
was unread. This module is what replaces those checks once it is read: a ledger
recording when the partition was opened, under which specification, against which
digests, and that it was opened exactly once.

The seal is not a file permission. Nothing here prevents a determined second read -
the point is that a second read cannot happen quietly. open_seal() refuses to run
twice, and assert_opened_once() fails loudly if the ledger says otherwise.

    make ch13-open     opens the seal, once
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

from src.eval import config as C


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify_digests() -> dict[str, str]:
    """Every sealed source must match the digest recorded at Step 0 preflight."""
    root = C.S.PROJECT_ROOT if hasattr(C, "S") else C.TEST_PARQUET.parents[2]
    seen, bad = {}, []
    for rel, want in C.EXPECTED_MD5.items():
        got = _md5(root / rel)
        seen[rel] = got
        if got != want:
            bad.append(f"  {rel}\n    expected {want}\n    found    {got}")
    if bad:
        raise RuntimeError("sealed sources have changed:\n" + "\n".join(bad))
    return seen


def spec_md5() -> str:
    if not C.SPEC_PATH.exists():
        raise RuntimeError(f"no specification at {C.SPEC_PATH}; run make ch13-spec")
    return hashlib.md5(C.SPEC_PATH.read_bytes()).hexdigest()


def is_open() -> bool:
    return C.SEAL_PATH.exists()


def ledger() -> dict:
    if not is_open():
        raise RuntimeError(f"seal is not open; no ledger at {C.SEAL_PATH}")
    return json.loads(C.SEAL_PATH.read_text())


def assert_sealed() -> None:
    """Guard for anything that must run BEFORE the partition is opened."""
    if is_open():
        raise RuntimeError(
            f"REFUSED: the seal is already open ({C.SEAL_PATH}). "
            f"This step must precede the first read."
        )


def assert_opened_once() -> dict:
    """Guard for anything that reads the test partition or its results."""
    led = ledger()
    if led.get("open_count") != 1:
        raise RuntimeError(
            f"seal ledger records open_count={led.get('open_count')}, expected 1"
        )
    current = spec_md5()
    if current != led["spec_md5"]:
        raise RuntimeError(
            "the measurement specification changed after the seal was opened:\n"
            f"  at open  {led['spec_md5']}\n  now      {current}"
        )
    verify_digests()
    return led


def open_seal() -> dict:
    assert_sealed()
    C.ensure_dirs()
    led = {
        "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "open_count": 1,
        "chapter": 13,
        "spec_path": str(C.SPEC_PATH),
        "spec_md5": spec_md5(),
        "digests_at_open": verify_digests(),
        "partition": {
            "day_min": C.TEST_DAY_MIN,
            "day_max": C.TEST_DAY_MAX,
            "expected_rows": C.EXPECTED_TEST_ROWS,
        },
        "pre_run_decision_log_rows": C.PRE_RUN_LOG_ROWS,
        "statement": (
            "The test partition of the IEEE-CIS dataset was read for the first "
            "time at the timestamp above, under the specification whose digest "
            "is recorded here, and was not read before it."
        ),
    }
    C.SEAL_PATH.write_text(json.dumps(led, indent=2, sort_keys=True) + "\n")
    return led


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.open:
        led = open_seal()
        print(json.dumps(led, indent=2, sort_keys=True))
        print("\nthe seal is open. the specification is now frozen.")
    elif a.status:
        print("open" if is_open() else "sealed")
        if is_open():
            print(json.dumps(ledger(), indent=2, sort_keys=True))
    else:
        ap.error("one of --open or --status")


if __name__ == "__main__":
    main()