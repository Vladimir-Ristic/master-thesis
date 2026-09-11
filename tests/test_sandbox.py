"""The sandbox is a safety net, so it is itself tested.

If the first test fails, the Chapter 8 suite is writing to the real feature layer
again and `make ch8-test` will destroy `data/processed/`. Re-run
`python fix_ch8_test_sandbox.py` before running anything else.
"""

import importlib
import os


def test_the_suite_never_writes_to_the_real_feature_layer():
    from src.features import config as FC

    assert "ch8_test_" in str(FC.DATA_PROCESSED), FC.DATA_PROCESSED
    assert "ch8_test_" in str(FC.DATA_INTERIM), FC.DATA_INTERIM


def test_the_real_paths_are_still_the_default_when_unset():
    """Production behaviour must be unchanged: unset variables, original paths."""
    from src.features import config as FC

    saved = {k: os.environ.pop(k, None)
             for k in ("CH8_INTERIM_DIR", "CH8_PROCESSED_DIR")}
    try:
        fresh = importlib.reload(FC)
        assert fresh.DATA_PROCESSED == fresh.PROJECT_ROOT / "data" / "processed"
        assert fresh.DATA_INTERIM == fresh.PROJECT_ROOT / "data" / "interim"
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        importlib.reload(FC)
