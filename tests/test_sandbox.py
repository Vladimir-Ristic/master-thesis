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
    assert "ch8_test_" in str(FC.ARTIFACT_DIR), FC.ARTIFACT_DIR
    assert "ch8_test_" in str(FC.REPORT_TABLES), FC.REPORT_TABLES
    assert "ch8_test_" in str(FC.REPORT_FIGURES), FC.REPORT_FIGURES


def test_the_real_paths_are_still_the_default_when_unset():
    """Production behaviour must be unchanged: unset variables, original paths."""
    from src.features import config as FC

    saved = {k: os.environ.pop(k, None)
             for k in ("CH8_INTERIM_DIR", "CH8_PROCESSED_DIR",
                       "CH8_ARTIFACTS_DIR", "CH8_REPORTS_DIR")}
    try:
        fresh = importlib.reload(FC)
        assert fresh.DATA_PROCESSED == fresh.PROJECT_ROOT / "data" / "processed"
        assert fresh.DATA_INTERIM == fresh.PROJECT_ROOT / "data" / "interim"
        assert fresh.ARTIFACT_DIR   == fresh.PROJECT_ROOT / "artifacts" / "features_v1"
        assert fresh.REPORT_TABLES  == fresh.PROJECT_ROOT / "reports" / "tables"
        assert fresh.REPORT_FIGURES == fresh.PROJECT_ROOT / "reports" / "figures" / "ch8"
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        importlib.reload(FC)
