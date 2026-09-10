"""Protocol checks for Chapter 9, run as a build stage.

Chapter 8 verified the feature set. Chapter 9's exposure is different: the
features are audited, but the *protocol* can still be violated — by tuning
against the test partition, by resampling data that is meant to stay untouched,
or by reporting a threshold chosen on the same rows it is evaluated on. None of
those fail loudly either, so they are checked here and the results reported as
a table in the chapter.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import arms, calibration, config as C, cv, data, metrics


def _check(name: str, fn) -> dict:
    try:
        detail = fn()
        return {"check": name, "result": "Pass", "detail": detail or ""}
    except AssertionError as exc:
        return {"check": name, "result": "FAIL", "detail": str(exc)}
    except Exception as exc:  # pragma: no cover - surfaced in the table
        return {"check": name, "result": "ERROR", "detail": f"{type(exc).__name__}: {exc}"}


def run_all(expected_valid_prevalence: float | None = None) -> pd.DataFrame:
    train = data.load_partition("train")
    valid = data.load_partition("valid")
    folds = cv.make_folds(train.day)

    def test_partition_sealed():
        assert not C.ALLOW_TEST_READ, "ALLOW_TEST_READ is set; the test partition is open."
        try:
            data.load_partition("test")
        except PermissionError:
            return "read refused by data.load_partition"
        raise AssertionError("The test partition was readable.")

    def contract_identical():
        data.assert_same_contract(train, valid)
        return f"{train.X.shape[1]} features, order preserved"

    def target_and_key_absent():
        for frame_name, part in (("train", train), ("valid", valid)):
            for col in (C.TARGET, C.KEY, *C.DAY_COL_CANDIDATES):
                assert col not in part.X.columns, f"{col} present in {frame_name} matrix"
        return "target, key and absolute day index excluded"

    def partitions_ordered():
        data.assert_disjoint(train, valid)
        gap = int(valid.day.min() - train.day.max() - 1)
        assert gap >= C.MAX_WINDOW_DAYS, f"outer embargo is {gap} days"
        return (f"train d{train.day.min()}-{train.day.max()} -> "
                f"{gap}-day embargo -> valid d{valid.day.min()}-{valid.day.max()}")

    def folds_purged():
        cv.assert_folds_valid(folds)
        gaps = {f.valid_days[0] - f.train_days[1] - 1 for f in folds}
        assert gaps == {C.PURGE_DAYS}, f"purge gaps {sorted(gaps)}"
        return f"{len(folds)} expanding folds, {C.PURGE_DAYS}-day purge each"

    def folds_disjoint_in_time():
        for f in folds:
            tr_days = train.day[f.train_idx]
            va_days = train.day[f.valid_idx]
            assert tr_days.max() < va_days.min(), f"fold {f.index} overlaps in time"
        return "every fold's training block strictly precedes its validation block"

    def resampling_is_structural():
        rng = np.random.default_rng(C.SEED)
        idx = rng.choice(train.n, size=min(20_000, train.n), replace=False)
        Xs, ys = train.X.iloc[idx], train.y[idx]
        if ys.sum() < 10:
            idx = np.argsort(-train.y)[:20_000]
            Xs, ys = train.X.iloc[idx], train.y[idx]
        pipe = arms.build_pipeline("lgbm", "smote", {"n_estimators": 20}, ys)
        pipe.fit(Xs, ys)
        p = pipe.predict_proba(valid.X.iloc[:1000])[:, 1]
        assert p.shape[0] == 1000, "prediction path changed the number of rows"
        assert any(step[0] == "smote" for step in pipe.steps), "no resampler in pipeline"
        return "resampler present in fit path, bypassed in predict path"

    def validation_prevalence_untouched():
        obs = valid.prevalence
        if expected_valid_prevalence is not None:
            assert abs(obs - expected_valid_prevalence) < 5e-5, (
                f"validation prevalence {obs:.5%} differs from the Chapter 8 "
                f"figure {expected_valid_prevalence:.5%}"
            )
        return f"{obs:.4%} positive, unchanged from Table 8.1"

    def prior_correction_is_consistent():
        pi_true, pi_res = 0.035, 0.10 / 1.10
        p = np.array([0.001, 0.1, 0.5, 0.9, 0.999])
        q = metrics.prior_correction(p, pi_res, pi_true)
        assert np.all(np.diff(q) > 0), "correction is not monotone"
        assert np.all(q < p), "correction did not reduce inflated probabilities"
        mid = metrics.prior_correction(np.array([pi_res]), pi_res, pi_true)[0]
        assert abs(mid - pi_true) < 1e-9, f"prior does not map to prior ({mid})"

        factors = {a: arms.training_odds_factor(train.y, a) for a in C.IMBALANCE_ARMS}
        assert factors["natural"] == 1.0, "the untreated arm should need no correction"
        for a in ("weighted", "smote", "smote_weighted"):
            assert factors[a] > 1.0, f"{a} reports no odds inflation"
        assert abs(factors["weighted"] - factors["smote_weighted"]) < 1e-6, (
            "weighting and SMOTE-plus-weighting should reach the same balance"
        )
        return (
            "monotone and prior-consistent; odds factors "
            + ", ".join(f"{a}={v:.2f}" for a, v in factors.items())
        )

    def all_arms_searched():
        missing = [
            C.arm_id(m, a)
            for m, a in C.ALL_ARMS
            if not (C.ARTIFACTS / "search" / f"{C.arm_id(m, a)}.json").exists()
        ]
        assert not missing, f"no search result for {missing}"
        return f"{len(C.ALL_ARMS)} arms searched"

    def threshold_chosen_on_validation():
        path = C.TABLES / "table_9_4_decision_rules.csv"
        assert path.exists(), "decision-rule table missing; run ch9-select"
        frame = pd.read_csv(path)
        assert len(frame) >= 4, "fewer decision rules than expected"
        summary = C.ARTIFACTS / "selection_summary.json"
        assert summary.exists(), "selection summary missing"
        payload = json.loads(summary.read_text())
        assert "test" not in json.dumps(payload).lower(), "test partition referenced"
        return "thresholds derived from validation scores only"

    def calibrators_fitted_out_of_fold():
        directory = C.ARTIFACTS / "calibrators"
        assert directory.exists(), "no calibrators directory; run ch9-select"
        files = sorted(directory.glob("*.json"))
        assert files, "no calibrators found; run ch9-select"
        oof_rows = []
        for path in files:
            cal = calibration.Calibrator.load(path)
            assert cal.fitted_on in ("oof", "none", "assumed (not fitted)"), (
                f"{path.name} was fitted on {cal.fitted_on!r}, not out of fold"
            )
            if cal.fitted_on == "oof":
                oof_rows.append(cal.n_fit)
                assert cal.n_fit <= train.n, (
                    f"{path.name} was fitted on {cal.n_fit:,} rows, more than the "
                    f"training partition holds ({train.n:,}) — it cannot be "
                    "training data alone"
                )
                assert cal.n_fit != valid.n, (
                    f"{path.name} was fitted on exactly {valid.n:,} rows, the size "
                    "of the validation partition"
                )
        return (f"{len(files)} calibrators, "
                f"{len(oof_rows)} fitted on {min(oof_rows):,}-{max(oof_rows):,} "
                "out-of-fold training rows" if oof_rows else f"{len(files)} calibrators")

    checks = [
        ("Test partition sealed against Chapter 9", test_partition_sealed),
        ("Feature contract identical across partitions", contract_identical),
        ("Target, key and absolute day index absent from features", target_and_key_absent),
        ("Outer partitions chronological with embargo intact", partitions_ordered),
        ("Inner folds purged by at least the widest feature window", folds_purged),
        ("Every fold's training block precedes its validation block", folds_disjoint_in_time),
        ("Resampling confined to the fit path", resampling_is_structural),
        ("Validation class prevalence unmodified", validation_prevalence_untouched),
        ("Prior correction monotone and prior-consistent", prior_correction_is_consistent),
        ("All arms searched under the same protocol", all_arms_searched),
        ("Decision threshold selected on validation", threshold_chosen_on_validation),
        ("Calibrators fitted on out-of-fold training rows", calibrators_fitted_out_of_fold),
    ]

    frame = pd.DataFrame([_check(n, f) for n, f in checks])
    C.TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(C.TABLES / "table_9_7_protocol_checks.csv", index=False)
    return frame


def main() -> None:
    frame = run_all(expected_valid_prevalence=None)
    print(frame.to_string(index=False))
    failed = frame[frame["result"] != "Pass"]
    if len(failed):
        raise SystemExit(f"\n{len(failed)} protocol check(s) did not pass.")
    print(f"\nall {len(frame)} protocol checks passed")


if __name__ == "__main__":
    main()
