"""Arm selection, probability calibration and threshold choice.

Everything here happens on the validation partition (days 127-154). The test
partition is not opened; `data.load_partition` refuses it.

Order of operations, and why it is this order:

1. Refit each arm on the **whole** training partition with its best CV
   configuration. The search fitted expanding sub-windows; the deployed model
   should see all of the training period.
2. Score the validation partition, and correct the predicted odds of every arm
   whose treatment altered the effective class balance. That is three of the
   four arms, not just the resampled ones: weighting the positive class is a
   prior shift as surely as resampling is.
3. Rank arms by validation PR-AUC — threshold-free, so the ranking does not
   depend on a decision rule.
4. Only then choose a threshold, on the winning arm, under each cost model.

Steps 3 and 4 are separated deliberately: it is what lets Chapter 9 report what
imbalance handling did to *ranking quality* independently of what threshold
calibration did to *cost*, which is the shape RQ1 asks for.
"""

from __future__ import annotations

import gc
import json
import warnings

import numpy as np
import pandas as pd

from . import arms, calibration, config as C, costs, cv, data, metrics, tracking


def _best_params(arm_id: str) -> dict:
    path = C.ARTIFACTS / "search" / f"{arm_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No search result for {arm_id}. Run `make ch9-search` first."
        )
    return json.loads(path.read_text())["best_params"]


def refit_and_score(
    model: str,
    arm: str,
    train: data.Partition,
    valid: data.Partition,
    params: dict | None = None,
    seed: int = C.SEED,
    calibrator=None,
):
    """Refit on the full training partition, return raw and calibrated scores.

    `calibrator` is applied to the raw scores. Passing None falls back to the
    analytic odds correction, which is kept only so the two can be compared.
    """
    arm_id = C.arm_id(model, arm)
    params = params if params is not None else _best_params(arm_id)

    pipe = arms.build_pipeline(model, arm, params, train.y, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(train.X, train.y)
        p_raw = pipe.predict_proba(valid.X)[:, 1]

    if calibrator is not None:
        p = calibrator.transform(p_raw)
    else:
        factor = arms.training_odds_factor(train.y, arm)
        p = metrics.odds_correction(p_raw, factor) if factor != 1.0 else p_raw
    return pipe, p_raw, p


def fit_calibrator(model, arm, train, folds, params, seed=C.SEED):
    """Fit the arm's calibrator on out-of-fold training predictions."""
    p_oof, y_oof, _, group = calibration.out_of_fold_predictions(
        model, arm, train, folds, params, seed=seed
    )
    factor = arms.training_odds_factor(train.y, arm)
    cal, report = calibration.choose(p_oof, y_oof, odds_factor=factor, group=group)
    return cal, report, p_oof, y_oof


def evaluate_all(
    models=None, imbalance=None, register_best: bool = True, seed: int = C.SEED
) -> dict:
    models = list(models or C.MODELS)
    imbalance = list(imbalance or C.IMBALANCE_ARMS)

    train = data.load_partition("train")
    valid = data.load_partition("valid")
    data.assert_same_contract(train, valid)
    data.assert_disjoint(train, valid)
    print(f"{train!r}\n{valid!r}")

    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (C.ARTIFACTS / "scores").mkdir(exist_ok=True)
    (C.ARTIFACTS / "scores_raw").mkdir(exist_ok=True)
    C.TABLES.mkdir(parents=True, exist_ok=True)

    folds = cv.make_folds(train.day)
    (C.ARTIFACTS / "calibrators").mkdir(parents=True, exist_ok=True)

    rows, scores, cal_rows = [], {}, []
    todo = [(m, a) for m in models for a in imbalance]
    for i, (model, arm) in enumerate(todo, start=1):
        arm_id = C.arm_id(model, arm)
        print(f"[{i}/{len(todo)}] {arm_id}: out-of-fold calibration, then refit")

        with tracking.run(f"select::{arm_id}", tags={"stage": "select", "arm": arm,
                                                     "model": model}):
            params = _best_params(arm_id)
            cal, cal_report, p_oof, y_oof = fit_calibrator(
                model, arm, train, folds, params, seed=seed
            )
            cal.save(C.ARTIFACTS / "calibrators" / f"{arm_id}.json")
            for entry in cal_report:
                cal_rows.append({"arm_id": arm_id, "model": model,
                                 "imbalance_arm": arm, "selected": entry["kind"] == cal.kind,
                                 **entry})
            assumed = arms.training_odds_factor(train.y, arm)
            print(f"    out-of-fold: {len(y_oof):,} rows, {y_oof.mean():.4%} positive; "
                  f"chose '{cal.kind}' on {cal.diagnostics['ranked_on']} "
                  f"(a={cal.a:.3f}, b={cal.b:.3f}); fitted odds factor "
                  f"{cal.diagnostics['fitted_shift_odds_factor']:.2f} "
                  f"vs assumed {assumed:.2f}")

            pipe, p_raw, p = refit_and_score(model, arm, train, valid, params,
                                             seed=seed, calibrator=cal)
            p_analytic = metrics.odds_correction(
                p_raw, arms.training_odds_factor(train.y, arm)
            )
            bundle = metrics.threshold_free_bundle(valid.y, p)
            bundle_raw = metrics.threshold_free_bundle(valid.y, p_raw)

            t_ed, cost_ed = costs.best_global_threshold(
                valid.y, p, valid.amount, cost_model="ed"
            )
            base = costs.baseline_costs(valid.y, valid.amount)
            row = {
                "arm_id": arm_id,
                "model": model,
                "imbalance_arm": arm,
                "valid_pr_auc": bundle["pr_auc"],
                "valid_roc_auc": bundle["roc_auc"],
                "brier_calibrated": bundle["brier"],
                "brier_raw": bundle_raw["brier"],
                "brier_analytic": metrics.brier(valid.y, p_analytic),
                "calibrator": cal.kind,
                "calibrator_a": cal.a,
                "calibrator_b": cal.b,
                "fitted_odds_factor": cal.diagnostics["fitted_shift_odds_factor"],
                "recall_at_p50": bundle["recall_at_p50"],
                "recall_at_p90": bundle["recall_at_p90"],
                "cost_ed_at_best_threshold": cost_ed,
                "savings_ed": costs.savings(cost_ed, base["ed_none"]),
                "best_threshold_ed": t_ed,
                "odds_factor": arms.training_odds_factor(train.y, arm),
                "mean_p_raw": float(np.mean(p_raw)),
                "mean_p_calibrated": float(np.mean(p)),
                "mean_p_analytic": float(np.mean(p_analytic)),
            }
            rows.append(row)
            scores[arm_id] = p
            np.save(C.ARTIFACTS / "scores" / f"{arm_id}.npy", p.astype(np.float32))
            np.save(C.ARTIFACTS / "scores_raw" / f"{arm_id}.npy", p_raw.astype(np.float32))

            tracking.log_params({"model": model, "imbalance_arm": arm,
                                 "calibrator": cal.kind, **params})
            tracking.log_dict(cal.to_dict(), f"calibrators/{arm_id}.json")
            tracking.log_metrics({k: v for k, v in row.items()
                                  if isinstance(v, (int, float))})

        del pipe
        gc.collect()

    frame = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False)
    frame.to_csv(C.TABLES / "table_9_3_validation.csv", index=False)
    np.save(C.ARTIFACTS / "scores" / "valid_y.npy", valid.y)
    np.save(C.ARTIFACTS / "scores" / "valid_amount.npy", valid.amount)

    print("\n=== validation partition ===")
    print(frame.to_string(index=False))

    winner = frame.iloc[0]
    win_model, win_arm = winner["model"], winner["imbalance_arm"]
    win_id = winner["arm_id"]
    print(f"\nselected arm: {win_id} (validation PR-AUC {winner['valid_pr_auc']:.4f}, "
          f"no-skill {valid.prevalence:.4f})")

    # --- threshold study on the winning arm -----------------------------------
    p_win = scores[win_id]
    decisions = costs.all_decision_rules(valid.y, p_win, valid.amount)
    dec_frame = pd.DataFrame([d.as_row() for d in decisions])
    dec_frame.insert(0, "arm_id", win_id)
    dec_frame.to_csv(C.TABLES / "table_9_4_decision_rules.csv", index=False)
    print("\n=== decision rules, selected arm ===")
    print(dec_frame[["rule", "threshold", "review_rate", "precision", "recall",
                     "cost_ed", "savings_ed", "cost_fx"]].to_string(index=False))

    sens = pd.DataFrame(costs.ca_sensitivity(valid.y, p_win, valid.amount))
    sens.to_csv(C.TABLES / "table_9_5_ca_sensitivity.csv", index=False)
    print("\n=== review-cost sensitivity ===")
    print(sens.to_string(index=False))

    # --- calibration: assumed against fitted ----------------------------------
    cal_frame = pd.DataFrame(cal_rows)
    cal_frame.to_csv(C.TABLES / "table_9_9_calibration.csv", index=False)

    print("\n=== calibration, assumed against fitted out of fold ===")
    print(f"validation prevalence {valid.prevalence:.4%}; "
          "a calibrated model's mean predicted probability should sit near it")
    cols = ["arm_id", "odds_factor", "fitted_odds_factor", "calibrator",
            "mean_p_raw", "mean_p_analytic", "mean_p_calibrated",
            "brier_raw", "brier_analytic", "brier_calibrated"]
    print(frame[cols].to_string(index=False))

    treated = frame[frame["odds_factor"] != 1.0]
    if len(treated):
        ratio = (treated["odds_factor"] / treated["fitted_odds_factor"]).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(ratio):
            print(f"\nThe assumed factor exceeds the fitted one by a median "
                  f"{ratio.median():.2f}x across the {len(treated)} treated arms "
                  "(above 1 means the analytic correction over-corrects).")
        wins = (treated["brier_calibrated"] < treated["brier_analytic"]).sum()
        print(f"The fitted calibrator beats the assumed correction on Brier score "
              f"in {wins} of {len(treated)} treated arms.")

    summary = {
        "selected_arm": win_id,
        "valid_pr_auc": float(winner["valid_pr_auc"]),
        "valid_no_skill_pr_auc": valid.prevalence,
        "ca_review": C.CA_REVIEW,
        "n_features": int(train.X.shape[1]),
        "selection_metric": C.PRIMARY_METRIC,
        "lowest_cost_arm": str(frame.sort_values("cost_ed_at_best_threshold")
                               .iloc[0]["arm_id"]),
        "calibrator": calibration.Calibrator.load(
            C.ARTIFACTS / "calibrators" / f"{win_id}.json"
        ).to_dict(),
    }
    (C.ARTIFACTS / "selection_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {C.ARTIFACTS / 'selection_summary.json'}")

    # --- register the selected model -----------------------------------------
    # Last, and isolated: everything above is the chapter's evidence, and a
    # serialisation problem in the registry must not discard it.
    if register_best:
        print(f"\nrefitting {win_id} for registration")
        with tracking.run(f"final::{win_id}", tags={"stage": "final", "arm": win_arm,
                                                    "model": win_model}):
            win_cal = calibration.Calibrator.load(
                C.ARTIFACTS / "calibrators" / f"{win_id}.json"
            )
            pipe, _, _ = refit_and_score(win_model, win_arm, train, valid,
                                         seed=seed, calibrator=win_cal)
            tracking.log_params({"model": win_model, "imbalance_arm": win_arm,
                                 **_best_params(win_id)})
            tracking.log_metrics(
                {
                    "valid_pr_auc": float(winner["valid_pr_auc"]),
                    "valid_roc_auc": float(winner["valid_roc_auc"]),
                    "threshold_ed_global": float(
                        dec_frame.loc[dec_frame["rule"].str.startswith(
                            "example-dependent, global"), "threshold"].iloc[0]
                    ),
                    "ca_review": C.CA_REVIEW,
                }
            )
            tracking.log_dict(
                {
                    "arm_id": win_id,
                    "best_params": _best_params(win_id),
                    "ca_review": C.CA_REVIEW,
                    "decision_rules": dec_frame.to_dict(orient="records"),
                    "feature_names": list(train.X.columns),
                },
                "final/selection.json",
            )
            outcome = tracking.log_model(pipe, register=True,
                                         input_example=valid.X.iloc[:2])
            summary["registration"] = outcome
        del pipe
        gc.collect()
        (C.ARTIFACTS / "selection_summary.json").write_text(json.dumps(summary, indent=2))

    return {"grid": frame, "decisions": dec_frame, "sensitivity": sens,
            "summary": summary}


def smote_ratio_sweep(model: str | None = None, seed: int = C.SEED) -> pd.DataFrame:
    """Effect of the resampling ratio, at a fixed model configuration.

    The grid fixes SMOTE at one ratio so that resampling intensity does not
    confound the model comparison. This is where that choice is defended: the
    same configuration is refitted across ratios and the ratio is shown to move
    PR-AUC by less than the model choice does.
    """
    train = data.load_partition("train")
    valid = data.load_partition("valid")
    model = model or "lgbm"
    params = _best_params(C.arm_id(model, "smote"))

    original = C.SMOTE_RATIO
    rows = []
    try:
        for ratio in C.SMOTE_RATIO_SWEEP:
            C.SMOTE_RATIO = ratio
            _, p_raw, p = refit_and_score(model, "smote", train, valid, params, seed,
                                          calibrator=None)
            rows.append(
                {
                    "smote_ratio": ratio,
                    "valid_pr_auc": metrics.pr_auc(valid.y, p),
                    "valid_roc_auc": metrics.roc_auc(valid.y, p),
                    "brier_raw": metrics.brier(valid.y, p_raw),
                    "brier_analytic": metrics.brier(valid.y, p),
                }
            )
            print(f"  ratio {ratio:.2f}: PR-AUC {rows[-1]['valid_pr_auc']:.4f}")
            gc.collect()
    finally:
        C.SMOTE_RATIO = original

    frame = pd.DataFrame(rows)
    frame.to_csv(C.TABLES / "table_9_6_smote_ratio.csv", index=False)
    return frame


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="also run the SMOTE ratio sweep")
    args = ap.parse_args()
    try:
        evaluate_all(register_best=not args.no_register)
    except Exception as exc:
        print(f"\nselection failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        if args.sweep:
            print("\n=== SMOTE ratio sweep ===")
            print(smote_ratio_sweep().to_string(index=False))


if __name__ == "__main__":
    main()
