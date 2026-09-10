"""Orchestrator: search all twelve arms, write the comparison table.

Each arm's result is written to its own JSON file as soon as it finishes, and an
arm whose file already exists is skipped. A twelve-arm search on the full
training partition runs for hours, so it has to survive being interrupted.

Usage:
    python -m src.models.grid                 # all arms, resuming
    python -m src.models.grid --force         # ignore existing results
    python -m src.models.grid --arms lgbm     # one model family
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from . import config as C, cv, data, tune


def search_dir():
    d = C.ARTIFACTS / "search"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_training_partition() -> data.Partition:
    part = data.load_partition("train")
    if C.ROW_CAP:
        part = data.subsample(part, C.ROW_CAP)
    return part


def fingerprint(part: data.Partition) -> dict:
    """What a cached search result was produced from.

    Recorded in every arm's JSON so a result can be recognised as belonging to
    a different run. Without it, a two-trial rehearsal on synthetic data leaves
    twelve files behind that the real run silently accepts as done.
    """
    return {
        "profile": C.PROFILE,
        "n_rows": int(part.n),
        "n_features": int(part.X.shape[1]),
        "day_min": int(part.day.min()),
        "day_max": int(part.day.max()),
        "n_folds": C.N_FOLDS,
        "purge_days": C.PURGE_DAYS,
    }


def _stale_reasons(cached: dict, current: dict) -> list[str]:
    recorded = cached.get("fingerprint")
    if recorded is None:
        return ["written before run fingerprints were recorded"]
    return [
        f"{k}: cached {recorded.get(k)!r} vs now {v!r}"
        for k, v in current.items()
        if recorded.get(k) != v
    ]


def run(models=None, imbalance=None, force: bool = False) -> pd.DataFrame:
    models = list(models or C.MODELS)
    imbalance = list(imbalance or C.IMBALANCE_ARMS)

    part = load_training_partition()
    folds = cv.make_folds(part.day)

    print(f"\n{part!r}")
    print(f"profile={C.PROFILE}  trials/boosting arm={C.N_TRIALS}  "
          f"logreg grid={len(C.LOGREG_GRID)}")
    for f in folds:
        print("  " + f.describe())

    fold_frame = pd.DataFrame(cv.fold_table(folds, part.y))
    C.TABLES.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(C.TABLES / "table_9_1_cv_folds.csv", index=False)
    print("\n" + fold_frame.to_string(index=False))

    rows = []
    fp = fingerprint(part)
    todo = [(m, a) for m in models for a in imbalance]
    for i, (model, arm) in enumerate(todo, start=1):
        arm_id = C.arm_id(model, arm)
        path = search_dir() / f"{arm_id}.json"
        if path.exists() and not force:
            cached = json.loads(path.read_text())
            stale = _stale_reasons(cached, fp)
            if not stale:
                print(f"\n[{i}/{len(todo)}] {arm_id}: cached, skipping")
                rows.append(cached)
                continue
            print(f"\n[{i}/{len(todo)}] {arm_id}: cached result discarded — "
                  + "; ".join(stale))

        print(f"\n[{i}/{len(todo)}] {arm_id}: searching")
        result = tune.run_arm(model, arm, part, folds)
        result["fingerprint"] = fp
        path.write_text(json.dumps(result, indent=2))
        print(
            f"    cv PR-AUC {result['pr_auc_mean']:.4f} "
            f"(sd {result['pr_auc_std']:.4f}) over {result['n_trials_run']} trials, "
            f"{result['n_pruned']} pruned, {result['search_seconds']:.0f}s"
        )
        rows.append(result)

    frame = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    cols = [
        "arm_id", "model", "imbalance_arm", "pr_auc_mean", "pr_auc_std",
        "n_trials_run", "n_pruned", "search_seconds",
    ]
    out = frame[cols]
    out.to_csv(C.TABLES / "table_9_2_arm_search.csv", index=False)
    print("\n=== search complete ===")
    print(out.to_string(index=False))
    print(f"\nwritten to {C.TABLES / 'table_9_2_arm_search.csv'}")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None,
                    help="model families to run (logreg lgbm xgb)")
    ap.add_argument("--imbalance", nargs="*", default=None,
                    help="imbalance arms to run")
    ap.add_argument("--force", action="store_true", help="ignore cached results")
    args = ap.parse_args()
    run(models=args.arms, imbalance=args.imbalance, force=args.force)


if __name__ == "__main__":
    main()
