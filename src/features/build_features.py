"""
Chapter 8 orchestrator: staged raw tables -> partitioned model-ready feature sets.

Run order is fixed and each stage is a pure step over the previous one:

    1. read the interim typed frames produced by extract.py
    2. stateless row-wise transforms            (transforms.py)
    3. entity keys and past-only aggregates     (entity.py)
    4. chronological partition with embargo     (split.py)
    5. reduce the V block on training rows only (encoders.fit_v_reduction)
    6. missingness encoding                     (missingness.py)
    7. fit encoders on the training partition   (encoders.py)
    8. apply to all partitions and persist

Steps 2, 3 and 6 are computed over the full frame on purpose. Every feature they
create is either a function of the row alone or an aggregate over strictly
earlier transactions, so computing them before the split is not leakage -- it is
the same computation a production system performs, where a transaction arriving
today is scored against the history that preceded it. Steps 5 and 7 are the only
stages that estimate parameters from data, and both see the training partition
alone; the split therefore has to happen before them, which is why it sits at
step 4 rather than at the end.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config as cfg
from . import entity, missingness, split, transforms
from .encoders import FeatureEncoder, fit_v_reduction


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _feature_family(name: str) -> str:
    if name.endswith("_freq"):
        return "Frequency encoding"
    if name.startswith(("acct_", "cardaddr_", "card_secs", "card_cnt", "amt_to_", "amt_acct_")):
        return "Entity aggregation"
    if name.startswith("tx_"):
        return "Temporal"
    if name.startswith("amt_"):
        return "Amount"
    if name.startswith("n_missing") or name.endswith("_is_missing") or name in {
        "has_identity", "v_missing_share"
    }:
        return "Missingness"
    if name.endswith(("_email_provider", "_email_suffix")) or name == "email_domains_match":
        return "Email"
    if name.startswith("v") and name[1:].isdigit():
        return "Anonymised V (retained)"
    if name in cfg.C_COLS:
        return "C (counting)"
    if name in cfg.D_COLS:
        return "D (timedelta)"
    if name in cfg.M_COLS:
        return "M (match flag)"
    if name.startswith("id_") or name in {"devicetype", "deviceinfo"}:
        return "Identity"
    return "Named / business"


def build(verbose: bool = True) -> dict:
    from .extract import read_interim

    t0 = time.time()
    base, vdf = read_interim()
    v_columns = [c for c in vdf.columns if c != cfg.KEY]
    if verbose:
        _log(f"loaded base {base.shape} and V block {vdf.shape}", t0)

    base = base.sort_values(cfg.TIME, kind="mergesort").reset_index(drop=True)

    base = transforms.apply_stateless(base)
    if verbose:
        _log("stateless transforms applied", t0)

    base = entity.add_entity_keys(base)
    base = entity.add_entity_aggregates(base)
    if verbose:
        _log(f"entity aggregates applied ({base.shape[1]} columns)", t0)

    masks = split.make_splits(base["tx_day"])
    summary = split.split_summary(base, masks)
    if verbose:
        _log("splits validated\n" + summary.to_string(index=False), t0)

    # The V block is summarised and reduced while it is still a standalone frame,
    # so the base frame never has to carry all 339 anonymised columns at once --
    # the difference between a ~2.5 GB and a ~1 GB peak on the 4 GB WSL2 allocation.
    base = base.merge(missingness.v_missingness_frame(vdf), on=cfg.KEY, how="left")
    train_ids = pd.Index(base.loc[masks.train, cfg.KEY])
    v_train = vdf.loc[vdf[cfg.KEY].isin(train_ids), v_columns]
    v_keep = fit_v_reduction(v_train)
    del v_train
    if verbose:
        _log(f"V reduction fitted on training rows: {len(v_keep)}/{len(v_columns)} retained", t0)

    base = base.merge(vdf[[cfg.KEY] + v_keep], on=cfg.KEY, how="left")
    del vdf
    if verbose:
        _log(f"retained V columns joined ({base.shape[1]} columns)", t0)

    base = missingness.add_missingness_features(base)
    if verbose:
        _log("missingness features applied", t0)

    encoder = FeatureEncoder().fit(base.loc[masks.train], v_keep)
    if verbose:
        _log(
            f"encoder fitted: {len(encoder.feature_names_)} features, "
            f"{len(encoder.v_keep_)}/{len(v_columns)} V columns retained, "
            f"{len(encoder.identity_drop_)} identity columns dropped",
            t0,
        )

    cfg.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_TABLES.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, mask in [("train", masks.train), ("valid", masks.valid), ("test", masks.test)]:
        part = base.loc[mask]
        X = encoder.transform(part)
        X.insert(0, cfg.KEY, part[cfg.KEY].to_numpy())
        X.insert(1, cfg.TARGET, part[cfg.TARGET].to_numpy())
        X.insert(2, "tx_day", part["tx_day"].to_numpy())
        path = cfg.DATA_PROCESSED / f"{name}_{cfg.FEATURE_VERSION}.parquet"
        X.to_parquet(path, index=False)
        written[name] = {"path": str(path), "rows": len(X), "cols": X.shape[1]}
        if verbose:
            _log(f"wrote {path.name}: {len(X):,} rows x {X.shape[1]} cols", t0)
        del X, part

    encoder.save(cfg.ARTIFACT_DIR)
    summary.to_csv(cfg.REPORT_TABLES / "table_8_1_split_summary.csv", index=False)

    families = pd.Series(
        {f: _feature_family(f) for f in encoder.feature_names_}, name="family"
    )
    family_table = (
        families.value_counts().rename_axis("Feature family").reset_index(name="Count")
    )
    family_table.to_csv(cfg.REPORT_TABLES / "table_8_2_feature_families.csv", index=False)

    metadata = {
        "feature_version": cfg.FEATURE_VERSION,
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "n_features": len(encoder.feature_names_),
        "split_days": {
            "train": cfg.TRAIN_DAYS,
            "embargo": cfg.EMBARGO_DAYS,
            "valid": cfg.VALID_DAYS,
            "test": cfg.TEST_DAYS,
        },
        "embargo_rationale": cfg.EMBARGO_RATIONALE,
        "windows_seconds": cfg.WINDOWS,
        "partitions": written,
        "features": [
            {"name": f, "family": _feature_family(f)} for f in encoder.feature_names_
        ],
    }
    (cfg.ARTIFACT_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

    if verbose:
        _log("done", t0)
        print("\nFeature families:")
        print(family_table.to_string(index=False))
    return metadata


def main() -> None:
    build(verbose=True)


if __name__ == "__main__":
    main()
