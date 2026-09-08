"""
Contract tests for the assembled pipeline.

These are the assertions the written chapter relies on: that the partitions are
chronological and disjoint, that the embargo is genuinely empty, that no encoder
parameter was estimated outside the training period, and that the leakage checks
would in fact fire if something were wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import config as cfg
from src.features import transforms
from src.features.encoders import FeatureEncoder, fit_v_reduction, is_categorical_like


# ------------------------------------------------------------- stateless
def test_cyclical_hour_encoding_wraps():
    df = pd.DataFrame({cfg.TIME: [0, 23 * 3600, 24 * 3600 - 1]})
    out = transforms.add_time_parts(df.copy())
    # hour 23 must sit closer to hour 0 than hour 12 does
    p = out[["tx_hour_sin", "tx_hour_cos"]].to_numpy()
    d_23_to_0 = np.linalg.norm(p[1] - p[0])
    mid = transforms.add_time_parts(pd.DataFrame({cfg.TIME: [12 * 3600]}))
    d_12_to_0 = np.linalg.norm(mid[["tx_hour_sin", "tx_hour_cos"]].to_numpy()[0] - p[0])
    assert d_23_to_0 < d_12_to_0


def test_amount_decomposition():
    df = pd.DataFrame({"transactionamt": [100.0, 59.99, 1234.567]})
    out = transforms.add_amount_features(df.copy())
    assert out["amt_is_round"].tolist() == [1, 0, 0]
    assert out["amt_cents"].tolist() == pytest.approx([0.0, 0.99, 0.567], abs=1e-5)
    assert out["amt_log"].iloc[0] == pytest.approx(np.log1p(100.0), rel=1e-6)


def test_match_flags_keep_missing_distinct_from_false():
    df = pd.DataFrame({"m1": ["T", "F", None], "m4": ["M0", "M1", "M2"]})
    out = transforms.normalise_match_flags(df.copy())
    assert out["m1"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(out["m1"].iloc[2])
    assert out["m4"].tolist() == ["M0", "M1", "M2"]     # 3-level, left categorical


def test_day_counter_anchor_is_constant_for_a_stable_account():
    df = pd.DataFrame({cfg.TIME: [0, 5 * 86_400, 40 * 86_400], "d1": [10.0, 15.0, 50.0]})
    out = transforms.normalise_day_counters(transforms.add_time_parts(df.copy()))
    assert out["d1_anchor"].nunique() == 1


# ------------------------------------------------------------ partitions
def test_partitions_are_chronological_and_disjoint(partitions):
    tr, va, te = partitions["train"], partitions["valid"], partitions["test"]
    assert tr["tx_day"].max() < va["tx_day"].min()
    assert va["tx_day"].max() < te["tx_day"].min()
    assert not (set(tr[cfg.KEY]) & set(va[cfg.KEY]))
    assert not (set(tr[cfg.KEY]) & set(te[cfg.KEY]))
    assert not (set(va[cfg.KEY]) & set(te[cfg.KEY]))


def test_embargo_interval_is_absent_from_every_partition(partitions):
    lo, hi = cfg.EMBARGO_DAYS
    for name, df in partitions.items():
        inside = df["tx_day"].between(lo, hi - 1).sum()
        assert inside == 0, f"{name} contains {inside} embargoed rows"


def test_embargo_is_at_least_as_wide_as_the_widest_window():
    width_days = cfg.EMBARGO_DAYS[1] - cfg.EMBARGO_DAYS[0]
    assert width_days >= cfg.MAX_WINDOW_SECONDS / cfg.SECONDS_PER_DAY


def test_feature_contract_is_identical_across_partitions(partitions):
    meta = {cfg.KEY, cfg.TARGET, "tx_day"}
    cols = [[c for c in df.columns if c not in meta] for df in partitions.values()]
    assert cols[0] == cols[1] == cols[2]


def test_feature_matrix_is_entirely_numeric(partitions):
    meta = {cfg.KEY, cfg.TARGET, "tx_day"}
    for name, df in partitions.items():
        bad = [c for c in df.columns if c not in meta and is_categorical_like(df[c])]
        assert not bad, f"{name} carries non-numeric features: {bad}"


def test_absolute_time_index_is_not_a_feature(partitions):
    features = set(partitions["train"].columns) - {cfg.KEY, cfg.TARGET, "tx_day"}
    assert not any(c.endswith("_anchor") for c in features)
    assert cfg.TIME not in features


# -------------------------------------------------------------- encoders
def test_encoder_parameters_come_from_the_training_period_only(built_pipeline):
    """
    A category that occurs only after the training period must be unknown.

    The check is done directly against the fitted maps rather than against
    transformed output, because that is where a full-dataset fit would show up.
    """
    encoder = FeatureEncoder.load(cfg.ARTIFACT_DIR)
    from src.features.extract import read_interim

    base, _ = read_interim()
    base = transforms.apply_stateless(base.sort_values(cfg.TIME).reset_index(drop=True))
    train_mask = base["tx_day"] < cfg.TRAIN_DAYS[1]

    checked = 0
    for col, mapping in encoder.ordinal_maps_.items():
        if col not in base.columns:
            continue
        train_levels = {str(v) for v in base.loc[train_mask, col].dropna().unique()}
        assert set(mapping) <= train_levels, f"{col} learned a level outside training"
        checked += 1
    assert checked > 0

    for col, mapping in encoder.freq_maps_.items():
        if col not in base.columns:
            continue
        # Maps are keyed by the string rendering of a level (see encoders.py).
        counts = base.loc[train_mask, col].astype("string").value_counts()
        for level, n in list(mapping.items())[:50]:
            assert counts.get(level, 0) == n, f"{col} frequency includes later rows"


def test_unseen_categories_map_to_the_reserved_code():
    encoder = FeatureEncoder()
    train = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
    encoder.fit(train, v_keep=[])
    out = encoder.transform(pd.DataFrame({"cat": ["a", "zzz", None], "num": [1.0, 2.0, 3.0]}))
    assert out["cat"].tolist()[0] == 0.0
    assert out["cat"].tolist()[1] == cfg.UNSEEN_CATEGORY_CODE
    assert np.isnan(out["cat"].tolist()[2])


def test_v_reduction_drops_collinear_columns():
    rng = np.random.default_rng(3)
    n = 2_000
    latent = rng.normal(size=n)
    train = pd.DataFrame(
        {
            "v1": latent,
            "v2": latent + rng.normal(scale=0.01, size=n),   # near-duplicate of v1
            "v3": rng.normal(size=n),                        # independent
            "num": rng.normal(size=n),
        }
    )
    keep = fit_v_reduction(train[["v1", "v2", "v3"]])
    assert "v3" in keep
    assert len({"v1", "v2"} & set(keep)) == 1


# --------------------------------------------------------------- leakage
def test_leakage_probe_detects_an_injected_target_copy(partitions):
    """The safety net must actually fire when a leaky feature is present."""
    from src.features import validate

    poisoned = {k: v.copy() for k, v in partitions.items()}
    for df in poisoned.values():
        df["leaky_copy_of_target"] = df[cfg.TARGET].astype(float)

    ranking = validate.check_single_feature_separation(poisoned)
    assert ranking.iloc[0]["feature"] == "leaky_copy_of_target"
    assert ranking.iloc[0]["auc"] > cfg.MAX_SINGLE_FEATURE_AUC


def test_no_feature_separates_the_classes_on_its_own(partitions):
    from src.features import validate

    ranking = validate.check_single_feature_separation(partitions)
    top = ranking.iloc[0]
    assert top["auc"] <= cfg.MAX_SINGLE_FEATURE_AUC, f"suspicious feature: {top['feature']}"


def test_entity_aggregates_never_include_the_current_transaction(built_pipeline):
    """
    Spot-check the account history count against a brute-force recomputation.

    Verifying the primitive is not enough on its own; this confirms the primitive
    was wired to the right key and the right timestamp column.
    """
    from src.features import entity
    from src.features.extract import read_interim

    base, _ = read_interim()
    base = base.sort_values(cfg.TIME, kind="mergesort").reset_index(drop=True)
    base = transforms.apply_stateless(base)
    base = entity.add_entity_keys(base)
    base = entity.add_entity_aggregates(base)

    sample = base.sample(n=60, random_state=17)
    for _, row in sample.iterrows():
        expected = int(
            (
                (base["uid_account"] == row["uid_account"])
                & (base[cfg.TIME] < row[cfg.TIME])
            ).sum()
        )
        assert row["acct_cnt_hist"] == expected
