"""
Missingness as signal rather than as a defect.

Section 7.7 established that missingness in this dataset is heavy, uneven, and
partly structural: R_emaildomain is absent whenever there is no distinct
recipient, dist1 and dist2 measure different things and are not both expected to
apply, and 75.6% of transactions have no identity record at all -- a subpopulation
whose fraud rate differs from the linked subpopulation by a factor of 3.7.

Two consequences follow, and this module implements both.

First, the pattern of absence is itself predictive, so it is encoded explicitly:
per-family missing counts and indicator columns for the structurally optional
fields. Little and Rubin's framework makes the reason this matters precise -- if
absence is not missing-at-random with respect to fraud, discarding the pattern
discards information.

Second, the values themselves are *not* imputed here. XGBoost and LightGBM both
implement a default-direction rule that learns which side of a split a missing
value should follow, which subsumes the missing-incorporated-in-attribute
approach evaluated by Twala et al. Imputing before training would replace a
learnable signal with a constant. The logistic regression baseline of Chapter 9
does require imputation, but that is a model-specific requirement and belongs
inside that model's own pipeline, not in a shared feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def v_missingness_frame(vdf: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the V block's per-row missingness before it is joined.

    Computed on the standalone V frame so the caller can attach two summary
    columns instead of holding all 339 anonymised columns alongside the base
    frame; only the retained subset is joined afterwards.
    """
    v_cols = [c for c in vdf.columns if c != cfg.KEY]
    missing = vdf[v_cols].isna().sum(axis=1)
    return pd.DataFrame(
        {
            cfg.KEY: vdf[cfg.KEY].to_numpy(),
            "n_missing_v": missing.astype("int16").to_numpy(),
            "v_missing_share": (missing / max(len(v_cols), 1)).astype("float32").to_numpy(),
        }
    )


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach identity-linkage, per-family missing counts and structural flags."""
    identity_cols = [
        c for c in cfg.IDENTITY_NUMERIC + cfg.IDENTITY_CATEGORICAL if c in df.columns
    ]
    if identity_cols:
        # A transaction is identity-linked when the left join produced any value.
        df["has_identity"] = (~df[identity_cols].isna().all(axis=1)).astype("int8")
        df["n_missing_identity"] = df[identity_cols].isna().sum(axis=1).astype("int16")
    else:
        df["has_identity"] = np.int8(0)
        df["n_missing_identity"] = np.int16(0)

    named = [c for c in cfg.NAMED_NUMERIC + cfg.NAMED_CATEGORICAL if c in df.columns]
    d_cols = [c for c in cfg.D_COLS if c in df.columns]
    m_cols = [c for c in cfg.M_COLS if c in df.columns]

    df["n_missing_named"] = df[named].isna().sum(axis=1).astype("int16")
    df["n_missing_d"] = df[d_cols].isna().sum(axis=1).astype("int16")
    df["n_missing_m"] = df[m_cols].isna().sum(axis=1).astype("int16")

    for col in cfg.STRUCTURAL_MISSING_COLS:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype("int8")

    return df


def missingness_feature_names(df: pd.DataFrame) -> list[str]:
    names = ["has_identity", "n_missing_identity", "n_missing_named",
             "n_missing_d", "n_missing_m", "n_missing_v", "v_missing_share"]
    names += [f"{c}_is_missing" for c in cfg.STRUCTURAL_MISSING_COLS]
    return [c for c in names if c in df.columns]
