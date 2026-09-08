"""
Stateful encoders -- every parameter here is estimated from the training period
only, then applied unchanged to validation and test.

This is the module where leakage would enter if it entered anywhere. A category
frequency, a correlation matrix or a retained-column list computed over the full
dataset silently encodes the evaluation period into the training features;
Kaufman et al. classify exactly this as leakage in training data. Fitting on the
training period alone is also the deployable choice, since a model in production
can only ever have seen the past.

The fitted object is serialised to disk. Chapter 11's serving layer reloads this
same object rather than re-deriving the mappings, which is what makes the
features computed at inference identical to those computed at training time.

All category maps are keyed by the string rendering of a level. That keeps the
artifact independent of pandas dtype behaviour, which differs between pandas 2
and 3, and makes the mapping inspectable in the JSON summary written alongside.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

# Identifier-like keys are frequency-encoded only. Assigning them ordinal codes
# would hand the model an arbitrary integer identifier to split on, which fits
# individual accounts rather than behaviour.
_FREQUENCY_ONLY = {"uid_card", "uid_card_addr", "uid_account"}


def is_categorical_like(series: pd.Series) -> bool:
    """
    True for any column that is not already numeric.

    Written against dtype *kind* rather than against ``object`` specifically:
    pandas 2 represents text columns as ``object`` while pandas 3 gives them a
    dedicated string dtype, and a check for ``object`` alone silently lets text
    columns through to the model on pandas 3.
    """
    if pd.api.types.is_bool_dtype(series):
        return False
    return not pd.api.types.is_numeric_dtype(series)


def _as_str(series: pd.Series) -> pd.Series:
    """String view of a column with missing values preserved as NA."""
    return series.astype("string")


# --------------------------------------------------------------------------
# V-column reduction (fitted separately so the full V block never has to be
# joined onto the base frame before it is reduced)
# --------------------------------------------------------------------------
def fit_v_reduction(v_train: pd.DataFrame) -> list[str]:
    """
    Reduce the anonymised V block to one representative per correlated cluster.

    The V columns arrive in blocks that share an identical missingness pattern,
    which is a structural fingerprint of how Vesta generated them: columns
    produced by the same underlying computation are populated for the same
    transactions. Grouping by missing-value count recovers those blocks, and
    within each block the columns are heavily collinear because they are
    variations on one ranking or counting construction.

    Within a block, columns are visited in descending order of distinct values
    and a column is retained only if its absolute correlation with every
    already-retained column stays below the threshold, which keeps the
    higher-resolution member of each cluster. Only training-period rows are
    passed in, so no evaluation-period covariance influences the selection.
    """
    cols = list(v_train.columns)
    if not cols:
        return []
    n = min(len(v_train), cfg.V_REDUCTION_SAMPLE_ROWS)
    sample = v_train.sample(n=n, random_state=42) if n < len(v_train) else v_train

    blocks: dict[int, list[str]] = defaultdict(list)
    nan_counts = sample.isna().sum()
    for col in cols:
        blocks[int(nan_counts[col])].append(col)

    keep: list[str] = []
    for _, block_cols in sorted(blocks.items()):
        if len(block_cols) == 1:
            keep.append(block_cols[0])
            continue
        block = sample[block_cols]
        nunique = block.nunique(dropna=True).sort_values(ascending=False)
        corr = block.corr().abs()
        kept_here: list[str] = []
        for col in nunique.index:
            if not kept_here:
                kept_here.append(col)
                continue
            worst = corr.loc[col, kept_here].max()
            if pd.isna(worst) or worst < cfg.V_CORRELATION_THRESHOLD:
                kept_here.append(col)
        keep.extend(kept_here)

    return sorted(keep, key=lambda c: int(c[1:]))


@dataclass
class FeatureEncoder:
    """Fit on the training partition; apply to every partition."""

    identity_drop_: list[str] = field(default_factory=list)
    v_keep_: list[str] = field(default_factory=list)
    freq_maps_: dict[str, dict] = field(default_factory=dict)
    ordinal_maps_: dict[str, dict] = field(default_factory=dict)
    constant_drop_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    # ----------------------------------------------------------------- fit
    def fit(self, train: pd.DataFrame, v_keep: list[str] | None = None) -> "FeatureEncoder":
        """
        ``train`` must already contain only the retained V columns; pass the list
        of those columns as ``v_keep`` so the artifact records what was selected.
        """
        if v_keep is None:
            v_keep = [c for c in train.columns if c.startswith("v") and c[1:].isdigit()]
        self.v_keep_ = list(v_keep)
        self.identity_drop_ = self._fit_identity_drop(train)
        self.freq_maps_ = self._fit_frequency_maps(train)
        self.ordinal_maps_ = self._fit_ordinal_maps(train)

        preview = self.transform(train, _skip_reindex=True)
        nunique = preview.nunique(dropna=False)
        self.constant_drop_ = sorted(nunique[nunique <= 1].index.tolist())
        self.feature_names_ = [c for c in preview.columns if c not in set(self.constant_drop_)]
        self.fitted_ = True
        return self

    def _fit_identity_drop(self, train: pd.DataFrame) -> list[str]:
        """
        Drop identity columns that are near-empty even among linked transactions.

        Section 7.7 measured missingness on the 144,233 identity-linked rows, not
        on the whole dataset, and the same denominator is used here: a column that
        is absent for 96% of the rows that *have* an identity record carries too
        little coverage to support a split, and cannot be named in business terms
        either, so it fails both the statistical and the RQ2 interpretability test.
        """
        identity_cols = [
            c for c in cfg.IDENTITY_NUMERIC + cfg.IDENTITY_CATEGORICAL
            if c in train.columns
        ]
        if not identity_cols or "has_identity" not in train.columns:
            return []
        linked = train.loc[train["has_identity"] == 1, identity_cols]
        if linked.empty:
            return []
        share = linked.isna().mean()
        return sorted(share[share > cfg.IDENTITY_MISSING_DROP_THRESHOLD].index.tolist())

    def _fit_frequency_maps(self, train: pd.DataFrame) -> dict[str, dict]:
        maps = {}
        for col in cfg.FREQUENCY_ENCODE_COLS:
            if col not in train.columns:
                continue
            counts = _as_str(train[col]).value_counts(dropna=True)
            maps[col] = {str(k): int(v) for k, v in counts.items()}
        return maps

    def _fit_ordinal_maps(self, train: pd.DataFrame) -> dict[str, dict]:
        maps = {}
        for col in train.columns:
            if col in _FREQUENCY_ONLY or col in set(self.identity_drop_):
                continue
            if not is_categorical_like(train[col]):
                continue
            levels = sorted({str(v) for v in _as_str(train[col]).dropna().unique()})
            maps[col] = {level: i for i, level in enumerate(levels)}
        return maps

    # ----------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame, _skip_reindex: bool = False) -> pd.DataFrame:
        out = df.copy()

        drop = set(self.identity_drop_)
        drop |= {cfg.KEY, cfg.TARGET, cfg.TIME}
        # Absolute time indices are excluded from the feature matrix. tx_day and
        # the D-counter anchors take values in the training range only, so a split
        # learned on them cannot transfer to a later evaluation period; they exist
        # to partition the data and to build entity keys, not to be predictors.
        drop |= {"tx_day"}
        drop |= {c for c in out.columns if c.endswith("_anchor")}
        v_all = {c for c in out.columns if c.startswith("v") and c[1:].isdigit()}
        drop |= (v_all - set(self.v_keep_))

        for col, mapping in self.freq_maps_.items():
            if col not in out.columns:
                continue
            s = _as_str(out[col])
            mapped = s.map(mapping)
            # Present but unseen in training -> frequency 0; absent stays absent.
            values = np.where(
                s.isna().to_numpy(),
                np.nan,
                pd.to_numeric(mapped, errors="coerce").fillna(0.0).to_numpy(),
            )
            out[f"{col}_freq"] = values.astype("float32")

        for col, mapping in self.ordinal_maps_.items():
            if col not in out.columns:
                continue
            s = _as_str(out[col])
            mapped = pd.to_numeric(s.map(mapping), errors="coerce")
            values = np.where(
                s.isna().to_numpy(),
                np.nan,
                mapped.fillna(cfg.UNSEEN_CATEGORY_CODE).to_numpy(),
            )
            out[col] = values.astype("float32")

        drop |= _FREQUENCY_ONLY & set(out.columns)
        out = out.drop(columns=[c for c in drop if c in out.columns])

        # Safety net: nothing non-numeric may reach the model matrix.
        leftover = [c for c in out.columns if is_categorical_like(out[c])]
        out = out.drop(columns=leftover)

        if not _skip_reindex:
            out = out.reindex(columns=self.feature_names_)
        return out

    # ------------------------------------------------------------ persist
    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "feature_encoder.joblib")
        summary = {
            "feature_version": cfg.FEATURE_VERSION,
            "n_features": len(self.feature_names_),
            "identity_columns_dropped": self.identity_drop_,
            "v_columns_kept": self.v_keep_,
            "constant_columns_dropped": self.constant_drop_,
            "frequency_encoded": sorted(self.freq_maps_),
            "ordinal_encoded": {k: len(v) for k, v in sorted(self.ordinal_maps_.items())},
            "feature_names": self.feature_names_,
        }
        (directory / "encoder_summary.json").write_text(json.dumps(summary, indent=2))

    @staticmethod
    def load(directory: Path) -> "FeatureEncoder":
        import joblib

        return joblib.load(directory / "feature_encoder.joblib")
