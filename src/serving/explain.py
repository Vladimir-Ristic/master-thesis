"""Chapter 11: reason codes under the Section 10.6 definition.

For a transaction the threshold sends to review, the reason codes are the three
features with the largest positive contribution to the margin, each reported with
the feature's value and its contribution in log-odds. An anonymised contributor is
rendered as its group, never as an individual column and never as a business claim.

The explainer is path-dependent TreeSHAP on the booster alone: Chapter 10 confirmed
it needs no background dataset, which is what lets the service ship the model and
nothing else. It is built once, not per request.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import shap
import warnings

from src.serving import config as C
from src.serving.artifacts import Artifacts, load_artifacts
from src.serving.schemas import ReasonCode


@lru_cache(maxsize=1)
def explainer():
    art = load_artifacts()
    return shap.TreeExplainer(art.estimator, feature_perturbation="tree_path_dependent")


def shap_values(X: pd.DataFrame) -> np.ndarray:
    """Positive-class attributions, shaped (n_rows, n_features)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*TreeExplainer shap values output has changed.*")
        sv = explainer().shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv


def reason_codes(
    X: pd.DataFrame, art: Artifacts, k: int = C.REASON_CODE_TOP_K
) -> list[list[ReasonCode]]:
    Xf = X[art.feature_names]
    sv = shap_values(Xf)
    out: list[list[ReasonCode]] = []
    for i in range(len(Xf)):
        row = sv[i]
        codes: list[ReasonCode] = []
        for j in np.argsort(-row)[:k]:
            if row[j] <= 0:
                break
            value = Xf.iloc[i, j]
            codes.append(
                ReasonCode(
                    feature=art.render_feature(art.feature_names[j]),
                    value=None if pd.isna(value) else float(value),
                    contribution=float(row[j]),
                )
            )
        out.append(codes)
    return out