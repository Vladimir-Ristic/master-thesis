"""Chapter 11: the decision path.

predict_proba -> Platt -> threshold. Every parameter is read from Chapter 9's
recorded artifacts; nothing here is a constant typed by hand. The recorded
odds_factor is provenance only - the fitted calibrator already subsumes the prior
correction, and applying it again would correct twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.serving.artifacts import Artifacts


@dataclass(frozen=True)
class Scored:
    transactionid: int
    probability_raw: float
    probability: float
    flagged: bool

    @property
    def decision(self) -> str:
        return "review" if self.flagged else "accept"


def score_frame(X: pd.DataFrame, art: Artifacts) -> list[Scored]:
    p_raw = art.pipeline.predict_proba(X[art.feature_names])[:, 1]
    p_cal = np.asarray(art.calibrator.transform(p_raw), dtype="float64")
    flagged = p_cal >= art.threshold
    return [
        Scored(int(tid), float(r), float(c), bool(f))
        for tid, r, c, f in zip(X.index, p_raw, p_cal, flagged)
    ]