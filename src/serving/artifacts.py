"""Chapter 11: load every serving artifact exactly once.

Nothing here recomputes anything Chapters 8-10 already produced. The calibrator is
Chapter 9's own Calibrator object, the family and group maps are Chapter 10's
committed JSON, and the feature contract is read from the pinned file and asserted
against the registered model. A serving layer that reimplements any of these can
disagree with training without ever raising.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import mlflow

import src.features.encoders
from src.models.calibration import Calibrator
from src.serving import config as C

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Artifacts:
    pipeline: Any
    estimator: Any
    booster: Any
    feature_names: list[str]
    calibrator: Calibrator
    odds_factor: float
    threshold: float
    decision_rule: str
    family_of: dict[str, str]
    group_of: dict[str, str]
    encoder: Any
    model_name: str
    model_version: str
    source: str
    loaded_at: float

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def render_feature(self, feature: str) -> str:
        return render_feature(feature, self.group_of)


def _load_contract() -> list[str]:
    names = json.loads(C.CONTRACT_PATH.read_text())["feature_names"]
    if len(names) != C.EXPECTED_N_FEATURES:
        raise RuntimeError(f"contract has {len(names)} names, expected {C.EXPECTED_N_FEATURES}")
    return names


def _load_threshold() -> tuple[float, str]:
    with C.DECISION_RULES_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["rule"] == C.DECISION_RULE:
                return float(row["threshold"]), row["rule"]
    raise RuntimeError(f"rule {C.DECISION_RULE!r} not found in {C.DECISION_RULES_PATH}")


def _load_calibrator() -> tuple[Calibrator, float]:
    payload = json.loads(C.SELECTION_SUMMARY_PATH.read_text())["calibrator"]
    factor = float(payload.get("diagnostics", {}).get("assumed_odds_factor", 1.0))
    return Calibrator.from_dict(payload), factor


def _load_maps() -> tuple[dict[str, str], dict[str, str], list[str]]:
    fm = json.loads(C.FEATURE_MAP_PATH.read_text())
    return fm["feature_family"], fm["v_group"], fm["feature_order"]


def _load_model() -> tuple[Any, Any, str]:
    """Fork 4: registry first, baked copy only if the registry is unreachable."""
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    try:
        pipeline = mlflow.sklearn.load_model(C.MODEL_URI)
        source = C.MODEL_URI
    except Exception as exc:  # noqa: BLE001 - any registry failure falls back
        if not C.MODEL_FALLBACK_DIR.exists():
            raise RuntimeError(
                f"registry unreachable ({exc}) and no fallback at {C.MODEL_FALLBACK_DIR}"
            ) from exc
        log.warning("registry unreachable (%s); loading baked fallback", exc)
        pipeline = mlflow.sklearn.load_model(str(C.MODEL_FALLBACK_DIR))
        source = f"fallback:{C.MODEL_FALLBACK_DIR}"
    estimator = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    return pipeline, estimator, source


@lru_cache(maxsize=1)
def load_artifacts() -> Artifacts:
    t0 = time.time()
    contract = _load_contract()
    pipeline, estimator, source = _load_model()
    booster = estimator.booster_
    names = list(booster.feature_name())

    if names != contract:
        raise RuntimeError("registered model feature names do not match the pinned contract")
    if booster.num_trees() != C.EXPECTED_N_TREES:
        raise RuntimeError(f"{booster.num_trees()} trees, expected {C.EXPECTED_N_TREES}")
    if estimator.get_params()["num_leaves"] != C.EXPECTED_NUM_LEAVES:
        raise RuntimeError("num_leaves does not match the Chapter 9 configuration")

    family_of, group_of, feature_order = _load_maps()
    if feature_order != contract:
        raise RuntimeError("Chapter 10 feature_map order disagrees with the contract")
    if set(family_of) != set(contract):
        raise RuntimeError("family map does not cover the contract exactly")

    calibrator, odds_factor = _load_calibrator()
    threshold, rule = _load_threshold()

    art = Artifacts(
        pipeline=pipeline,
        estimator=estimator,
        booster=booster,
        feature_names=contract,
        calibrator=calibrator,
        odds_factor=odds_factor,
        threshold=threshold,
        decision_rule=rule,
        family_of=family_of,
        group_of=group_of,
        encoder=joblib.load(C.ENCODER_PATH),
        model_name=C.MODEL_NAME,
        model_version=C.MODEL_VERSION,
        source=source,
        loaded_at=t0,
    )
    log.info("artifacts loaded in %.1fs from %s", time.time() - t0, source)
    return art


def main() -> None:
    a = load_artifacts()
    print(f"source          {a.source}")
    print(f"model           {a.model_name} v{a.model_version}")
    print(f"trees / leaves  {a.booster.num_trees()} / {a.estimator.get_params()['num_leaves']}")
    print(f"features        {a.n_features}")
    print(f"calibrator      {a.calibrator}")
    print(f"odds factor     {a.odds_factor}")
    print(f"rule            {a.decision_rule}")
    print(f"threshold       {a.threshold!r}")
    print(f"families/groups {len(set(a.family_of.values()))} / {len(set(a.group_of.values()))}")
    print(f"encoder         {len(a.encoder.feature_names_)} features")
    print(f"render examples {a.render_feature('v70')} | {a.render_feature('transactionamt')}")


def render_feature(feature: str, group_of: dict[str, str]) -> str:
    """Section 10.6: an anonymised feature is reported as its group, never as an
    individual column and never as a business claim."""
    group = group_of.get(feature)
    return f"V-group {group[1:]}" if group else feature


if __name__ == "__main__":
    main()