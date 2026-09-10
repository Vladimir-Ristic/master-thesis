"""A thin wrapper over MLflow.

Two reasons this exists rather than calling MLflow directly from the modelling
code. First, the test suite and the smoke profile must run with no tracking
server present, so every call degrades to a no-op when tracking is disabled or
unreachable. Second, it keeps one place where the experiment name, run naming
convention and registered model name are decided, which is what makes the run
history readable in Chapter 12 rather than a pile of unnamed runs.

Disable with CH9_MLFLOW=0.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re

from . import config as C

ENABLED = os.environ.get("CH9_MLFLOW", "1") != "0"

_mlflow = None
_ready = False


def _client():
    global _mlflow, _ready
    if not ENABLED:
        return None
    if _ready:
        return _mlflow
    try:
        import mlflow

        mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
        _mlflow = mlflow
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[tracking] MLflow unavailable ({exc}); continuing without tracking.")
        _mlflow = None
    _ready = True
    return _mlflow


def available() -> bool:
    return _client() is not None


@contextlib.contextmanager
def run(name: str, nested: bool = False, tags: dict | None = None):
    mlflow = _client()
    if mlflow is None:
        yield None
        return
    with mlflow.start_run(run_name=name, nested=nested) as active:
        if tags:
            mlflow.set_tags(tags)
        yield active


def log_params(params: dict) -> None:
    mlflow = _client()
    if mlflow is not None:
        mlflow.log_params({k: v for k, v in params.items() if v is not None})


def log_metrics(metrics: dict, step: int | None = None) -> None:
    mlflow = _client()
    if mlflow is None:
        return
    clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
    mlflow.log_metrics(clean, step=step)


def log_dict(obj: dict, path: str) -> None:
    mlflow = _client()
    if mlflow is not None:
        mlflow.log_dict(obj, path)


def log_artifact(path) -> None:
    mlflow = _client()
    if mlflow is not None:
        mlflow.log_artifact(str(path))


_UNTRUSTED = re.compile(r"Untrusted types found in the file: (\[[^\]]*\])")


def log_model(pipeline, artifact_path: str = "model", register: bool = False,
              input_example=None) -> dict:
    """Log a fitted pipeline, optionally into the model registry.

    Registration matters for RQ3: Chapter 11's FastAPI service loads the model
    by registry name and version rather than by file path, so the serving layer
    has no knowledge of where training happened.

    MLflow 3 serialises scikit-learn models with `skops`, which refuses to write
    a class it cannot audit — and neither LightGBM nor XGBoost is on its trusted
    list. Rather than fall back to pickle, the declared types are read from the
    refusal and passed back as `skops_trusted_types`, so the safer format is
    kept and the artifact carries an explicit record of every class it
    deserialises. That list is logged alongside the model: for RQ3 it is the
    provenance statement Chapter 11's serving container needs, since it names
    exactly what the runtime is trusted to load.

    Returns a dict describing what happened. Never raises: a tracking failure
    must not destroy a completed modelling run.
    """
    mlflow = _client()
    if mlflow is None:
        return {"status": "tracking disabled"}

    kwargs = {"name": artifact_path, "sk_model": pipeline}
    if input_example is not None:
        # float64 avoids MLflow's integer-column schema warning, and matches
        # what a JSON request body will deserialise to in Chapter 11.
        try:
            kwargs["input_example"] = input_example.astype("float64")
        except Exception:
            kwargs["input_example"] = input_example
    if register:
        kwargs["registered_model_name"] = C.REGISTERED_MODEL_NAME

    def attempt(extra: dict | None = None):
        call = dict(kwargs, **(extra or {}))
        try:
            mlflow.sklearn.log_model(**call)
        except TypeError:  # pragma: no cover - older MLflow signature
            call["artifact_path"] = call.pop("name")
            mlflow.sklearn.log_model(**call)

    try:
        attempt()
        return {"status": "logged", "format": "skops", "trusted_types": []}
    except Exception as exc:
        match = _UNTRUSTED.search(str(exc))
        if match:
            try:
                types = ast.literal_eval(match.group(1))
            except Exception:
                types = []
            if types:
                try:
                    attempt({"skops_trusted_types": types})
                    log_dict({"serialization_format": "skops",
                              "skops_trusted_types": types},
                             f"{artifact_path}_serialization.json")
                    print(f"[tracking] model logged with {len(types)} declared "
                          f"trusted types: {', '.join(types)}")
                    return {"status": "logged", "format": "skops",
                            "trusted_types": types}
                except Exception as exc2:
                    print(f"[tracking] skops retry failed ({type(exc2).__name__}); "
                          "falling back to cloudpickle")
        try:
            attempt({"serialization_format": "cloudpickle"})
            print("[tracking] model logged with cloudpickle serialization")
            return {"status": "logged", "format": "cloudpickle", "trusted_types": []}
        except Exception as exc3:
            print(f"[tracking] could not log the model ({type(exc3).__name__}: "
                  f"{exc3}). Metrics and tables are unaffected; re-register later "
                  "with 'python -m src.models.select --register-only'.")
            return {"status": "failed", "error": f"{type(exc3).__name__}: {exc3}"}
