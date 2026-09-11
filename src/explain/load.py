import mlflow
from src.explain import config as C
from sklearn.pipeline import Pipeline
from functools import lru_cache

mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)   

@lru_cache(maxsize=1)
def load_model():
    try:
        return mlflow.sklearn.load_model(C.MODEL_URI)
    except Exception:
        m = mlflow.pyfunc.load_model(C.MODEL_URI)
        return getattr(m, "_model_impl", m)

@lru_cache(maxsize=1)
def load_estimator():
    """The fitted LGBMClassifier itself — what TreeExplainer needs."""
    m = load_model()
    return m.named_steps["model"] if isinstance(m, Pipeline) else m