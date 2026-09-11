"""Chapter 10, Step 3.7 — compute and cache TreeSHAP arrays for both sets."""
import time
import numpy as np
import pandas as pd
import shap

from src.explain import config as C
from src.explain.load import load_estimator


def _matrix(sv):
    """Normalise shap output across library versions to (n, n_features)."""
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def _base(ex):
    b = ex.expected_value
    return float(b[1]) if np.ndim(b) else float(b)


def explain(name, X, est, ex):
    t0 = time.time()
    sv = _matrix(ex.shap_values(X))
    dt = time.time() - t0

    base = _base(ex)
    margin = est.predict(X, raw_score=True)
    resid = float(np.abs(sv.sum(1) + base - margin).max())
    print(f"{name}: {sv.shape[0]:,} x {sv.shape[1]}, base {base:.4f}, "
          f"additivity max {resid:.1e}, {dt/60:.1f} min", flush=True)
    if resid > 1e-4:
        raise SystemExit(f"ABORT: additivity residual {resid:.1e} exceeds 1e-4 ({name})")

    out = C.SHAP_DIR / f"{name}{'_smoke' if C.LIMIT else ''}.npz"
    np.savez_compressed(
        out,
        shap=sv.astype(np.float32),
        base=np.float32(base),
        row_index=X.index.to_numpy(),
        feature_names=np.array(X.columns.tolist()),
    )
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
    return base


def main():
    C.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    est = load_estimator()
    names = est.booster_.feature_name()
    ex = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")

    valid = pd.read_parquet(C.DATA_DIR / "valid_v1.parquet")[names]
    train = pd.read_parquet(C.DATA_DIR / "train_v1.parquet")[names]
    sample = train.sample(n=min(C.SELECTION_ROWS, len(train)), random_state=C.SEED)

    if C.LIMIT:
        valid, sample = valid.iloc[:C.LIMIT], sample.iloc[:C.LIMIT]
        print(f"CH10_LIMIT={C.LIMIT} — SMOKE RUN, output is not analysis material",
              flush=True)

    b_valid = explain("valid", valid, est, ex)
    b_train = explain("train_sample", sample, est, ex)
    if b_valid != b_train:
        raise SystemExit(f"ABORT: base values differ ({b_valid} vs {b_train})")
    print("base values identical:", b_valid)


if __name__ == "__main__":
    main()