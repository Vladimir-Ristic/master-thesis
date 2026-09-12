"""Chapter 12: the frozen reference.

A monitor whose baseline is recomputed from a rolling recent window cannot detect
slow drift, because the baseline moves with it. The reference is therefore written
once, from the training partition and the commissioning operating point, and is
re-derived only when the model is retrained.

Feature reference  : the training partition, days 1-119.
Score reference    : the validation operating point - the point the system was
                     commissioned at, not the training distribution, which a
                     fitted model scores optimistically.
Attribution ref.   : the Chapter 10 family shares.
Monitored features : the top N by mean absolute SHAP from Chapter 10's own matrix.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.monitor import config as M
from src.serving.artifacts import load_artifacts
from src.serving.score import score_frame


def _md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _shap_ranking(contract: list[str]) -> pd.Series:
    """Mean |SHAP| per feature, from the Chapter 10 training-sample matrix."""
    path = M.ARTIFACTS_DIR.parent / "explain_v1" / "shap" / "train_sample.npz"
    z = np.load(path, allow_pickle=True)
    print(f"  npz keys: {list(z.keys())}")
    values, names = None, None
    for k in z.keys():
        a = z[k]
        if a.ndim == 2 and a.shape[1] in (len(contract), len(contract) + 1):
            if values is None or a.shape[0] > values.shape[0]:
                values = a
        elif a.ndim == 1 and a.dtype.kind in "USO" and len(a) == len(contract):
            names = [str(x) for x in a]
    if values is None:
        raise SystemExit(f"no 2-D SHAP array in {path}; keys={list(z.keys())}")
    values = values[:, : len(contract)]
    used_npz = names is not None
    names = names or contract
    print(f"  shap matrix {values.shape}; names from {'npz' if used_npz else 'contract'}")
    return pd.Series(np.abs(values).mean(axis=0), index=names).sort_values(ascending=False)


def build() -> dict:
    M.ensure_dirs()
    art = load_artifacts()
    contract = list(art.feature_names)
    print(f"contract: {len(contract)} features, threshold {art.threshold!r}")

    train = pd.read_parquet(M.TRAIN_PARQUET)
    print(f"train_v1: {train.shape}")

    ranking = _shap_ranking(contract)
    missing_share = train[contract].isna().mean()
    eligible = [f for f in ranking.index
                if f in missing_share.index and missing_share[f] <= M.MAX_MISSING_SHARE]
    monitored = eligible[: M.N_MONITORED]
    dropped = [f for f in ranking.index[: M.N_MONITORED] if f not in monitored]
    print(f"monitored: {len(monitored)} features; dropped for missingness: {dropped}")

    features = {}
    for f in monitored:
        col = train[f].to_numpy(dtype="float64")
        finite = col[np.isfinite(col)]
        edges = np.unique(np.nanquantile(finite, np.linspace(0, 1, M.N_BINS + 1)))
        edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
        counts = np.histogram(finite, bins=edges)[0].astype(float)
        features[f] = {
            "edges": edges.tolist(),
            "props": (counts / max(counts.sum(), 1)).tolist(),
            "missing_share": float(np.isnan(col).mean()),
            "mean_abs_shap": float(ranking[f]),
        }

    valid = pd.read_parquet(M.VALID_PARQUET)
    scored = score_frame(valid[contract], art)
    p = np.array([s.probability for s in scored])
    flagged = np.array([s.flagged for s in scored])
    y = valid["isfraud"].to_numpy().astype(int)
    tp = int((flagged & (y == 1)).sum())
    review_rate = float(flagged.mean())
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int((y == 1).sum()), 1)
    print(f"validation commissioning point: {int(flagged.sum())} of {len(p)} flagged, "
          f"review_rate {review_rate:.6f}, precision {precision:.6f}, recall {recall:.6f}")

    s_edges = np.unique(np.quantile(p, np.linspace(0, 1, M.N_BINS + 1)))
    s_edges = np.concatenate([[-np.inf], s_edges[1:-1], [np.inf]])
    s_counts = np.histogram(p, bins=s_edges)[0].astype(float)

    summary = json.loads((M.ARTIFACTS_DIR.parent / "explain_v1" / "summary.json").read_text())

    ref = {
        "version": "monitor_v1",
        "model": f"{art.model_name}/{art.model_version}",
        "threshold": float(art.threshold),
        "decision_rule": str(art.decision_rule),
        "window": {"start": M.WINDOW_START, "end": M.WINDOW_END},
        "sources": {
            "train_parquet_md5": _md5(M.TRAIN_PARQUET),
            "valid_parquet_md5": _md5(M.VALID_PARQUET),
        },
        "monitored_features": monitored,
        "features": features,
        "score": {"edges": s_edges.tolist(),
                  "props": (s_counts / s_counts.sum()).tolist()},
        "operating_point": {"review_rate": review_rate, "precision": precision,
                            "recall": recall, "n": int(len(p)),
                            "flagged": int(flagged.sum())},
        "attribution": {"named_share": summary["named_share_two_block"],
                        "anon_share": summary["anon_share_two_block"],
                        "named_reason_coverage": summary["named_reason_coverage"]},
    }
    M.REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
    print(f"\nwrote {M.REFERENCE_PATH}  ({M.REFERENCE_PATH.stat().st_size / 1024:.1f} kB)")
    print("top 10 monitored:", monitored[:10])
    return ref


if __name__ == "__main__":
    build()