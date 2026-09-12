"""Chapter 11: the scoring service.

Artifacts and the explainer are built once in the lifespan, never per request -
loading a 750-tree model and constructing a TreeExplainer costs seconds, and doing
it per call would dominate every latency observation.
"""

from __future__ import annotations

import pandas as pd
import os
import time
from contextlib import asynccontextmanager, contextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from psycopg2.pool import ThreadedConnectionPool

from src.serving import config as C
from src.serving.artifacts import load_artifacts
from src.serving.explain import explainer, reason_codes
from src.serving.featurize import featurize
from src.serving.persist import record, record_history
from src.serving.schemas import RawTransaction, ScoreResponse
from src.serving.score import score_frame

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    STATE["art"] = load_artifacts()
    explainer()
    STATE["pool"] = ThreadedConnectionPool(
        1, 8, host=C.PG_HOST, port=C.PG_PORT, dbname=C.PG_DB,
        user=C.PG_USER, password=C.PG_PASSWORD,
    )
    STATE["startup_s"] = time.time() - t0
    STATE["started_at"] = time.time()
    yield
    STATE["pool"].closeall()


app = FastAPI(title="fraud-detector", version="1", lifespan=lifespan)


@contextmanager
def _cursor():
    conn = STATE["pool"].getconn()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    finally:
        STATE["pool"].putconn(conn)


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("CH11_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
def health():
    art = STATE["art"]
    return {
        "status": "ok",
        "model": art.model_name,
        "version": art.model_version,
        "source": art.source,
        "n_features": art.n_features,
        "n_trees": art.booster.num_trees(),
        "threshold": art.threshold,
        "decision_rule": art.decision_rule,
        "explain_flagged_only": C.EXPLAIN_FLAGGED_ONLY,
        "startup_seconds": round(STATE["startup_s"], 2),
        "uptime_seconds": round(time.time() - STATE["started_at"], 1),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cpu_count": os.cpu_count(),
    }


def _score_many(
    txs: list[RawTransaction], persist: bool = True, store_features: bool = False
) -> list[ScoreResponse]:
    art = STATE["art"]
    t0 = time.perf_counter()
    with _cursor() as cur:
        X, hist = featurize(txs, art, cur)                      # <-- changed
        scored = score_frame(X, art)
        flagged = [s.transactionid for s in scored if s.flagged]
        if C.EXPLAIN_FLAGGED_ONLY:
            reasons = dict(zip(flagged, reason_codes(X.loc[flagged], art))) if flagged else {}
        else:
            reasons = dict(zip(X.index, reason_codes(X, art)))
        per_row = (time.perf_counter() - t0) * 1000.0 / max(len(scored), 1)
        if persist:
            feats = None
            if store_features:
                feats = {
                    int(tid): {
                        k: (None if pd.isna(v) else float(v))
                        for k, v in X.loc[tid].items()
                    }
                    for tid in X.index
                }
            record(cur, art, scored, reasons=reasons, features=feats,
                   latency_ms={s.transactionid: per_row for s in scored})
        record_history(cur, hist)                               # <-- added
    return [
        ScoreResponse(
            transactionid=s.transactionid,
            probability=s.probability,
            probability_raw=s.probability_raw,
            threshold=art.threshold,
            decision=s.decision,
            model_name=art.model_name,
            model_version=art.model_version,
            reason_codes=reasons.get(s.transactionid),
            latency_ms=round(per_row, 2),
        )
        for s in scored
    ]

@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_key)])
def score(tx: RawTransaction):
    return _score_many([tx])[0]


@app.post(
    "/score/batch",
    response_model=list[ScoreResponse],
    dependencies=[Depends(require_key)],
)
def score_batch(txs: list[RawTransaction], store_features: bool = False):
    if not txs or len(txs) > 500:
        raise HTTPException(status_code=422, detail="send 1 to 500 transactions")
    return _score_many(txs, store_features=store_features)