"""Chapter 11: raw payload -> the 290-column row the model expects.

Every stage is imported from src/features/ and runs in build_features' own order.
The single substitution is the entity-aggregate stage: Chapter 8 computed it with
vectorised prefix sums over the whole frame, which a service scoring one
transaction cannot do, so history.aggregates_for recovers the same 25 columns from
the store. The stateless transforms, the entity keys, the missingness block and the
fitted encoder are the training code, unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import config as fcfg
from src.features import entity, missingness, transforms
from src.serving.artifacts import Artifacts
from src.serving.history import NULL_SENTINEL, STORE_COLUMNS, aggregates_for
from src.serving.schemas import RawTransaction

NAMED_FIELDS = [f for f in RawTransaction.model_fields if f != "v"]
INT_FIELDS = {"transactionid", "transactiondt"}
FLOAT_FIELDS = {
    name
    for name, f in RawTransaction.model_fields.items()
    if name not in INT_FIELDS and name != "v" and "float" in str(f.annotation)
}
STR_FIELDS = set(NAMED_FIELDS) - INT_FIELDS - FLOAT_FIELDS


def _frames(txs: list[RawTransaction]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the two interim frames extract.py produced, for these rows only."""
    base = pd.DataFrame([{f: getattr(t, f) for f in NAMED_FIELDS} for t in txs])
    for c in INT_FIELDS:
        base[c] = base[c].astype("int64")
    for c in FLOAT_FIELDS:
        base[c] = pd.to_numeric(base[c], errors="coerce").astype("float32")
    for c in STR_FIELDS:
        base[c] = base[c].astype("object")

    vframe = pd.DataFrame([{fcfg.KEY: t.transactionid, **t.v} for t in txs])
    vframe[fcfg.KEY] = vframe[fcfg.KEY].astype("int64")
    for c in vframe.columns:
        if c != fcfg.KEY:
            vframe[c] = pd.to_numeric(vframe[c], errors="coerce").astype("float32")
    return base, vframe


def featurize(txs: list[RawTransaction], art: Artifacts, cur) ->  tuple[pd.DataFrame, pd.DataFrame]:
    """Return a frame of len(txs) rows and exactly art.feature_names columns."""
    base, vframe = _frames(txs)
    base = base.sort_values(fcfg.TIME, kind="mergesort").reset_index(drop=True)

    base = transforms.apply_stateless(base)
    base = entity.add_entity_keys(base)

    rows = [
        aggregates_for(
            cur,
            {
                "uid_card": r.uid_card,
                "uid_card_addr": r.uid_card_addr,
                "uid_account": r.uid_account,
            },
            int(r.transactiondt),
            float(r.transactionamt),
        )
        for r in base.itertuples()
    ]
    base = pd.concat(
        [base, pd.DataFrame(rows, index=base.index).astype("float32")], axis=1
    )

    base["productcd_key"] = base["productcd"].fillna(NULL_SENTINEL).astype(str)
    base["devicetype_key"] = base["devicetype"].fillna(NULL_SENTINEL).astype(str)
    hist = base[STORE_COLUMNS].copy()

    base = base.merge(missingness.v_missingness_frame(vframe), on=fcfg.KEY, how="left")
    base = base.merge(
        vframe[[fcfg.KEY] + list(art.encoder.v_keep_)], on=fcfg.KEY, how="left"
    )
    base = missingness.add_missingness_features(base)

    X = art.encoder.transform(base)[art.feature_names]
    X.index = base[fcfg.KEY].to_numpy()
    return X, hist