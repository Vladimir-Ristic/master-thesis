"""Chapter 11 fast loop: the request contract."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.serving.schemas import RawTransaction, V_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PARQUET = PROJECT_ROOT / "data" / "interim" / "base.parquet"


def _payload(**overrides):
    p = {
        "transactionid": 1,
        "transactiondt": 86400,
        "transactionamt": 59.0,
        "v": {c: None for c in V_COLUMNS},
    }
    p.update(overrides)
    return p


def test_camel_case_keys_from_the_source_system_are_accepted():
    t = RawTransaction(
        **{
            "TransactionID": 7,
            "TransactionDT": 1,
            "TransactionAmt": 1.0,
            "ProductCD": "W",
            "DeviceType": "mobile",
            "v": {c: None for c in V_COLUMNS},
        }
    )
    assert t.transactionid == 7
    assert t.productcd == "W"
    assert t.devicetype == "mobile"


def test_an_absent_named_field_is_none_and_never_zero():
    t = RawTransaction(**_payload())
    for name in ("card1", "c13", "d1", "id_01", "dist1"):
        assert getattr(t, name) is None


def test_an_explicit_null_is_preserved_as_none():
    t = RawTransaction(**_payload(card1=None, c13=None))
    assert t.card1 is None and t.c13 is None


def test_a_required_field_cannot_be_omitted():
    p = _payload()
    p.pop("transactionamt")
    with pytest.raises(ValidationError):
        RawTransaction(**p)


def test_an_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        RawTransaction(**_payload(nonsense=1))


def test_an_incomplete_v_block_is_rejected():
    v = {c: None for c in V_COLUMNS}
    v.pop("v200")
    with pytest.raises(ValidationError):
        RawTransaction(**_payload(v=v))


def test_an_unexpected_v_key_is_rejected():
    v = {c: None for c in V_COLUMNS}
    v["v340"] = None
    with pytest.raises(ValidationError):
        RawTransaction(**_payload(v=v))


@pytest.mark.skipif(not BASE_PARQUET.exists(), reason="interim layer not present")
def test_the_schema_declares_exactly_the_raw_columns():
    import pyarrow.parquet as pq

    raw = set(pq.ParquetFile(BASE_PARQUET).schema_arrow.names) - {"isfraud"}
    declared = set(RawTransaction.model_fields) - {"v"}
    assert declared == raw, (
        f"missing from schema: {sorted(raw - declared)} | "
        f"declared but not in raw data: {sorted(declared - raw)}"
    )