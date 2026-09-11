"""Chapter 11 request and response contract.

The payload mirrors the raw IEEE-CIS record, not the engineered matrix: the service
reproduces training-time features rather than trusting a client to supply them. Its
shape follows the thesis's own division - the named business fields are declared and
documented, while the 339 anonymised columns arrive as one opaque map, because no
schema can document what the data provider withheld.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

V_COLUMNS = tuple(f"v{i}" for i in range(1, 340))


class RawTransaction(BaseModel):
    """One transaction as the source system holds it."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # --- required: the key, the clock, and the amount --------------------
    transactionid: int
    transactiondt: int = Field(description="seconds from the dataset's undisclosed epoch")
    transactionamt: float

    # --- named business fields (all optional; absence is a trained state) -
    productcd: str | None = None
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    addr2: float | None = None
    dist1: float | None = None
    dist2: float | None = None
    p_emaildomain: str | None = None
    r_emaildomain: str | None = None

    c1: float | None = None
    c2: float | None = None
    c3: float | None = None
    c4: float | None = None
    c5: float | None = None
    c6: float | None = None
    c7: float | None = None
    c8: float | None = None
    c9: float | None = None
    c10: float | None = None
    c11: float | None = None
    c12: float | None = None
    c13: float | None = None
    c14: float | None = None

    d1: float | None = None
    d2: float | None = None
    d3: float | None = None
    d4: float | None = None
    d5: float | None = None
    d6: float | None = None
    d7: float | None = None
    d8: float | None = None
    d9: float | None = None
    d10: float | None = None
    d11: float | None = None
    d12: float | None = None
    d13: float | None = None
    d14: float | None = None
    d15: float | None = None

    m1: str | None = None
    m2: str | None = None
    m3: str | None = None
    m4: str | None = None
    m5: str | None = None
    m6: str | None = None
    m7: str | None = None
    m8: str | None = None
    m9: str | None = None

    id_01: float | None = None
    id_02: float | None = None
    id_03: float | None = None
    id_04: float | None = None
    id_05: float | None = None
    id_06: float | None = None
    id_07: float | None = None
    id_08: float | None = None
    id_09: float | None = None
    id_10: float | None = None
    id_11: float | None = None
    id_12: str | None = None
    id_13: str | None = None
    id_14: str | None = None
    id_15: str | None = None
    id_16: str | None = None
    id_17: str | None = None
    id_18: str | None = None
    id_19: str | None = None
    id_20: str | None = None
    id_21: str | None = None
    id_22: str | None = None
    id_23: str | None = None
    id_24: str | None = None
    id_25: str | None = None
    id_26: str | None = None
    id_27: str | None = None
    id_28: str | None = None
    id_29: str | None = None
    id_30: str | None = None
    id_31: str | None = None
    id_32: str | None = None
    id_33: str | None = None
    id_34: str | None = None
    id_35: str | None = None
    id_36: str | None = None
    id_37: str | None = None
    id_38: str | None = None

    devicetype: str | None = None
    deviceinfo: str | None = None

    # --- the anonymised block -------------------------------------------
    v: dict[str, float | None] = Field(
        description="v1 through v339; every key required, values nullable"
    )

    @model_validator(mode="before")
    @classmethod
    def _lowercase_keys(cls, data: Any) -> Any:
        """Trap 7: clients send TransactionAmt / ProductCD / DeviceType."""
        if isinstance(data, dict):
            return {(k.lower() if isinstance(k, str) else k): val for k, val in data.items()}
        return data

    @field_validator("v")
    @classmethod
    def _v_block_must_be_complete(cls, value: dict[str, float | None]):
        got = {k.lower() for k in value}
        expected = set(V_COLUMNS)
        if got != expected:
            missing = sorted(expected - got)[:5]
            unknown = sorted(got - expected)[:5]
            raise ValueError(
                f"v must hold exactly 339 keys v1..v339; missing e.g. {missing}, "
                f"unexpected e.g. {unknown}"
            )
        return {k.lower(): v for k, v in value.items()}


class ReasonCode(BaseModel):
    """Section 10.6. Anonymised contributors are rendered as their group."""

    feature: str
    value: float | str | None
    contribution: float = Field(description="contribution to the margin, in log-odds")


class ScoreResponse(BaseModel):
    transactionid: int
    probability: float = Field(description="Platt-calibrated")
    probability_raw: float
    threshold: float
    decision: Literal["review", "accept"]
    model_name: str
    model_version: str
    reason_codes: list[ReasonCode] | None = Field(
        default=None,
        description="present only for transactions the threshold sends to review",
    )
    latency_ms: float