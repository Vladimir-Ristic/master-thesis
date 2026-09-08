"""
Extraction layer: raw staging (Chapter 7) -> typed, memory-bounded DataFrames.

Chapter 7 loaded both source files into PostgreSQL as all-TEXT staging tables,
deliberately deferring type coercion (ELT). This module performs that deferred
coercion in SQL, so the cast is expressed once, in the database, rather than
being re-derived by every downstream script.

The reader is source-agnostic: ``source="postgres"`` reads the staging tables,
``source="csv"`` reads the original files. The CSV path exists so the pipeline
remains executable on a machine without the database running; results are
identical because the cast rules are shared.

Extraction is split into two passes -- the base frame (named, C, D, M and
identity columns) and the V-column block -- because the full 434-column frame
does not fit comfortably in the 4 GB WSL2 allocation documented in Chapter 7.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import config as cfg

_NUMERIC_PREFIXES = ("c", "d", "v")


def _is_numeric_raw_column(name: str) -> bool:
    """Decide the cast for a raw staging column from its name alone."""
    if name in {cfg.KEY, cfg.TARGET, cfg.TIME}:
        return True
    if name in cfg.NAMED_NUMERIC:
        return True
    if name in cfg.IDENTITY_NUMERIC:
        return True
    if name[0] in _NUMERIC_PREFIXES and name[1:].isdigit():
        return True
    return False


def _cast_expression(name: str, alias: str | None = None) -> str:
    """SQL fragment casting one TEXT staging column to its analytical type."""
    ref = f'{alias}."{name}"' if alias else f'"{name}"'
    if name in {cfg.KEY, cfg.TIME}:
        cast = "bigint"
    elif name == cfg.TARGET:
        cast = "smallint"
    elif _is_numeric_raw_column(name):
        cast = "double precision"
    else:
        cast = None
    body = f"NULLIF({ref}::text, '')"
    if cast:
        body = f"{body}::{cast}"
    return f'{body} AS "{name}"'


# --------------------------------------------------------------------------
# PostgreSQL access
# --------------------------------------------------------------------------
def connect():
    """Open a connection using the same .env variables as src/db/load_raw.py."""
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def list_raw_columns(conn, table: str) -> list[str]:
    sql = (
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = %s ORDER BY ordinal_position"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        return [r[0] for r in cur.fetchall()]


def _read_sql_chunked(conn, sql: str, columns: Sequence[str], chunksize: int = 100_000):
    """Stream a query through a server-side cursor to bound peak memory."""
    frames = []
    with conn.cursor(name="ch8_extract") as cur:
        cur.itersize = chunksize
        cur.execute(sql)
        while True:
            rows = cur.fetchmany(chunksize)
            if not rows:
                break
            frames.append(pd.DataFrame(rows, columns=list(columns)))
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Public extraction API
# --------------------------------------------------------------------------
def _select_sql(table: str, columns: Sequence[str]) -> str:
    projection = ",\n       ".join(_cast_expression(c) for c in columns)
    return f"SELECT {projection}\nFROM {table}"


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in {cfg.KEY, cfg.TIME}:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
        elif col == cfg.TARGET:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int8")
        elif _is_numeric_raw_column(col):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        else:
            df[col] = df[col].astype("object")
    return df


def _split_column_names(tx_cols: Iterable[str]) -> tuple[list[str], list[str]]:
    tx_cols = list(tx_cols)
    v_cols = sorted(
        (c for c in tx_cols if c.startswith("v") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    base_cols = [c for c in tx_cols if c not in set(v_cols)]
    return base_cols, v_cols


def extract(source: str = "postgres") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return ``(base, vcols)``.

    ``base`` holds every non-V transaction column left-joined to the identity
    table; ``vcols`` holds the transaction key plus the anonymised V block.
    The left join preserves all 590,540 transactions, so the 75.6% of rows with
    no identity record become explicit nulls rather than disappearing.
    """
    if source == "postgres":
        return _extract_postgres()
    if source == "csv":
        return _extract_csv()
    raise ValueError(f"unknown source: {source!r}")


def _extract_postgres() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = connect()
    try:
        tx_cols = list_raw_columns(conn, cfg.RAW_TX_TABLE)
        id_cols = list_raw_columns(conn, cfg.RAW_ID_TABLE)
        if not tx_cols:
            raise RuntimeError(f"staging table {cfg.RAW_TX_TABLE} not found or empty")
        base_cols, v_cols = _split_column_names(tx_cols)
        id_join_cols = [c for c in id_cols if c != cfg.KEY]

        tx_proj = ",\n       ".join(_cast_expression(c, "t") for c in base_cols)
        id_proj = ",\n       ".join(_cast_expression(c, "i") for c in id_join_cols)
        # Both keys are TEXT and originate from the same integer column, so the
        # join can use the index created in Chapter 7 instead of casting first.
        sql = (
            f"SELECT {tx_proj},\n       {id_proj}\n"
            f"FROM {cfg.RAW_TX_TABLE} t\n"
            f"LEFT JOIN {cfg.RAW_ID_TABLE} i\n"
            f'  ON t."{cfg.KEY}" = i."{cfg.KEY}"'
        )
        base = _read_sql_chunked(conn, sql, base_cols + id_join_cols)
        base = _downcast(base)

        v_sql = _select_sql(cfg.RAW_TX_TABLE, [cfg.KEY] + v_cols)
        vdf = _read_sql_chunked(conn, v_sql, [cfg.KEY] + v_cols)
        vdf = _downcast(vdf)
    finally:
        conn.close()
    return base, vdf


def _extract_csv() -> tuple[pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(cfg.RAW_TX_CSV, nrows=0)
    header.columns = header.columns.str.lower()
    base_cols, v_cols = _split_column_names(header.columns)

    tx = pd.read_csv(cfg.RAW_TX_CSV, usecols=None, low_memory=False)
    tx.columns = tx.columns.str.lower()
    idn = pd.read_csv(cfg.RAW_ID_CSV, low_memory=False)
    idn.columns = idn.columns.str.lower()

    vdf = _downcast(tx[[cfg.KEY] + v_cols].copy())
    base = tx[base_cols].merge(idn, on=cfg.KEY, how="left")
    del tx
    base = _downcast(base)
    return base, vdf


def write_interim(base: pd.DataFrame, vdf: pd.DataFrame) -> None:
    cfg.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    base.to_parquet(cfg.DATA_INTERIM / "base.parquet", index=False)
    vdf.to_parquet(cfg.DATA_INTERIM / "vcols.parquet", index=False)


def read_interim() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(cfg.DATA_INTERIM / "base.parquet")
    vdf = pd.read_parquet(cfg.DATA_INTERIM / "vcols.parquet")
    return base, vdf


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chapter 8 extraction stage")
    parser.add_argument("--source", choices=["postgres", "csv"], default="postgres")
    args = parser.parse_args()

    base, vdf = extract(args.source)
    write_interim(base, vdf)
    print(f"base:   {base.shape[0]:,} rows x {base.shape[1]} cols")
    print(f"vcols:  {vdf.shape[0]:,} rows x {vdf.shape[1]} cols")
    print(f"written to {cfg.DATA_INTERIM}")


if __name__ == "__main__":
    main()
