"""Postgres connections for the serving layer."""

from __future__ import annotations

import psycopg2

from src.serving import config as C


def connect():
    if not C.PG_DB or not C.PG_USER:
        raise RuntimeError("database credentials missing; check .env")
    return psycopg2.connect(
        host=C.PG_HOST, port=C.PG_PORT, dbname=C.PG_DB,
        user=C.PG_USER, password=C.PG_PASSWORD,
    )

from pathlib import Path

from src.features import config as fcfg

DDL_PATH = Path(fcfg.PROJECT_ROOT) / "src" / "db" / "create_serving_tables.sql"


def create_tables() -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(DDL_PATH.read_text())
        conn.commit()
    print("serving tables created")