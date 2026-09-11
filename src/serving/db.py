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