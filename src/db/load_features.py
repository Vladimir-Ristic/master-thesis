"""
Publish the Chapter 8 feature sets back into PostgreSQL as a typed features layer.

Chapter 7 established a staging layer of all-TEXT raw tables. This closes the
other end: a single ``features_v1`` table, properly typed, carrying a partition
label alongside the transaction key. Keeping the staging and feature layers
separate is what makes the pipeline re-runnable -- the raw tables are never
mutated, and rebuilding features is a matter of dropping and rewriting one table.

The table is what Chapter 9 reads for modelling and what Chapter 12's drift
monitoring compares against, so it is written once and treated as immutable for
a given feature version.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

from src.features import config as cfg
from src.features.extract import connect

TABLE = f"features_{cfg.FEATURE_VERSION}"
CHUNK_ROWS = 50_000


def _sql_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "bigint"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    return "double precision"


def load(replace: bool = True) -> None:
    conn = connect()
    conn.autocommit = True
    try:
        first = True
        with conn.cursor() as cur:
            for partition in ("train", "valid", "test"):
                path = cfg.DATA_PROCESSED / f"{partition}_{cfg.FEATURE_VERSION}.parquet"
                df = pd.read_parquet(path)
                df.insert(1, "partition", partition)

                if first:
                    if replace:
                        cur.execute(f"DROP TABLE IF EXISTS {TABLE};")
                    cols_sql = ", ".join(
                        f'"{c}" text' if c == "partition" else f'"{c}" {_sql_type(df[c].dtype)}'
                        for c in df.columns
                    )
                    cur.execute(f"CREATE TABLE {TABLE} ({cols_sql});")
                    first = False

                for start in range(0, len(df), CHUNK_ROWS):
                    chunk = df.iloc[start:start + CHUNK_ROWS]
                    buf = StringIO()
                    chunk.to_csv(buf, index=False, header=False)
                    buf.seek(0)
                    cur.copy_expert(f"COPY {TABLE} FROM STDIN WITH CSV", buf)
                print(f"{TABLE}: loaded {len(df):,} {partition} rows")
                del df

            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{TABLE}_partition ON {TABLE} ("partition");'
            )
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS idx_{TABLE}_key ON {TABLE} ("{cfg.KEY}");'
            )
            cur.execute(f"SELECT partition, count(*) FROM {TABLE} GROUP BY partition;")
            for row in cur.fetchall():
                print(f"  {row[0]}: {row[1]:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    load()
