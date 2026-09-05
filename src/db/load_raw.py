import os
from io import StringIO
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=os.getenv("POSTGRES_PORT"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
)
conn.autocommit = True


def load_table(csv_path: str, table_name: str, chunksize: int = 50000):
    """
    ELT Staging Pipeline
    --------------------
    Ingests large, high-dimensional datasets into PostgreSQL staging tables.
    
    Architectural choices:
    1. Chunking: Prevents Out-Of-Memory (OOM) crashes in WSL2/Docker resource limits.
    2. Pure TEXT Schema: Implements ELT pattern—stage raw data cleanly without type coercion errors.
    3. Low-level COPY API: Bypasses PostgreSQL's 65,535 SQL parameter limit for bulk writes.
    """
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    first = True
    total = 0
    with conn.cursor() as cur:
        for chunk in reader:
            chunk.columns = chunk.columns.str.lower()

            if first:
                cur.execute(f"DROP TABLE IF EXISTS {table_name};")
                cols_sql = ", ".join(f'"{c}" TEXT' for c in chunk.columns)
                cur.execute(f"CREATE TABLE {table_name} ({cols_sql});")
                first = False

            buf = StringIO()
            chunk.to_csv(buf, index=False, header=False)  # vectorized, fast
            buf.seek(0)
            cur.copy_expert(f"COPY {table_name} FROM STDIN WITH CSV", buf)
            total += len(chunk)
            print(f"{table_name}: {total} rows loaded")


if __name__ == "__main__":
    load_table("data/raw/train_transaction.csv", "raw_train_transaction")
    load_table("data/raw/train_identity.csv", "raw_train_identity")