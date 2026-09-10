# Tracking server image.
#
# The published MLflow image carries no PostgreSQL driver, so a backend store
# on Postgres needs one added. Building a pinned image rather than pip
# installing at container start means the service starts offline, starts fast,
# and is reproducible — which is the same argument Chapter 5 makes for
# containerising the rest of the stack.
FROM ghcr.io/mlflow/mlflow:v3.1.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.9

EXPOSE 5000
