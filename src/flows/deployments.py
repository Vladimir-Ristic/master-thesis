"""Chapter 12: the schedules.

Scoring runs at 01:00 and monitoring at 02:00, in that order and on that gap,
because the monitor reads what the scoring run wrote. The separation is the point:
the model's decisions and the judgement of those decisions are different jobs with
different failure modes, and coupling them into one flow would mean a monitoring
fault could stop transactions being scored.
"""

from __future__ import annotations

from prefect import serve

from src.flows.monitor_daily import monitor_daily
from src.flows.score_batch import score_day


def main() -> None:
    scoring = score_day.to_deployment(
        name="score-batch-daily",
        cron="0 1 * * *",
        parameters={"day": 147},
        tags=["ch12", "scoring"],
        description="Scores one day of transactions through the Chapter 11 service.",
    )
    monitoring = monitor_daily.to_deployment(
        name="monitor-daily",
        cron="0 2 * * *",
        tags=["ch12", "monitoring"],
        description="Drift, attribution and delayed-label signals; emits a retraining verdict.",
    )
    serve(scoring, monitoring)


if __name__ == "__main__":
    main()