"""Chapter 12: the scheduled monitoring flow.

Chapter 11 shipped one flow with no schedule attached, deliberately. This is the
other half: the run that reads the decision log the scoring flow wrote and turns
it into a verdict.

The window is recomputed in full on each run rather than appended to. It costs
seconds, it is idempotent, and it means a late-arriving or re-scored day cannot
leave the monitoring tables in a state that depends on the order runs happened in.
"""

from __future__ import annotations

from prefect import flow, get_run_logger, task

from src.monitor import drift, signals, trigger


@task(retries=1, retry_delay_seconds=30)
def compute_drift() -> int:
    return len(drift.run(with_evidently=False))


@task(retries=1, retry_delay_seconds=30)
def compute_signals() -> int:
    return len(signals.run())


@task
def decide() -> dict:
    d = trigger.run()
    latest = d.iloc[-1]
    return {
        "tx_day": int(latest.tx_day),
        "decision": str(latest.decision),
        "reason": str(latest.reason),
        "review_rate": float(latest.review_rate),
        "psi_excess": float(latest.psi_excess),
    }


@flow(name="monitor-daily")
def monitor_daily() -> dict:
    log = get_run_logger()
    n_days = compute_drift()
    compute_signals()
    verdict = decide()
    log.info("monitored %d days; latest day %d -> %s (%s)",
             n_days, verdict["tx_day"], verdict["decision"], verdict["reason"])
    if verdict["decision"] == "retrain":
        log.warning("RETRAINING CANDIDATE: day %d, review rate %.4f, psi excess %.3f",
                    verdict["tx_day"], verdict["review_rate"], verdict["psi_excess"])
    return verdict


if __name__ == "__main__":
    monitor_daily()