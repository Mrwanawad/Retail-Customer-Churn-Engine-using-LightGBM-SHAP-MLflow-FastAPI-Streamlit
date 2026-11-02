"""Airflow DAGs (Airflow 2.x/3.x TaskFlow).

churn_weekly_scoring   Mondays 05:00: extract -> validate -> clean -> features -> score -> monitor
                       monitor pushes `retrain`; when true it triggers churn_monthly_retrain early.
churn_monthly_retrain  1st of month 03:00: features -> train -> score -> monitor -> export app data

Each task calls the same function the local runner uses (churn.pipeline), so dev and prod run one code path.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DEFAULTS = {"owner": "data-science", "retries": 2, "retry_delay": timedelta(minutes=10)}


@dag(schedule="0 5 * * 1", start_date=datetime(2025, 1, 6), catchup=False, default_args=DEFAULTS,
     tags=["churn"], max_active_runs=1)
def churn_weekly_scoring():
    @task
    def extract(ds=None):
        from churn import ingest

        return ingest.run(source="postgres", extract_date=ds)

    @task
    def validate():
        from churn import ingest, validation

        validation.run_raw_contracts(ingest.read_bronze)

    @task
    def clean():
        from churn.cleaning import run

        run.run()

    @task
    def build_features():
        from churn import features

        features.build()

    @task
    def score():
        from churn import score as s

        s.run()

    @task
    def monitor() -> bool:
        from churn import monitor as m

        return m.run()["retrain"]

    @task.short_circuit
    def needs_retrain(flag: bool) -> bool:
        return flag

    trigger = TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="churn_monthly_retrain")
    flag = monitor()
    extract() >> validate() >> clean() >> build_features() >> score() >> flag
    needs_retrain(flag) >> trigger


@dag(schedule="0 3 1 * *", start_date=datetime(2025, 1, 1), catchup=False, default_args=DEFAULTS,
     tags=["churn"], max_active_runs=1)
def churn_monthly_retrain():
    @task
    def build_features():
        from churn import features

        features.build()

    @task(execution_timeout=timedelta(hours=2))
    def train():
        from churn import train as t

        return t.run()["model_version"]

    @task
    def score():
        from churn import score as s

        s.run()

    @task
    def monitor():
        from churn import monitor as m

        m.run()

    @task
    def export_app_data():
        from churn import export_app

        export_app.run()

    build_features() >> train() >> score() >> monitor() >> export_app_data()


churn_weekly_scoring()
churn_monthly_retrain()
