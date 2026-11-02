"""Run pipeline stages locally in DAG order. Airflow calls the same functions (see dags/).

python -m churn.pipeline weekly     extract -> validate -> clean -> features -> score -> monitor
python -m churn.pipeline retrain    features -> train -> score -> monitor -> export app data
python -m churn.pipeline all        simulate exports + weekly + retrain
"""

from __future__ import annotations

import argparse

from churn import export_app, features, ingest, monitor, score, train, validation
from churn.cleaning import run as cleaning
from churn.simulate import exports

WEEKLY = [
    ("extract", lambda: ingest.run()),
    ("validate", lambda: validation.run_raw_contracts(ingest.read_bronze)),
    ("clean", cleaning.run),
    ("features", features.build),
    ("score", score.run),
    ("monitor", monitor.run),
]
RETRAIN = [
    ("features", features.build),
    ("train", train.run),
    ("score", score.run),
    ("monitor", monitor.run),
    ("export_app", export_app.run),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("flow", choices=["weekly", "retrain", "all"])
    flow = ap.parse_args().flow
    if flow == "weekly":
        steps = WEEKLY
    elif flow == "retrain":
        steps = RETRAIN
    else:
        # first build: the model must exist before weekly scoring
        steps = [("simulate", exports.main), *WEEKLY[:4], *RETRAIN[1:]]
    for name, fn in steps:
        print(f"==> {name}", flush=True)
        fn()


if __name__ == "__main__":
    main()
