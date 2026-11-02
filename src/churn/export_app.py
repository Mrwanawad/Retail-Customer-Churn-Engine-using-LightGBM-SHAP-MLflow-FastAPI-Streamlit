"""Copy the small artifacts the Streamlit demo needs into app/data (committed, a few MB)."""

from __future__ import annotations

import json
import shutil

import pandas as pd

from churn.config import ROOT, get_logger, load_config

log = get_logger(__name__)


def run() -> None:
    cfg = load_config()
    gold, rep = cfg.path("gold"), cfg.path("reports")
    out = ROOT / "app" / "data"
    out.mkdir(parents=True, exist_ok=True)

    snaps = pd.read_parquet(gold / "snapshots.parquet")
    latest = snaps[snaps.cutoff_day == snaps.cutoff_day.max()]
    latest.to_parquet(out / "latest_snapshot.parquet", index=False)

    label_cols = [c for c in snaps.columns if c.startswith("churn_")]
    trend = snaps.groupby(["cutoff_day", "snapshot_date"])[label_cols].mean().reset_index()
    trend["active_customers"] = snaps.groupby("cutoff_day").size().to_numpy()
    trend.to_parquet(out / "churn_trend.parquet", index=False)

    for f in ("churn_scores.parquet", "survival_by_segment.parquet", "survival_curves.parquet"):
        shutil.copy(gold / f, out / f)
    for f in ("lift.csv", "reliability.csv", "shap_importance.csv", "baselines.csv", "test_predictions.parquet",
              "cox_summary.csv", "survival_metrics.json"):
        shutil.copy(rep / "training" / f, out / f)
    for f in ("monitoring_summary.json", "feature_drift.csv", "realised_vs_predicted.csv", "evidently_drift.html"):
        src = rep / "monitoring" / f
        if src.exists():
            shutil.copy(src, out / f)
    shutil.copy(rep / "cleaning_report.json", out / "cleaning_report.json")
    shutil.copy(rep / "validation" / "raw_contracts.json", out / "raw_contracts.json")

    # identities stay out of the public demo: names/phones/emails are not needed by any page
    total = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    log.info("app data -> %s (%.1f MB)", out, total)
    (out / "manifest.json").write_text(json.dumps(sorted(p.name for p in out.iterdir()), indent=2))


if __name__ == "__main__":
    run()
