"""Weekly batch scoring -> churn_scores, target list, survival by segment.

churn_scores columns: customer_id, churn_probability (calibrated, primary horizon), churn_prob_30d/60d/90d,
risk_segment, reason_1..3, clv, p_alive_bgnbd, expected_profit, targeted, scoring_date, model_version.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import polars as pl

from churn import decision, survival
from churn.clv import fit_clv
from churn.config import get_logger, load_config
from churn.explain import Explainer
from churn.features import load_silver

log = get_logger(__name__)


def load_bundle(cfg=None) -> dict:
    cfg = cfg or load_config()
    return joblib.load(cfg.path("models") / "production" / "churn_model.joblib")


def score_snapshot(snap: pd.DataFrame, bundle: dict, sales: pl.DataFrame, cfg) -> pd.DataFrame:
    meta = bundle["meta"]
    feats = meta["features_numeric"] + meta["features_categorical"]
    X = snap[feats]
    out = snap[["customer_id", "cutoff_day", "snapshot_date"]].copy()
    for h, m in bundle["models"].items():
        out[f"churn_prob_{h}d"] = m.predict_proba(X)[:, 1]
    out["churn_probability"] = out[f"churn_prob_{cfg.churn.primary_horizon}d"]
    out["risk_segment"] = decision.risk_segment(out.churn_probability.to_numpy()).to_numpy()

    reasons = Explainer(bundle["base_model"]).top_reasons(X, k=3)
    out = out.join(reasons)

    cutoff = int(snap.cutoff_day.iloc[0])
    clv = fit_clv(sales, cutoff, cfg.decision.clv_horizon_days, cfg.decision.margin)
    out = out.merge(clv[["clv", "p_alive_bgnbd"]], left_on="customer_id", right_index=True, how="left")
    out["clv"] = out.clv.fillna(0.0)

    d = cfg.decision
    out = decision.select_targets(out, d.offer_success_rate, d.offer_cost)
    out["scoring_date"] = snap.snapshot_date.iloc[0]
    out["model_version"] = meta["model_version"]
    return out.reset_index(drop=True)


def run(cutoff_day: int | None = None) -> pd.DataFrame:
    cfg = load_config()
    gold = cfg.path("gold")
    bundle = load_bundle(cfg)
    snaps = pd.read_parquet(gold / "snapshots.parquet")
    cutoff_day = cutoff_day or int(snaps.cutoff_day.max())
    snap = snaps[snaps.cutoff_day == cutoff_day].reset_index(drop=True)
    sales = load_silver(cfg)["sales"]

    scores = score_snapshot(snap, bundle, sales, cfg)
    scores.to_parquet(gold / "churn_scores.parquet", index=False)
    targets = scores[scores.targeted].drop(columns=[c for c in scores.columns if c.endswith(("_feature", "_shap"))])
    targets.to_csv(gold / "target_list.csv", index=False)
    log.info("scored %d customers at %s: %d targeted, expected profit $%.0f, segments %s",
             len(scores), scores.scoring_date.iloc[0], len(targets), targets.expected_profit.sum(),
             scores.risk_segment.value_counts().to_dict())

    # survival per segment, estimated on labelled history scored with the same model
    meta = bundle["meta"]
    feats = meta["features_numeric"] + meta["features_categorical"]
    hist = snaps[snaps.cutoff_day.isin(meta["split"]["test_cutoffs"] + meta["split"]["valid_cutoffs"])].copy()
    hist["risk_segment"] = decision.risk_segment(
        bundle["models"][cfg.churn.primary_horizon].predict_proba(hist[feats])[:, 1]
    ).to_numpy()
    seg_summary, curves = survival.km_by_segment(hist)
    train_hist = snaps[snaps.cutoff_day.isin(meta["split"]["train_cutoffs"])]
    _, c_index, cox_summary = survival.fit_cox(train_hist, hist)
    seg_summary.to_parquet(gold / "survival_by_segment.parquet", index=False)
    curves.to_parquet(gold / "survival_curves.parquet", index=False)
    cox_summary.to_csv(cfg.path("reports") / "training" / "cox_summary.csv", index=False)
    (cfg.path("reports") / "training" / "survival_metrics.json").write_text(json.dumps({"c_index_test": c_index}, indent=2))
    log.info("survival: Cox C-index %.3f\n%s", c_index, seg_summary.to_string(index=False))

    if os.environ.get("DATABASE_URL"):
        from sqlalchemy import create_engine

        cols = [c for c in scores.columns if not c.endswith(("_feature", "_shap"))]
        scores[cols].to_sql("churn_scores", create_engine(os.environ["DATABASE_URL"]), if_exists="append", index=False)
        log.info("appended to PostgreSQL churn_scores")
    return scores


if __name__ == "__main__":
    run()
