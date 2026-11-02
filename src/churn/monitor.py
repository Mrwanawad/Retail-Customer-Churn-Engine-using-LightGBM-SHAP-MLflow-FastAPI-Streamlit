"""Production monitoring.

  input drift       PSI per feature, latest snapshot vs the training window (+ Evidently HTML report)
  prediction drift  PSI of churn probabilities, latest vs test-period scores
  realised vs pred  once labels mature (primary_horizon days later): calibration gap and PR-AUC per cutoff
  uplift            churn of treated vs random control holdout (when campaign outcomes exist)
  retrain trigger   PR-AUC drop beyond tolerance, or drift on too many features / on predictions
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from churn.config import get_logger, load_config
from churn.score import load_bundle

log = get_logger(__name__)

PSI_ALERT = 0.2
MAX_DRIFTED_SHARE = 0.3
PR_AUC_TOLERANCE = 0.05


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(ref[c].to_numpy(float), cur[c].to_numpy(float))} for c in cols]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out["drifted"] = out.psi > PSI_ALERT
    return out


def realised_vs_predicted(snaps: pd.DataFrame, bundle: dict, label: str, after: int) -> pd.DataFrame:
    meta = bundle["meta"]
    feats = meta["features_numeric"] + meta["features_categorical"]
    model = bundle["models"][int(label.split("_")[1].rstrip("d"))]
    rows = []
    for c, g in snaps[(snaps.cutoff_day >= after) & snaps[label].notna()].groupby("cutoff_day"):
        p = model.predict_proba(g[feats])[:, 1]
        y = g[label].to_numpy()
        rows.append({
            "cutoff_day": int(c), "snapshot_date": str(g.snapshot_date.iloc[0])[:10], "customers": len(g),
            "predicted_rate": float(p.mean()), "realised_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        })
    return pd.DataFrame(rows)


def uplift(scores: pd.DataFrame, outcome_col: str) -> dict:
    """Treated vs control churn among targeted customers. Needs outcomes joined after the horizon."""
    t = scores[scores.targeted & ~scores.holdout][outcome_col]
    c = scores[scores.targeted & scores.holdout][outcome_col]
    if len(t) == 0 or len(c) == 0:
        return {"status": "no outcomes yet"}
    return {"treated_churn": float(t.mean()), "control_churn": float(c.mean()), "uplift": float(c.mean() - t.mean()),
            "n_treated": int(len(t)), "n_control": int(len(c))}


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, path) -> bool:
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        # declare every feature numeric; otherwise low-cardinality counts are inferred as categorical or skipped
        schema = DataDefinition(numerical_columns=list(ref.columns))
        ds = lambda d: Dataset.from_pandas(d.astype(float), data_definition=schema)  # noqa: E731
        Report([DataDriftPreset()]).run(current_data=ds(cur), reference_data=ds(ref)).save_html(str(path))
        return True
    except Exception as e:  # report is a nice-to-have; PSI drives the trigger
        log.warning("Evidently report skipped: %s", e)
        return False


def run() -> dict:
    cfg = load_config()
    out_dir = cfg.path("reports") / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(cfg)
    meta = bundle["meta"]
    num = meta["features_numeric"]
    label = meta["label"]
    snaps = pd.read_parquet(cfg.path("gold") / "snapshots.parquet")
    scores = pd.read_parquet(cfg.path("gold") / "churn_scores.parquet")

    ref = snaps[snaps.cutoff_day.isin(meta["split"]["train_cutoffs"])]
    cur = snaps[snaps.cutoff_day == snaps.cutoff_day.max()]
    drift = feature_drift(ref, cur, num)
    drift.to_csv(out_dir / "feature_drift.csv", index=False)

    test_preds = pd.read_parquet(cfg.path("reports") / "training" / "test_predictions.parquet")
    pred_psi = psi(test_preds.p_churn.to_numpy(), scores.churn_probability.to_numpy())

    rvp = realised_vs_predicted(snaps, bundle, label, after=min(meta["split"]["test_cutoffs"]))
    rvp.to_csv(out_dir / "realised_vs_predicted.csv", index=False)
    baseline_pr = meta["test_metrics"]["pr_auc"]
    latest_pr = float(rvp.pr_auc.iloc[-1]) if len(rvp) else float("nan")

    reasons = []
    if not np.isnan(latest_pr) and latest_pr < baseline_pr - PR_AUC_TOLERANCE:
        reasons.append(f"PR-AUC {latest_pr:.3f} < baseline {baseline_pr:.3f} - {PR_AUC_TOLERANCE}")
    if drift.drifted.mean() > MAX_DRIFTED_SHARE:
        reasons.append(f"{drift.drifted.mean():.0%} of features drifted (PSI > {PSI_ALERT})")
    if pred_psi > PSI_ALERT:
        reasons.append(f"prediction PSI {pred_psi:.3f} > {PSI_ALERT}")

    html = evidently_report(ref[num].sample(min(5000, len(ref)), random_state=cfg.seed), cur[num], out_dir / "evidently_drift.html")
    summary = {
        "reference_cutoffs": meta["split"]["train_cutoffs"],
        "current_cutoff": int(cur.cutoff_day.iloc[0]),
        "features_drifted": int(drift.drifted.sum()),
        "features_checked": len(drift),
        "top_drift": drift.head(5).to_dict(orient="records"),
        "prediction_psi": pred_psi,
        "baseline_pr_auc": baseline_pr,
        "latest_matured_pr_auc": latest_pr,
        "uplift": uplift(scores, "churn_60d") if "churn_60d" in scores else {"status": "no outcomes yet"},
        "retrain": bool(reasons),
        "retrain_reasons": reasons,
        "evidently_report": "evidently_drift.html" if html else None,
    }
    (out_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("monitoring: %d/%d features drifted, prediction PSI %.3f, matured PR-AUC %.3f (baseline %.3f) -> retrain=%s %s",
             summary["features_drifted"], len(drift), pred_psi, latest_pr, baseline_pr, summary["retrain"], reasons)
    return summary


if __name__ == "__main__":
    run()
