"""Training pipeline.

out-of-time split -> baseline zoo (5 model families x imbalance strategies) -> Optuna on LightGBM
(StratifiedGroupKFold by customer inside the training window) -> probability calibration on the
validation snapshots -> test evaluation -> SHAP -> MLflow tracking + registry -> export to models/production.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from churn import metrics as M
from churn.config import ROOT, get_logger, load_config
from churn.modeling import feature_columns, make_model, out_of_time_split, positive_weight

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASELINES = [
    ("logreg", "weight"),
    ("random_forest", "weight"),
    ("lightgbm", "none"),
    ("lightgbm", "weight"),
    ("lightgbm", "under"),
    ("lightgbm", "smotenc"),
    ("xgboost", "weight"),
    ("catboost", "weight"),
]


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("retail-churn")


def run_baselines(split, num, cat, label, seed) -> pd.DataFrame:
    X, y = split.train[num + cat], split.train[label].to_numpy()
    Xv, yv = split.valid[num + cat], split.valid[label].to_numpy()
    pw = positive_weight(y)
    rows = []
    for name, imb in BASELINES:
        with mlflow.start_run(run_name=f"baseline-{name}-{imb}", nested=True):
            model = make_model(name, num, cat, imbalance=imb, pos_weight=pw, seed=seed).fit(X, y)
            p = model.predict_proba(Xv)[:, 1]
            r = M.classification_report(yv, p)
            mlflow.log_params({"model": name, "imbalance": imb})
            mlflow.log_metrics({f"valid_{k}": v for k, v in r.items()})
            rows.append({"model": name, "imbalance": imb, **r})
            log.info("baseline %-14s %-8s PR-AUC %.4f  ROC-AUC %.4f  recall@10%% %.3f", name, imb, r["pr_auc"], r["roc_auc"], r["recall_at_10pct"])
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def tune_lightgbm(split, num, cat, label, trials, folds, seed) -> dict:
    X, y = split.train[num + cat], split.train[label].to_numpy()
    groups = split.train.customer_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_idx = list(cv.split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        }
        imb = trial.suggest_categorical("imbalance", ["none", "weight"])
        scores = []
        for i, (tr, va) in enumerate(fold_idx):
            m = make_model("lightgbm", num, cat, params, imbalance=imb, pos_weight=positive_weight(y[tr]), seed=seed)
            m.fit(X.iloc[tr], y[tr])
            scores.append(average_precision_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    log.info("optuna best CV PR-AUC %.4f params %s", study.best_value, study.best_params)
    mlflow.log_metric("cv_pr_auc_best", study.best_value)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    return study.best_params


def calibrate(model, Xv: pd.DataFrame, yv: np.ndarray, seed: int):
    """Pick sigmoid vs isotonic on a split of the validation snapshots, then refit on all of it."""
    rng = np.random.default_rng(seed)
    a = rng.random(len(yv)) < 0.5
    best, best_brier = "sigmoid", np.inf
    for method in ("sigmoid", "isotonic"):
        c = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(Xv[a], yv[a])
        b = brier_score_loss(yv[~a], c.predict_proba(Xv[~a])[:, 1])
        log.info("calibration %-8s brier %.5f", method, b)
        if b < best_brier:
            best, best_brier = method, b
    return CalibratedClassifierCV(FrozenEstimator(model), method=best).fit(Xv, yv), best


def shap_global(model, X: pd.DataFrame, n: int = 3000, seed: int = 42) -> pd.DataFrame:
    pre, est = model.named_steps["pre"], model.named_steps["model"]
    Xs = X.sample(min(n, len(X)), random_state=seed)
    Xt = pd.DataFrame(pre.transform(Xs), columns=pre.get_feature_names_out())
    sv = shap.TreeExplainer(est).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv
    return (
        pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(sv).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def profit_threshold(p: np.ndarray, clv: np.ndarray, cfg) -> float:
    """Target when expected profit > 0: p * clv * success - cost > 0."""
    d = cfg.decision
    ev = p * clv * d.offer_success_rate - d.offer_cost
    return float(p[ev > 0].min()) if (ev > 0).any() else 1.0


def run() -> dict:
    cfg = load_config()
    setup_mlflow()
    seed, ch, sp = cfg.seed, cfg.churn, cfg.split
    snap = pd.read_parquet(cfg.path("gold") / "snapshots.parquet")
    num, cat = feature_columns(snap)
    label = f"churn_{ch.primary_horizon}d"
    split = out_of_time_split(snap, label, ch.primary_horizon, sp.valid_cutoffs, sp.test_cutoffs)
    log.info("split train=%d (%d cutoffs) valid=%d test=%d score=%d | churn rate train %.3f test %.3f",
             len(split.train), split.train.cutoff_day.nunique(), len(split.valid), len(split.test), len(split.score),
             split.train[label].mean(), split.test[label].mean())

    out_dir = cfg.path("models") / "production"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = cfg.path("reports") / "training"
    rep_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"train-{label}") as run:
        mlflow.log_params({
            "label": label, "n_features": len(num) + len(cat), "train_rows": len(split.train),
            "train_cutoffs": str(sorted(split.train.cutoff_day.unique())),
            "valid_cutoffs": str(sorted(split.valid.cutoff_day.unique())),
            "test_cutoffs": str(sorted(split.test.cutoff_day.unique())),
        })
        base = run_baselines(split, num, cat, label, seed)
        base.to_csv(rep_dir / "baselines.csv", index=False)
        mlflow.log_artifact(rep_dir / "baselines.csv")

        best = tune_lightgbm(split, num, cat, label, cfg.training.optuna_trials, cfg.training.cv_folds, seed)
        imb = best.pop("imbalance")
        y = split.train[label].to_numpy()
        yv = split.valid[label].to_numpy()

        # final pick on the validation snapshots: tuned LightGBM vs the best tree baseline
        # (tree models only, so SHAP TreeExplainer can explain every production score)
        tuned = make_model("lightgbm", num, cat, best, imbalance=imb, pos_weight=positive_weight(y), seed=seed)
        tuned.fit(split.train[num + cat], y)
        tuned_pr = average_precision_score(yv, tuned.predict_proba(split.valid[num + cat])[:, 1])
        trees = base[base.model.isin(["random_forest", "lightgbm", "xgboost", "catboost"])].iloc[0]
        log.info("selection: tuned lightgbm valid PR-AUC %.4f vs %s/%s %.4f", tuned_pr, trees.model, trees.imbalance, trees.pr_auc)
        if trees.pr_auc > tuned_pr:
            family, imb, best = trees.model, trees.imbalance, {}
            model = make_model(family, num, cat, imbalance=imb, pos_weight=positive_weight(y), seed=seed)
            model.fit(split.train[num + cat], y)
        else:
            family, model = "lightgbm", tuned
        mlflow.log_params({"final_family": family, "final_imbalance": imb})
        mlflow.log_metrics({"valid_pr_auc_tuned_lgbm": tuned_pr, "valid_pr_auc_best_tree_baseline": float(trees.pr_auc)})

        calibrated, method = calibrate(model, split.valid[num + cat], yv, seed)

        Xt, yt = split.test[num + cat], split.test[label].to_numpy()
        p_raw = model.predict_proba(Xt)[:, 1]
        p = calibrated.predict_proba(Xt)[:, 1]
        # threshold from the decision rule with a spend-based CLV proxy (full CLV lives in scoring)
        clv_proxy = split.test.spend_180d.to_numpy() * cfg.decision.margin
        thr = profit_threshold(p, clv_proxy, cfg)
        test_metrics = M.classification_report(yt, p, threshold=thr)
        test_metrics["pr_auc_uncalibrated"] = float(average_precision_score(yt, p_raw))
        test_metrics["brier_uncalibrated"] = float(brier_score_loss(yt, p_raw))
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log.info("TEST %s", json.dumps({k: round(v, 4) for k, v in test_metrics.items()}))

        lift = M.lift_table(yt, p)
        rel = M.reliability(yt, p)
        rel_raw = M.reliability(yt, p_raw).rename(columns={"observed_rate": "observed_rate_uncal", "mean_predicted": "mean_predicted_uncal"})
        imp = shap_global(model, split.test[num + cat], seed=seed)
        lift.to_csv(rep_dir / "lift.csv", index=False)
        pd.concat([rel, rel_raw], axis=1).to_csv(rep_dir / "reliability.csv", index=False)
        imp.to_csv(rep_dir / "shap_importance.csv", index=False)
        preds = split.test[["customer_id", "cutoff_day", label, "spend_180d"]].assign(p_churn=p, p_raw=p_raw)
        preds.to_parquet(rep_dir / "test_predictions.parquet", index=False)
        for f in ("lift.csv", "reliability.csv", "shap_importance.csv"):
            mlflow.log_artifact(rep_dir / f)

        # secondary horizons reuse the tuned params
        horizon_models = {}
        for h in ch.horizons:
            if h == ch.primary_horizon:
                horizon_models[h] = calibrated
                continue
            lab = f"churn_{h}d"
            s_h = out_of_time_split(snap, lab, h, sp.valid_cutoffs, sp.test_cutoffs)
            m_h = make_model(family, num, cat, best, imbalance=imb, pos_weight=positive_weight(s_h.train[lab]), seed=seed)
            m_h.fit(s_h.train[num + cat], s_h.train[lab])
            c_h = CalibratedClassifierCV(FrozenEstimator(m_h), method=method).fit(s_h.valid[num + cat], s_h.valid[lab])
            r_h = M.classification_report(s_h.test[lab].to_numpy(), c_h.predict_proba(s_h.test[num + cat])[:, 1])
            mlflow.log_metrics({f"test_{lab}_{k}": v for k, v in r_h.items()})
            log.info("horizon %dd test PR-AUC %.4f (base rate %.3f)", h, r_h["pr_auc"], r_h["base_rate"])
            horizon_models[h] = c_h

        sig = infer_signature(Xt.head(50), p[:50])
        info = mlflow.sklearn.log_model(calibrated, name="model", signature=sig, input_example=Xt.head(3),
                                        serialization_format="cloudpickle",
                                        registered_model_name=f"churn-{ch.primary_horizon}d")
        version = f"{family}-{label}-{datetime.now(UTC):%Y%m%d}-{run.info.run_id[:8]}"
        meta = {
            "model_version": version,
            "mlflow_run_id": run.info.run_id,
            "registered_model": f"churn-{ch.primary_horizon}d",
            "registered_version": getattr(info, "registered_model_version", None),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label": label,
            "features_numeric": num,
            "features_categorical": cat,
            "model_family": family,
            "best_params": best,
            "imbalance": imb,
            "calibration": method,
            "threshold": thr,
            "test_metrics": test_metrics,
            "baselines": base.to_dict(orient="records"),
            "split": {
                "train_cutoffs": [int(c) for c in sorted(split.train.cutoff_day.unique())],
                "valid_cutoffs": [int(c) for c in sorted(split.valid.cutoff_day.unique())],
                "test_cutoffs": [int(c) for c in sorted(split.test.cutoff_day.unique())],
            },
        }
        joblib.dump({"models": horizon_models, "base_model": model, "meta": meta}, out_dir / "churn_model.joblib", compress=3)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        mlflow.log_artifact(out_dir / "metadata.json")
        log.info("exported %s -> %s", version, out_dir)
    return meta


if __name__ == "__main__":
    run()
