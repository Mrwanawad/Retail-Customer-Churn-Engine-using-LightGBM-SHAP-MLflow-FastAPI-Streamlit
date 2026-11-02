"""Model and business metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


def top_k_mask(p: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(len(p) * frac)))
    idx = np.argsort(-p, kind="stable")[:k]
    m = np.zeros(len(p), bool)
    m[idx] = True
    return m


def recall_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    m = top_k_mask(p, frac)
    return float(y[m].sum() / max(y.sum(), 1))


def precision_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    return float(y[top_k_mask(p, frac)].mean())


def lift_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = np.minimum(df.index * bins // len(df), bins - 1) + 1
    base = df.y.mean()
    t = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_p=("p", "mean"))
    t["churn_rate"] = t.churners / t.customers
    t["lift"] = t.churn_rate / base
    t["cum_gain"] = t.churners.cumsum() / max(df.y.sum(), 1)
    t["cum_lift"] = (t.churners.cumsum() / t.customers.cumsum()) / base
    return t.reset_index()


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def classification_report(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(y.mean()),
        "recall_at_10pct": recall_at(y, p, 0.10),
        "precision_at_10pct": precision_at(y, p, 0.10),
        "lift_at_10pct": precision_at(y, p, 0.10) / max(y.mean(), 1e-9),
    }
    if threshold is not None:
        out["threshold"] = float(threshold)
        out["f1_at_threshold"] = float(f1_score(y, p >= threshold))
    return out
