"""Decision layer: who gets a retention offer.

expected_profit = P(churn) * CLV * offer_success - offer_cost
Target a customer when expected_profit > 0, optionally capped at a budget of top-k customers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_BINS = [0.0, 0.08, 0.20, 0.50, 1.0001]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]


def risk_segment(p) -> pd.Series:
    return pd.cut(pd.Series(p), RISK_BINS, labels=RISK_LABELS, right=False).astype(str)


def expected_profit(p, clv, success: float, cost: float) -> np.ndarray:
    return np.asarray(p) * np.asarray(clv) * success - cost


def select_targets(df: pd.DataFrame, success: float, cost: float, budget_k: int | None = None,
                   holdout_share: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Rank by expected profit; a random share of targeted customers is held out as control
    so campaign uplift can be measured instead of assumed."""
    out = df.copy()
    out["expected_profit"] = expected_profit(out.churn_probability, out.clv, success, cost)
    out = out.sort_values("expected_profit", ascending=False)
    out["targeted"] = out.expected_profit > 0
    if budget_k is not None:
        out["targeted"] &= np.arange(len(out)) < budget_k
    rng = np.random.default_rng(seed)
    out["holdout"] = out.targeted & (rng.random(len(out)) < holdout_share)
    return out


def profit_curve(p, clv, success: float, cost: float, y=None) -> pd.DataFrame:
    """Cumulative expected (and, if labels given, realised) profit when targeting the top-k by expected profit."""
    ep = expected_profit(p, clv, success, cost)
    order = np.argsort(-ep)
    out = pd.DataFrame({
        "k": np.arange(1, len(ep) + 1),
        "threshold_p": np.asarray(p)[order],
        "expected_profit": ep[order],
        "cum_expected_profit": np.cumsum(ep[order]),
        "cum_cost": cost * np.arange(1, len(ep) + 1),
    })
    if y is not None:
        realised = np.asarray(y)[order] * np.asarray(clv)[order] * success - cost
        out["cum_realised_profit"] = np.cumsum(realised)
    out["roi"] = out.cum_expected_profit / out.cum_cost
    return out
