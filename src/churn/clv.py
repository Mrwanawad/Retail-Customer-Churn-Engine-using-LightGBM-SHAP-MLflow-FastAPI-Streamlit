"""Customer lifetime value at a cutoff: BG/NBD (purchase rate, P(alive)) + Gamma-Gamma (basket value).

The decision layer needs the value at stake *if the customer is retained*, so CLV here is
expected purchases over the horizon at the customer's own rate x expected basket value x margin.
BG/NBD's P(alive) is kept as a classic non-ML churn benchmark for the model card.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from lifetimes import BetaGeoFitter, GammaGammaFitter

from churn.config import get_logger

log = get_logger(__name__)


def rfm_at(sales: pl.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """lifetimes-style summary (frequency, recency, T, monetary) from daily purchase totals."""
    daily = (
        sales.filter(pl.col("day") <= cutoff_day)
        .group_by("customer_id", "day").agg(pl.col("basket_value").sum().alias("v"))
    )
    s = daily.group_by("customer_id").agg(
        (pl.len() - 1).alias("frequency"),
        (pl.col("day").max() - pl.col("day").min()).alias("recency"),
        (cutoff_day - pl.col("day").min()).alias("T"),
        # Gamma-Gamma uses the mean value of repeat purchases only
        ((pl.col("v").sum() - pl.col("v").sort_by("day").first()) / (pl.len() - 1).clip(lower_bound=1)).alias("monetary"),
    )
    return s.to_pandas().set_index("customer_id").astype(float)


def fit_clv(sales: pl.DataFrame, cutoff_day: int, horizon_days: int, margin: float) -> pd.DataFrame:
    rfm = rfm_at(sales, cutoff_day)
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm.frequency, rfm.recency, rfm["T"])
    rep = rfm[(rfm.frequency > 0) & (rfm.monetary > 0)]
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(rep.frequency, rep.monetary)

    out = pd.DataFrame(index=rfm.index)
    out["p_alive_bgnbd"] = bgf.conditional_probability_alive(rfm.frequency, rfm.recency, rfm["T"])
    out["exp_basket_value"] = ggf.conditional_expected_average_profit(rfm.frequency, rfm.monetary.clip(lower=0.01))
    # purchase rate if retained: the customer's own observed rate, shrunk toward BG/NBD's population rate
    r, alpha = bgf.params_["r"], bgf.params_["alpha"]
    rate = (r + rfm.frequency) / (alpha + rfm["T"])
    out["exp_purchases_if_retained"] = rate * horizon_days
    out["clv"] = (out.exp_purchases_if_retained * out.exp_basket_value * margin).clip(lower=0)
    log.info("CLV at day %d: median $%.0f, p_alive median %.2f", cutoff_day, out.clv.median(), out.p_alive_bgnbd.median())
    return out.replace([np.inf, -np.inf], np.nan).fillna({"clv": 0.0})
