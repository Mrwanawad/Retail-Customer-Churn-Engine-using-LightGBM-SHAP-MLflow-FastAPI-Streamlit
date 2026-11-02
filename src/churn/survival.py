"""Time-to-churn: Kaplan-Meier per risk segment + Cox PH on the main drivers.

Duration = days from snapshot until the customer's last purchase before a `primary_horizon`-day
silence; event = that silence was fully observed. Restricted mean survival time (RMST) to 365 days
is reported because most segments never drop below 50% survival, which leaves the median undefined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index, restricted_mean_survival_time

COX_FEATURES = [
    "recency_days", "overdue_ratio", "gap_mean", "visit_trend", "spend_trend", "baskets_90d",
    "spend_90d", "tenure_days", "complaints_90d", "discount_share_90d",
]
RMST_HORIZON = 365


def km_by_segment(df: pd.DataFrame, segment_col: str = "risk_segment") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per segment, survival curves long format)."""
    rows, curves = [], []
    for seg, g in df.groupby(segment_col):
        km = KaplanMeierFitter().fit(g.surv_duration, g.surv_event, label=str(seg))
        med = km.median_survival_time_
        rows.append({
            "risk_segment": seg,
            "customers": len(g),
            "events": int(g.surv_event.sum()),
            "median_days_to_churn": None if np.isinf(med) else float(med),
            "rmst_days_365": float(restricted_mean_survival_time(km, t=RMST_HORIZON)),
            "survival_90d": float(km.predict(90)),
            "survival_180d": float(km.predict(180)),
        })
        sf = km.survival_function_.reset_index()
        sf.columns = ["days", "survival"]
        curves.append(sf.assign(risk_segment=seg))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def fit_cox(train: pd.DataFrame, test: pd.DataFrame) -> tuple[CoxPHFitter, float, pd.DataFrame]:
    cols = [c for c in COX_FEATURES if c in train.columns]
    prep = lambda d: d[cols + ["surv_duration", "surv_event"]].fillna(d[cols].median()).assign(  # noqa: E731
        **{c: lambda x, c=c: np.log1p(x[c].clip(lower=0)) for c in cols}
    )
    tr = prep(train)
    tr = tr[tr.surv_duration > 0]
    cph = CoxPHFitter(penalizer=0.05).fit(tr, "surv_duration", "surv_event")
    te = prep(test)
    risk = cph.predict_partial_hazard(te)
    c_index = concordance_index(te.surv_duration, -risk, te.surv_event)
    summary = cph.summary[["coef", "exp(coef)", "p"]].reset_index().rename(columns={"covariate": "feature"})
    return cph, float(c_index), summary
