"""Per-customer SHAP reasons, shared by batch scoring, the API and the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

# plain-language templates; {v} is the customer's value
REASON_TEXT = {
    "recency_days": "No visit in {v:.0f} day{s}",
    "overdue_ratio": "Overdue: {v:.1f}x their usual gap between visits",
    "gap_trend": "Visits getting further apart ({v:.1f}x usual gap)",
    "gap_mean": "Usual gap between visits is {v:.0f} days",
    "gap_max": "Longest past gap {v:.0f} days",
    "gap_std": "Irregular visit pattern",
    "gap_mean_90": "Recent gap between visits {v:.0f} days",
    "visit_trend": "Visits vs previous quarter: {v:.0%}",
    "spend_trend": "Spend vs previous quarter: {v:.0%}",
    "spend_momentum_28d": "Last-4-week spend pace {v:.0%} of quarter average",
    "baskets_28d": "{v:.0f} basket{s} in last 4 weeks",
    "baskets_90d": "{v:.0f} basket{s} in last 90 days",
    "baskets_180d": "{v:.0f} basket{s} in last 180 days",
    "visit_days_28d": "{v:.0f} shopping day{s} in last 4 weeks",
    "visit_days_90d": "{v:.0f} shopping day{s} in last 90 days",
    "spend_28d": "${v:,.0f} spent in last 4 weeks",
    "spend_90d": "${v:,.0f} spent in last 90 days",
    "spend_180d": "${v:,.0f} spent in last 180 days",
    "avg_basket_90d": "Average basket ${v:,.2f}",
    "discount_share_90d": "{v:.0%} of basket value from discounts",
    "coupon_basket_share_90d": "Coupons on {v:.0%} of baskets",
    "complaints_90d": "{v:.0f} complaint{s} in last 90 days",
    "complaints_180d": "{v:.0f} complaint{s} in last 180 days",
    "unresolved_complaints_180d": "{v:.0f} unresolved complaint{s}",
    "days_since_complaint": "Complaint {v:.0f} days ago",
    "returns_90d": "{v:.0f} return{s} in last 90 days",
    "stores_90d": "Shops at {v:.0f} store{s}",
    "tenure_days": "Customer for {v:.0f} day{s}",
    "points_balance": "{v:,.0f} unspent loyalty points",
    "campaigns_targeted_180d": "Targeted by {v:.0f} campaign{s} in 180 days",
    "coupons_redeemed_180d": "{v:.0f} campaign coupon{s} redeemed",
    "commodities_90d": "Buys across {v:.0f} product line{s}",
    "avg_departments_90d": "{v:.1f} departments per basket",
}


def reason_text(feature: str, value) -> str:
    tpl = REASON_TEXT.get(feature)
    if tpl is None or value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{feature.replace('_', ' ')} = {value}"
    try:
        v = float(value)
        return tpl.format(v=v, s="" if round(v) == 1 else "s")
    except (TypeError, ValueError):
        return f"{feature.replace('_', ' ')} = {value}"


class Explainer:
    def __init__(self, base_model):
        self.pre = base_model.named_steps["pre"]
        self.tree = shap.TreeExplainer(base_model.named_steps["model"])
        self.names = list(self.pre.get_feature_names_out())

    def shap_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = pd.DataFrame(self.pre.transform(X), columns=self.names, index=X.index)
        sv = self.tree.shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv  # (n, features, classes) for sklearn forests
        return pd.DataFrame(sv, columns=self.names, index=X.index)

    @property
    def base_value(self) -> float:
        ev = self.tree.expected_value
        return float(np.ravel(ev)[-1])

    def top_reasons(self, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Top-k features pushing each customer's risk up (positive SHAP), with readable text."""
        sv = self.shap_frame(X)
        sv = sv[[c for c in sv.columns if not c.startswith("missingindicator_")]]
        rows = []
        for idx, r in sv.iterrows():
            top = r[r > 0].sort_values(ascending=False).head(k)
            out = {}
            for i, (feat, val) in enumerate(top.items(), start=1):
                out[f"reason_{i}"] = reason_text(feat, X.at[idx, feat] if feat in X.columns else None)
                out[f"reason_{i}_feature"] = feat
                out[f"reason_{i}_shap"] = float(val)
            rows.append(out)
        return pd.DataFrame(rows, index=X.index)
