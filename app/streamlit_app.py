"""Retail Churn & Retention Engine: interactive demo.

Reads only committed artifacts: models/production/churn_model.joblib and app/data/*.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.decision import RISK_LABELS, expected_profit, profit_curve, risk_segment  # noqa: E402
from churn.explain import Explainer  # noqa: E402

DATA = ROOT / "app" / "data"
SEG_COLORS = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#f4a261", "Critical": "#e76f51"}

st.set_page_config(page_title="Retail Churn Engine", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------- loaders
MODEL_PATH = ROOT / "models" / "production" / "churn_model.joblib"


# cache keys include the file's mtime so a redeploy with new artifacts never serves stale ones
@st.cache_resource
def _load_bundle(mtime: float):
    b = joblib.load(MODEL_PATH)
    b["explainer"] = Explainer(b["base_model"])
    return b


def load_bundle():
    return _load_bundle(MODEL_PATH.stat().st_mtime)


def load(name: str):
    return _load(name, (DATA / name).stat().st_mtime)


@st.cache_data
def _load(name: str, mtime: float):
    p = DATA / name
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return p.read_text(encoding="utf-8")


bundle = load_bundle()
meta = bundle["meta"]
FEATS = meta["features_numeric"] + meta["features_categorical"]
H = int(meta["label"].split("_")[1].rstrip("d"))
scores = load("churn_scores.parquet")
latest = load("latest_snapshot.parquet").set_index("customer_id")
day_one = pd.Timestamp("2024-01-01")
to_date = lambda d: (day_one + pd.Timedelta(days=int(d) - 1)).strftime("%d %b %Y")  # noqa: E731


def predict(X: pd.DataFrame) -> dict[int, np.ndarray]:
    return {h: m.predict_proba(X[FEATS])[:, 1] for h, m in bundle["models"].items()}


def kpi(col, label, value, help_=None):
    col.metric(label, value, help=help_)


# ---------------------------------------------------------------- pages
def overview():
    st.title("Churn overview")
    st.caption(f"Scoring snapshot **{to_date(scores.cutoff_day.iloc[0])}** · model `{meta['model_version']}` · "
               f"churn = no purchase in the next {H} days")
    trend = load("churn_trend.parquet")
    matured = trend.dropna(subset=[f"churn_{H}d"])
    c = st.columns(5)
    kpi(c[0], "Active customers", f"{len(scores):,}")
    kpi(c[1], f"Latest observed {H}d churn", f"{matured[f'churn_{H}d'].iloc[-1]:.1%}",
        f"Snapshot {to_date(matured.cutoff_day.iloc[-1])}, the last one whose {H}-day window has closed")
    at_risk = scores.risk_segment.isin(["High", "Critical"]).sum()
    kpi(c[2], "High + Critical risk", f"{at_risk:,}")
    kpi(c[3], "Targeted this week", f"{int(scores.targeted.sum()):,}", "Expected profit > 0; 10% of them held out as control")
    kpi(c[4], "Expected campaign profit", f"${scores.loc[scores.targeted, 'expected_profit'].sum():,.0f}",
        "All customers with positive expected profit, at $5 offer cost and 25% success (tune on Campaign planner)")

    left, right = st.columns([3, 2])
    with left:
        long = trend.melt(id_vars=["snapshot_date"], value_vars=[c for c in trend.columns if c.startswith("churn_")],
                          var_name="horizon", value_name="churn_rate").dropna()
        long["horizon"] = long.horizon.str.replace("churn_", "").str.replace("d", " days")
        fig = px.line(long, x="snapshot_date", y="churn_rate", color="horizon", markers=True,
                      title="Observed churn rate by snapshot")
        fig.update_yaxes(tickformat=".0%", title=None)
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, width="stretch")
    with right:
        seg = scores.groupby("risk_segment").agg(customers=("customer_id", "size"), avg_p=("churn_probability", "mean"),
                                                 clv=("clv", "sum")).reindex(RISK_LABELS).reset_index()
        fig = px.bar(seg, x="risk_segment", y="customers", color="risk_segment", color_discrete_map=SEG_COLORS,
                     text="customers", title="Customers by risk segment", hover_data={"avg_p": ":.1%", "clv": ":$,.0f"})
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    lift = load("lift.csv")
    a, b = st.columns(2)
    with a:
        fig = px.bar(lift, x="decile", y="lift", title="Lift by risk decile (out-of-time test)", text_auto=".1f")
        fig.add_hline(y=1, line_dash="dot")
        fig.update_xaxes(dtick=1, title="decile (1 = highest predicted risk)")
        st.plotly_chart(fig, width="stretch")
    with b:
        g = pd.concat([pd.DataFrame({"share_contacted": [0], "cum_gain": [0]}),
                       lift.assign(share_contacted=lift.decile / 10)[["share_contacted", "cum_gain"]]])
        fig = go.Figure([
            go.Scatter(x=g.share_contacted, y=g.cum_gain, mode="lines+markers", name="model"),
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dot")),
        ])
        fig.update_layout(title="Cumulative gains", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="customers contacted", yaxis_title="churners reached")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Expected time to churn by segment")
    surv = load("survival_by_segment.parquet").set_index("risk_segment").reindex(RISK_LABELS).reset_index()
    curves = load("survival_curves.parquet")
    a, b = st.container(), st.container()  # table full width, then curves
    table = pd.DataFrame({
        "Segment": surv.risk_segment,
        "Customer-snapshots": surv.customers,
        "Still shopping at 90d": surv.survival_90d.map("{:.0%}".format),
        "Still shopping at 180d": surv.survival_180d.map("{:.0%}".format),
        "Mean active days (next 365)": surv.rmst_days_365.round(0).astype(int),
        "Median days to churn": pd.to_numeric(surv.median_days_to_churn, errors="coerce").map(
            lambda v: "not reached" if pd.isna(v) else f"{v:.0f}"),
    })
    a.dataframe(table, hide_index=True, width="stretch")
    a.caption("Kaplan-Meier on the validation and test snapshots, grouped by the model's risk segment.")
    fig = px.line(curves, x="days", y="survival", color="risk_segment", color_discrete_map=SEG_COLORS,
                  line_shape="hv", title="Kaplan-Meier: share still shopping")
    fig.update_yaxes(tickformat=".0%")
    b.plotly_chart(fig, width="stretch")


def _label(cid: str) -> str:
    r = scores.set_index("customer_id").loc[cid]
    return f"{cid} · {r.churn_probability:.0%} · {r.risk_segment}"


def customer_scorer():
    st.title("Customer scorer")
    order = scores.sort_values("churn_probability", ascending=False).customer_id.tolist()
    cid = st.selectbox("Customer (sorted by risk)", order, format_func=_label)
    row = scores.set_index("customer_id").loc[cid]
    X = latest.loc[[cid]]
    decision_label = "Hold out" if row.holdout else ("Target" if row.targeted else "Skip")
    c = st.columns(6)
    kpi(c[0], f"P(churn {H}d)", f"{row.churn_probability:.1%}")
    others = [h for h in sorted(bundle["models"]) if h != H]
    for col, h in zip(c[1:3], others, strict=False):
        kpi(col, f"P(churn {h}d)", f"{row[f'churn_prob_{h}d']:.1%}")
    kpi(c[3], "Segment", row.risk_segment)
    kpi(c[4], "CLV if retained", f"${row.clv:,.0f}", "Expected margin over the next 180 days (BG/NBD + Gamma-Gamma)")
    kpi(c[5], "Offer", decision_label,
        f"Expected profit ${row.expected_profit:,.2f}. 'Hold out' = random control group, not contacted")

    ex: Explainer = bundle["explainer"]
    sv = ex.shap_frame(X[FEATS]).iloc[0]
    sv = sv[[s for s in sv.index if not s.startswith("missingindicator_")]]
    top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(10)
    rest = sv.drop(top.index).sum()
    def show_value(f):
        v = X.iloc[0][f] if f in X.columns else None
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:,.2f}".rstrip("0").rstrip(".") if isinstance(v, (int, float, np.number)) else str(v)

    labels = [f"{f} = {show_value(f)}" for f in top.index[::-1]]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * (len(top) + 1) + ["total"],
        y=["baseline", "other features", *labels, "model score (log-odds)"],
        x=[ex.base_value, rest, *top.values[::-1], None],
        increasing=dict(marker_color="#e76f51"), decreasing=dict(marker_color="#2a9d8f"),
    ))
    fig.update_layout(title="Why this score: SHAP contributions (red raises risk)", height=520, margin=dict(l=10))
    a, b = st.columns([3, 2])
    a.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Top reasons**")
        for i in (1, 2, 3):
            if isinstance(row.get(f"reason_{i}"), str):
                st.markdown(f"{i}. {row[f'reason_{i}']}")
        st.markdown("**Profile at snapshot**")
        show = ["recency_days", "gap_mean", "overdue_ratio", "baskets_90d", "spend_90d", "spend_trend",
                "discount_share_90d", "complaints_90d", "tenure_days", "n_cards"]
        prof = X[show].astype(float).T.rename(columns={cid: "value"})
        prof["value"] = prof["value"].map(lambda v: "n/a" if pd.isna(v) else f"{v:,.2f}")
        st.dataframe(prof, width="stretch")
    st.caption("SHAP is computed on the uncalibrated booster (log-odds); the probability shown above is after calibration.")


def what_if():
    st.title("What-if analysis")
    st.caption("Change behaviour and watch the calibrated probability move. Derived features (overdue ratio, visit days) are kept consistent.")
    order = scores.sort_values("churn_probability", ascending=False).customer_id.tolist()
    # default: a regular shopper in the Medium segment, so every slider has room to move both ways
    typical = latest.loc[scores[scores.risk_segment == "Medium"].customer_id]
    typical = typical[typical.baskets_90d.between(10, 40)]
    default = order.index(typical.index[0]) if len(typical) else 0
    cid = st.selectbox("Start from customer", order, index=default, format_func=_label)
    base = latest.loc[[cid]].copy()
    b = base.iloc[0]

    def val(col, lo, hi, fallback):
        v = b[col]
        return float(np.clip(v, lo, hi)) if pd.notna(v) else fallback

    c1, c2 = st.columns(2)
    with c1:
        recency = st.slider("Days since last visit", 0, 90, int(val("recency_days", 0, 90, 7)))
        baskets = st.slider("Baskets in last 90 days", 1, 200, int(val("baskets_90d", 1, 200, 10)))
        complaints = st.slider("Complaints in last 90 days", 0, 5, int(val("complaints_90d", 0, 5, 0)))
    with c2:
        disc = st.slider("Discount share of basket value", 0.0, 0.6, val("discount_share_90d", 0, 0.6, 0.05), 0.01, format="%.2f")
        spend_tr = st.slider("Spend vs previous quarter", 0.0, 3.0, val("spend_trend", 0, 3, 1.0), 0.05, format="%.2fx")
        visit_tr = st.slider("Visits vs previous quarter", 0.0, 3.0, val("visit_trend", 0, 3, 1.0), 0.05, format="%.2fx")

    x = base.copy()
    x["recency_days"] = recency
    x["overdue_ratio"] = recency / max(b.gap_mean if pd.notna(b.gap_mean) else 7, 1)
    scale = baskets / max(b.baskets_90d, 1)
    for col in ("baskets_90d", "visit_days_90d", "spend_90d"):
        x[col] = b[col] * scale
    x["baskets_28d"] = b.baskets_28d * scale if recency < 28 else 0
    x["spend_28d"] = b.spend_28d * scale if recency < 28 else 0
    x["discount_share_90d"] = disc
    x["spend_trend"] = spend_tr
    x["visit_trend"] = visit_tr
    x["complaints_90d"] = complaints
    x["complaints_180d"] = max(b.complaints_180d, complaints)

    p0, p1 = predict(base)[H][0], predict(x)[H][0]
    c = st.columns(3)
    kpi(c[0], "Original", f"{p0:.1%}")
    c[1].metric("What-if", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
    c[2].metric("Segment", risk_segment([p1]).iloc[0])

    sweep = pd.concat([x.assign(recency_days=r, overdue_ratio=r / max(b.gap_mean if pd.notna(b.gap_mean) else 7, 1)) for r in range(0, 91, 3)])
    sweep["p"] = predict(sweep)[H]
    fig = px.line(sweep, x="recency_days", y="p", title="Risk as the silence gets longer (other settings fixed)")
    fig.add_vline(x=recency, line_dash="dot")
    fig.update_yaxes(tickformat=".0%", title=f"P(churn {H}d)")
    st.plotly_chart(fig, width="stretch")


def campaign_planner():
    st.title("Campaign planner")
    st.caption("expected profit = P(churn) × CLV × offer success − offer cost. Target everyone with positive expected profit.")
    c1, c2, c3 = st.columns(3)
    cost = c1.slider("Offer cost per customer ($)", 1.0, 30.0, 5.0, 0.5)
    success = c2.slider("Offer success rate (churners saved)", 0.05, 0.6, 0.25, 0.01)
    budget = c3.number_input("Budget cap (customers, 0 = none)", 0, len(scores), 0, 50)

    curve = profit_curve(scores.churn_probability, scores.clv, success, cost)
    k_pos = int((curve.expected_profit > 0).sum())
    k = min(k_pos, budget) if budget else k_pos
    chosen = curve.iloc[k - 1] if k else None
    c = st.columns(4)
    kpi(c[0], "Customers targeted", f"{k:,}")
    kpi(c[1], "Campaign cost", f"${k * cost:,.0f}")
    kpi(c[2], "Expected profit", f"${chosen.cum_expected_profit if k else 0:,.0f}")
    kpi(c[3], "Projected ROI", f"{chosen.roi:.0%}" if k else "–", "expected profit / cost")
    fig = px.line(curve.head(min(len(curve), max(3 * k_pos, 300))), x="k", y="cum_expected_profit",
                  title="Cumulative expected profit by number targeted (ranked by expected profit)")
    if k:
        fig.add_vline(x=k, line_dash="dot", annotation_text=f"target {k}")
    fig.update_yaxes(tickprefix="$", title=None)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Does the rule hold on history?")
    tp = load("test_predictions.parquet")
    clv_proxy = tp.spend_180d * 0.25
    hist = profit_curve(tp.p_churn, clv_proxy, success, cost, y=tp[meta["label"]])
    k_hist = int((hist.expected_profit > 0).sum())
    hist = hist.head(max(3 * k_hist, 200))
    fig = go.Figure([
        go.Scatter(x=hist.k, y=hist.cum_expected_profit, name="expected"),
        go.Scatter(x=hist.k, y=hist.cum_realised_profit, name="realised (actual churners)"),
    ])
    fig.update_layout(title="Out-of-time test snapshots: expected vs realised campaign profit (CLV proxy = 180d spend × margin)",
                      xaxis_title="customers targeted", yaxis_tickprefix="$")
    st.plotly_chart(fig, width="stretch")

    tgt = scores.copy()
    tgt["expected_profit"] = expected_profit(tgt.churn_probability, tgt.clv, success, cost)
    tgt = tgt.sort_values("expected_profit", ascending=False).head(k)
    cols = ["customer_id", "churn_probability", "risk_segment", "clv", "expected_profit", "holdout", "reason_1", "reason_2", "reason_3"]
    st.dataframe(tgt[cols].style.format({"churn_probability": "{:.1%}", "clv": "${:,.0f}", "expected_profit": "${:,.2f}"}),
                 hide_index=True, width="stretch", height=320)
    st.download_button("Download target list (CSV)", tgt[cols].to_csv(index=False), "target_list.csv", "text/csv")


def model_card():
    st.title("Model card")
    m = meta["test_metrics"]
    st.markdown(
        f"**{meta['model_version']}** · `{meta.get('model_family', 'lightgbm')}` in a sklearn Pipeline (ColumnTransformer), "
        "picked on validation PR-AUC against an Optuna-tuned LightGBM and 7 baselines · "
        f"calibrated with **{meta['calibration']}** regression · label: no purchase in the next {H} days · "
        f"imbalance handling: `{meta['imbalance']}`"
    )
    c = st.columns(6)
    kpi(c[0], "PR-AUC", f"{m['pr_auc']:.3f}", f"base rate {m['base_rate']:.3f}")
    kpi(c[1], "ROC-AUC", f"{m['roc_auc']:.3f}")
    kpi(c[2], "Recall @ top 10%", f"{m['recall_at_10pct']:.1%}")
    kpi(c[3], "Lift @ top 10%", f"{m['lift_at_10pct']:.1f}x")
    kpi(c[4], "Brier", f"{m['brier']:.4f}", f"uncalibrated {m['brier_uncalibrated']:.4f}")
    kpi(c[5], "F1 @ profit threshold", f"{m['f1_at_threshold']:.3f}", f"threshold {m['threshold']:.3f}")

    sp = meta["split"]
    st.markdown(
        f"**Out-of-time validation.** Train snapshots {to_date(sp['train_cutoffs'][0])} → {to_date(sp['train_cutoffs'][-1])} "
        f"({len(sp['train_cutoffs'])} cutoffs) · calibration {', '.join(map(to_date, sp['valid_cutoffs']))} · "
        f"test {', '.join(map(to_date, sp['test_cutoffs']))}. Each set's {H}-day label window closes before the next set "
        "starts, so no training label overlaps the evaluation period. Hyper-parameters use grouped stratified CV "
        "(by customer) inside the training window only."
    )
    a, b = st.columns(2)
    with a:
        rel = load("reliability.csv")
        fig = go.Figure([
            go.Scatter(x=rel.mean_predicted_uncal, y=rel.observed_rate_uncal, mode="lines+markers", name="uncalibrated"),
            go.Scatter(x=rel.mean_predicted, y=rel.observed_rate, mode="lines+markers", name="calibrated"),
            go.Scatter(x=[0, rel.max().max()], y=[0, rel.max().max()], mode="lines", name="perfect", line=dict(dash="dot")),
        ])
        fig.update_layout(title="Reliability curve (test)", xaxis_title="predicted", yaxis_title="observed")
        st.plotly_chart(fig, width="stretch")
    with b:
        imp = load("shap_importance.csv").head(15)
        fig = px.bar(imp[::-1], x="mean_abs_shap", y="feature", orientation="h", title="Global importance (mean |SHAP|)")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Model comparison (validation snapshots)")
    base = load("baselines.csv")
    st.dataframe(base[["model", "imbalance", "pr_auc", "roc_auc", "recall_at_10pct", "lift_at_10pct", "brier"]]
                 .style.format(precision=4), hide_index=True, width="stretch")

    st.subheader("Survival model")
    sm = load("survival_metrics.json")
    st.markdown(f"Cox PH on log-transformed behaviour drivers · test **C-index {sm['c_index_test']:.3f}**")
    st.dataframe(load("cox_summary.csv").style.format(precision=3), hide_index=True, width="stretch")

    st.subheader("Data quality")
    cr, rc = load("cleaning_report.json"), load("raw_contracts.json")
    t = cr["transactions"]
    dq = cr.get("dedup_quality", {})
    c = st.columns(4)
    kpi(c[0], "POS lines in → out", f"{t['lines_in'] / 1e6:.2f}M → {t['lines_out'] / 1e6:.2f}M",
        f"{t['lines_in']:,} raw lines → {t['lines_out']:,} clean lines")
    kpi(c[1], "Double-loaded lines removed", f"{t['duplicate_lines_dropped']:,}")
    kpi(c[2], "Loyalty cards → customers", f"{cr['customers']['cards_in']:,} → {cr['customers']['customers_out']:,}")
    if dq:
        kpi(c[3], "Identity resolution P / R", f"{dq['precision']:.1%} / {dq['recall']:.1%}")
    with st.expander("Cleaning report and raw data contracts"):
        st.json(cr)
        st.dataframe(pd.DataFrame([{"table": r["table"], "rows": r["rows"], "status": r["status"],
                                    "failures": json.dumps(r["failures"])} for r in rc]), hide_index=True, width="stretch")

    st.subheader("Monitoring snapshot")
    if (DATA / "monitoring_summary.json").exists():
        ms = load("monitoring_summary.json")
        c = st.columns(4)
        kpi(c[0], "Features drifted (PSI > 0.2)", f"{ms['features_drifted']} / {ms['features_checked']}")
        kpi(c[1], "Prediction PSI", f"{ms['prediction_psi']:.3f}")
        kpi(c[2], "Latest matured PR-AUC", f"{ms['latest_matured_pr_auc']:.3f}", f"baseline {ms['baseline_pr_auc']:.3f}")
        kpi(c[3], "Retrain triggered", "Yes" if ms["retrain"] else "No", "; ".join(ms["retrain_reasons"]) or None)
        rvp = load("realised_vs_predicted.csv")
        if len(rvp):
            fig = px.line(rvp.melt(id_vars="snapshot_date", value_vars=["predicted_rate", "realised_rate"]),
                          x="snapshot_date", y="value", color="variable", markers=True, title="Predicted vs realised churn")
            fig.update_yaxes(tickformat=".1%", rangemode="tozero", title=None)
            st.plotly_chart(fig, width="stretch")
        if (DATA / "evidently_drift.html").exists():
            with st.expander("Evidently data drift report"):
                st.caption("Evidently flags drift with per-feature statistical tests, which become very sensitive at "
                           "thousands of rows. The retrain trigger uses PSI > 0.2, a magnitude threshold, so the two "
                           "counts differ by design.")
                st.components.v1.html(load("evidently_drift.html"), height=900, scrolling=True)


PAGES = [
    (overview, "Overview", ":material/insights:", "overview"),
    (customer_scorer, "Customer scorer", ":material/person_search:", "scorer"),
    (what_if, "What-if", ":material/tune:", "what-if"),
    (campaign_planner, "Campaign planner", ":material/campaign:", "campaign"),
    (model_card, "Model card", ":material/fact_check:", "model-card"),
]
start = os.environ.get("CHURN_APP_START_PAGE", "overview")  # lets tests open any page headlessly
pg = st.navigation([
    # the default page lives at "/", so it gets no url_path of its own
    st.Page(f, title=t, icon=i, default=True) if u == start else st.Page(f, title=t, icon=i, url_path=u)
    for f, t, i, u in PAGES
])
st.sidebar.caption("Data: Dunnhumby *The Complete Journey* (2,500 households, 2 years), rebuilt as messy source-system exports.")
pg.run()
