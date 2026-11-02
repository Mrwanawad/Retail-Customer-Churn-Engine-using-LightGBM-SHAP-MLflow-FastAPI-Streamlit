# Retail Customer Churn & Retention Engine — Architecture

## Scenario
A retail business (multi-branch stores + loyalty program) loses repeat customers silently. The retention team needs a weekly ranked list of customers likely to stop buying in the next 30/60 days, the reason each one is at risk, and whether a retention offer is worth its cost. The system scores customers in batch, explains each score, and picks the profit-maximizing set of customers to target.

## Input
- Customer master: demographics, loyalty tier, signup date, preferred branch.
- Transactions: invoice-level purchases (date, branch, basket value, items, discounts, payment method).
- Engagement: loyalty point activity, returns, coupon redemption, complaints.
- Offer history: past campaigns, offer cost, response flag.
- Churn label definition: no purchase within N days after the observation cutoff (N set at build time).

## Output
- `churn_scores` table: customer_id, churn_probability (calibrated), risk_segment, top 3 SHAP reasons, scoring_date, model_version.
- Target list: customers selected by expected-profit rule.
- Survival estimate: expected time-to-churn per segment.
- Dashboard: churn rate trends, segment breakdown, lift chart, campaign ROI.
- REST endpoint for on-demand scoring of a single customer.

## Architecture

```
 Source DB / CSV exports
        |
        v
 [Ingestion]  SQL extract -> raw parquet (partitioned by snapshot date)
        |
        v
 [Validation]  pandera schemas: types, ranges, nulls, duplicates
        |
        v
 [Cleaning]  dedup customers (fuzzy name + phone normalization), fix types,
             handle returns/negative invoices, cap outliers
        |
        v
 [Feature Engineering]  point-in-time snapshot features (no future data):
             RFM, tenure, purchase frequency trends, basket diversity,
             discount dependency, complaint counts, days since last visit
        |
        v
 [Training Pipeline]  sklearn Pipeline(ColumnTransformer + model)
             time-based split -> imbalance handling -> Optuna tuning
             -> calibration -> SHAP -> MLflow log + registry
        |
        v
 [Decision Layer]  expected profit = P(churn) * CLV * offer_success - offer_cost
             -> threshold / top-k selection
        |
        +--> [Batch Scoring]  weekly job -> churn_scores table
        +--> [Serving API]    FastAPI /predict, /explain (Docker)
        +--> [Dashboard]      Streamlit
        |
        v
 [Monitoring]  Evidently: data drift, prediction drift, realized churn vs predicted
             -> retrain trigger
```

Orchestration: Airflow DAG (extract -> validate -> features -> score -> monitor), monthly retrain DAG.

## Tools Used
- Data: pandas, polars, NumPy, SQL (PostgreSQL), pyarrow/parquet
- Profiling & EDA: ydata-profiling, Matplotlib, Seaborn, Plotly Express
- Validation: pandera
- Cleaning: rapidfuzz (customer dedup), phonenumbers
- Modeling: scikit-learn (Pipeline, ColumnTransformer, LogisticRegression, RandomForest, CalibratedClassifierCV), LightGBM, XGBoost, CatBoost
- Imbalance: imbalanced-learn (SMOTE-NC, RandomUnderSampler), class weights
- Tuning: Optuna
- Explainability: SHAP
- Survival: lifelines (Kaplan-Meier, CoxPH)
- CLV: lifetimes (BG/NBD, Gamma-Gamma)
- MLOps: MLflow (tracking + registry), Airflow, Docker, Evidently
- Serving & UI: FastAPI, Pydantic, Streamlit
- Quality: pytest, ruff

## Tools Not Used (and why)
- Deep learning (PyTorch/TensorFlow): tabular data; gradient boosting wins on accuracy, speed, and explainability.
- LLMs / RAG: not needed for a structured prediction task.
- Spark: data fits on one machine with polars.
- Kubernetes / Kubeflow: batch weekly scoring; Docker + Airflow is enough.
- Real-time streaming (Kafka): churn decisions are weekly, not per-second.
- AutoML: tuning and feature choices must be explainable in review.

## Evaluation Metrics
- Primary: PR-AUC (imbalanced target).
- Secondary: ROC-AUC, recall@top-10%, precision@top-k, F1 at chosen threshold.
- Calibration: Brier score, reliability curve.
- Business: lift / cumulative gains at top deciles, expected profit curve vs. threshold, campaign ROI.
- Survival: concordance index (C-index).
- Validation scheme: out-of-time split (train on older snapshots, test on latest); stratified CV within training window only.

## Leakage Controls
- Features computed only from data before the snapshot cutoff.
- Drop fields written after churn (cancellation notes, closure flags).
- Encoders/scalers fit inside the Pipeline on training folds only.

## Production Monitoring
- Input drift per feature (PSI / Evidently data drift).
- Prediction distribution drift.
- Realized churn vs. predicted once labels mature (N days later).
- Campaign uplift tracking: treated vs. control holdout.
- Retrain trigger: PR-AUC drop beyond threshold or significant drift.

## Deployment Plan — Streamlit Community Cloud
**Why Streamlit:** tabular model, small artifacts (LightGBM + preprocessing pipeline, a few MB), and the demo value is interactive what-if analysis. Streamlit gives UI + model in one Python app, free hosting from a GitHub repo.

**Demo app pages:**
1. Overview: churn rate trends, segment breakdown, lift / gains chart.
2. Customer scorer: form or customer picker -> calibrated churn probability + top SHAP reasons (waterfall plot).
3. What-if: change recency / frequency / discount usage -> watch probability move.
4. Campaign planner: sliders for offer cost and success rate -> expected-profit curve, chosen threshold, number of customers targeted, projected ROI.
5. Model card: metrics (PR-AUC, lift@10%, Brier), out-of-time validation, monitoring snapshot (Evidently report as HTML).

**Setup:**
- `app/streamlit_app.py` loads model from `models/` (exported from MLflow registry), sample scored data from parquet.
- `requirements.txt` pinned; `.streamlit/config.toml` for theme.
- Secrets (if any) in Streamlit secrets, never in repo.
- `@st.cache_resource` for model, `@st.cache_data` for data.

**Limits to know:** ~1 GB RAM; app hibernates after inactivity — open it before the interview to wake it.
