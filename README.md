# Retail Customer Churn & Retention Engine

Weekly batch system that ranks loyalty customers by their risk of not buying again in the next 60 days (with 30 and 90 day scores too), explains each score, and picks the set of customers worth a retention offer by expected profit.

Built on **Dunnhumby *The Complete Journey***: two years of real supermarket purchases from 2,500 households (2.6M line items, 582 stores, real campaigns and coupon redemptions). Design: [architecture.md](architecture.md).

## Results (out-of-time test: snapshots 2025-09-07 and 2025-10-05)

| Metric | Value |
|---|---|
| PR-AUC (primary) | **0.350** (base rate 0.071, so 5x random) |
| ROC-AUC | 0.885 |
| Recall @ top 10% | 48.2% of churners caught by contacting 10% of customers |
| Lift @ top 10% | 4.8x |
| Brier (calibrated) | 0.053 |
| 30-day / 90-day model PR-AUC | 0.500 (base 0.144) / 0.219 (base 0.041) |
| Cox PH C-index (time to churn) | 0.838 |
| Identity resolution, pairwise precision / recall | 99.3% / 94.7% |

Model selection on the validation snapshots: Optuna-tuned LightGBM (PR-AUC 0.400) narrowly beat the best baseline, RandomForest (0.398), followed by LogisticRegression (0.362), XGBoost (0.348) and CatBoost (0.289). Class weights, under-sampling and SMOTE-NC did not beat the unweighted tuned model, so imbalance is handled by calibration and the decision threshold instead of resampling.

## Pipeline

```
data/raw/dunnhumby        original Kaggle CSVs
   │  churn.simulate.exports   rebuild as messy source-system exports (CRM, POS, loyalty, marketing)
data/exports
   │  churn.ingest             every column read as text + lineage → parquet partitioned by extract date
data/bronze
   │  churn.validation         pandera raw contracts (warn on dirty rows, fail on schema change)
   │  churn.cleaning.run       identity resolution, POS harmonisation, dedup, returns, outliers
data/silver                    + strict pandera schemas
   │  churn.features           point-in-time snapshots every 28 days, labels for 30/60/90 days, survival target
data/gold/snapshots.parquet
   │  churn.train              out-of-time split → 8 baselines → Optuna → selection → calibration → SHAP → MLflow registry
models/production
   │  churn.score              calibrated probabilities, top-3 SHAP reasons, CLV (BG/NBD + Gamma-Gamma), expected profit, targets, KM by segment
   │  churn.monitor            PSI drift, prediction drift, realised vs predicted, uplift, retrain trigger, Evidently report
   ├─ churn.api                FastAPI /predict /explain (Docker)
   └─ app/streamlit_app.py     5-page demo
```

`dags/churn_dags.py` runs the same functions in Airflow: weekly scoring (Mondays) and monthly retraining, which also starts early when monitoring raises the retrain flag.

### Why the data is rebuilt as messy exports

Dunnhumby is a curated research panel: one clean household key, integer day numbers, no duplicates. Real retail feeds aren't like that, so `churn.simulate.exports` rebuilds the data as four source systems would deliver it. **What each household bought, when, where and for how much is untouched.** Only identities, formats and system defects are simulated:

| Defect injected | Cleaning answer |
|---|---|
| 6% of households registered twice (nickname/typo, reformatted phone, uppercased or new email), baskets split across both cards | Blocking on phone, email and name+store, rapidfuzz scoring with nickname folding, union-find, golden record |
| Names as `SMITH, JOHN` / ` john  smith `, 7 phone formats, 3 date formats, tier casing | Parsers + `phonenumbers` E.164 + validated email |
| POS vendor upgrade in June 2025: new column names, `dd/mm/yyyy` timestamps, `BR00364` branch codes, `1,234.50` amounts | Two contracts, one harmonised schema |
| Two monthly files loaded twice (252k duplicate lines) | Dedup on (invoice, line) |
| Test store and test card, baskets without a scanned card | Excluded and counted |
| Returns as negative lines, 2k of them without a matching sale | Flagged, linked to the original, orphans kept |
| Prices keyed ×100 | Fixed when unit price > 50× the product median |
| `account_status`, `closed_date`, current loyalty tier in the CRM | Dropped as leakage: written after churn or at export time |

Real defects in Dunnhumby itself also surface: float residue such as `2.22e-16` in discount columns (caught by the raw contracts), positive discounts (sign errors), and fuel lines that record litres as quantity.

Complaints don't exist in Dunnhumby, so they are synthetic. Their rate rises in the 8 weeks before a long lapse, so complaint features carry some planted signal (`days_since_complaint` ranks 7th in SHAP importance). Everything else the model learns comes from real behaviour. The simulator writes the true card→household mapping to `data/exports/_truth/` so identity resolution can be scored; the pipeline never reads it.

### Leakage controls

- Features use only rows dated on or before the snapshot cutoff. `tests/test_features_split.py` checks that appending future data changes no feature value.
- Labels whose window hasn't closed stay null (the last 2 snapshots are scoring-only).
- Purged out-of-time split: train labels close before the validation snapshots begin, and validation labels close before the test snapshots begin.
- Hyper-parameter CV is grouped by customer and stratified, inside the training window only.
- Encoders and imputers are fit inside the sklearn Pipeline.

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt && pip install -e .
python -m churn.download                               # Kaggle → data/raw/dunnhumby
python -m churn.pipeline all                           # simulate → ingest → validate → clean → features → train → score → monitor → export
pytest -q
streamlit run app/streamlit_app.py
uvicorn churn.api:app --port 8000                      # or: docker build -t churn-api . && docker run -p 8000:8000 churn-api
mlflow ui --backend-store-uri sqlite:///mlflow.db      # runs, metrics, registered model
```

```bash
curl -X POST localhost:8000/explain -H "Content-Type: application/json" -d '{"customer_id": "C000123"}'
```

The full build takes about 6 minutes on a laptop, most of it in Optuna. `python -m churn.ingest --source postgres` pulls the same tables with `sql/extract/*.sql` when `DATABASE_URL` is set, and scoring then appends to a `churn_scores` table.

## Deploy: Streamlit Community Cloud

The app reads only committed artifacts: `models/production/` (0.7 MB) and `app/data/` (6.5 MB, no names, phones or emails). In Community Cloud choose the repo, main file `app/streamlit_app.py`, Python 3.12. `requirements.txt` holds the app's pinned dependencies. The app sleeps when idle, so open it a few minutes before a demo.

## Layout

```
configs/config.yaml   all parameters (churn horizon, cutoffs, simulation rates, offer economics)
src/churn/            pipeline package
sql/extract/          source extract queries for the PostgreSQL path
dags/                 Airflow DAGs
app/                  Streamlit app + exported demo data
models/production/    exported model bundle + metadata (from the MLflow registry run)
tests/                cleaning, leakage, split, decision, API and app smoke tests
```
