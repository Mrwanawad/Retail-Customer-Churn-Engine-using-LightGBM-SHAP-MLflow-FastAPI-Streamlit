# Project Checkpoint

Status of the Retail Customer Churn & Retention Engine, its deployment, and what remains open.

## Links

| What | Where |
|---|---|
| Live app (production) | https://retail-churn-engine.streamlit.app |
| Repository | https://github.com/Mrwanawad/Retail-Customer-Churn-Engine-using-LightGBM-SHAP-MLflow-FastAPI-Streamlit |
| Branch deployed | `main` |
| Scope and design | [architecture.md](architecture.md) |
| Technical documentation | [README.md](README.md) |

## Status by component

| Component | Status | Verified by |
|---|---|---|
| Messy source-system exports (CRM, POS legacy + v2, loyalty, marketing) | Done | Pipeline run, cleaning report |
| Ingestion to bronze (CSV path) | Done | Pipeline run |
| Ingestion from PostgreSQL (`sql/extract/*.sql`) | Code complete | Not run: no database available |
| Raw contracts and silver schemas (pandera) | Done | Pipeline run, unit tests |
| Identity resolution and transaction cleaning | Done | Unit tests; pairwise precision 99.3%, recall 94.7% |
| Point-in-time features and 30/60/90-day labels | Done | Leakage and purge tests |
| Training, tuning, calibration, SHAP, MLflow registry | Done | Pipeline run; model `churn-60d` registered |
| CLV, survival, decision layer | Done | Pipeline run, app checks |
| Monitoring (PSI, Evidently, realised vs predicted, retrain flag) | Done | Pipeline run, app Model card |
| Streamlit app, 5 pages | Done, deployed | AppTest smoke tests; manual testing of every page on the live URL |
| FastAPI service | Done | API tests; endpoint and error-case checks against the running container |
| Docker image for the API | Done | Built and run locally; health check healthy, non-root user |
| Airflow DAGs (weekly scoring, monthly retrain) | Code complete | Not run: no Airflow environment |
| Tests and lint | 27 passing, ruff clean | `pytest -q`, `ruff check` |

## Headline results (out-of-time test)

| Metric | Value |
|---|---|
| PR-AUC, 60-day churn | 0.350 (base rate 0.071) |
| ROC-AUC | 0.885 |
| Recall / lift @ top 10% | 48.2% / 4.8x |
| Brier (calibrated) | 0.053 |
| 30-day / 90-day PR-AUC | 0.500 / 0.219 |
| Cox PH C-index | 0.838 |
| Campaign | 155 customers targeted, $671 expected profit |

## Deployment strategy

The system splits into a **heavy offline pipeline** and **light serving layers**. Training and scoring never run on the hosting platform; the serving layers only load finished artifacts.

### 1. Streamlit app on Streamlit Community Cloud (live)

| Setting | Value |
|---|---|
| Platform | Streamlit Community Cloud, connected to the GitHub repository |
| Entry point | `app/streamlit_app.py` |
| Python | 3.12 |
| Dependencies | `requirements.txt` (app-only, pinned; no training libraries) |
| Theme and server settings | `.streamlit/config.toml` |
| Inputs at runtime | `models/production/churn_model.joblib` + `metadata.json` (0.7 MB) and `app/data/` (7.3 MB) |
| Personal data | None: no names, phones or emails are exported to `app/data/` |
| Redeploy trigger | Every push to `main` |
| Cache invalidation | Caches are keyed by artifact file modification time, so a redeploy with new artifacts never serves stale data |

**Release procedure (new model or new weekly scores):**

```bash
python -m churn.pipeline retrain     # or: weekly
pytest -q
git add models/production app/data
git commit -m "Refresh model and scores"
git push origin main                 # Community Cloud redeploys automatically
```

Then open the live URL and check the Model card page shows the new model version.

**Operational notes:**
- The app sleeps after a period without visitors; open it a few minutes before a demo so it is warm.
- Logs, reboot and settings are under "Manage app" on the live page (owner account only).

### 2. FastAPI scoring service in Docker (image ready, not hosted)

| Setting | Value |
|---|---|
| Image | `Dockerfile`, base `python:3.12-slim`, dependencies from `requirements-api.txt` |
| Contents | `churn` package, `configs/`, production model bundle, latest feature snapshot |
| Runtime | Uvicorn, 2 workers, port 8000, non-root user `api`, built-in health check on `/health` |
| Configuration | `CHURN_ROOT=/srv` tells the installed package where configs and models live |
| Size | about 1.2 GB |

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

For production hosting (Cloud Run, Azure Container Apps, ECS, or Kubernetes), push the image to a registry and mount `data/gold/snapshots.parquet` from shared storage instead of baking it into the image, so weekly scores reach the API without a rebuild.

### 3. Batch pipeline under Airflow (code ready)

| DAG | Schedule | Tasks |
|---|---|---|
| `churn_weekly_scoring` | Mondays 05:00 | extract → validate → clean → build_features → score → monitor → trigger retrain if flagged |
| `churn_monthly_retrain` | 1st of month 03:00, or triggered by monitoring | build_features → train → score → monitor → export_app_data |

Production setup needs: an Airflow deployment with the `churn` package installed, `DATABASE_URL` pointing at the source PostgreSQL, shared storage for `data/` and `models/`, and an MLflow tracking server in place of the local SQLite file.

## Open items

1. Run the PostgreSQL ingestion and score write-back against a real database.
2. Run both DAGs in an Airflow environment.
3. Host the API image and point it at shared storage for the latest snapshot.
4. Move MLflow from local SQLite to a shared tracking server.
5. Feed real campaign outcomes back in so measured uplift (treated vs control holdout) replaces expected profit.

## Known caveats

- Complaints are synthetic (Dunnhumby has none) and carry a planted signal; every other feature comes from real purchases.
- On historical snapshots, realised campaign profit peaks below the model's expected profit, so treat expected profit as an upper bound.
- The Evidently report and the PSI check count drifted features differently; PSI drives the retrain trigger.
