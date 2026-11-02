"""On-demand scoring API.

GET  /health
POST /predict   {"customer_id": "C000123"}  or  {"features": {...}}  -> calibrated probabilities + segment
POST /explain   same body -> top reasons + full SHAP contributions
Run: uvicorn churn.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from churn import decision
from churn.config import load_config
from churn.explain import Explainer
from churn.score import load_bundle

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    bundle = load_bundle(cfg)
    snaps = pd.read_parquet(cfg.path("gold") / "snapshots.parquet")
    latest = snaps[snaps.cutoff_day == snaps.cutoff_day.max()].set_index("customer_id")
    STATE.update(cfg=cfg, bundle=bundle, latest=latest, explainer=Explainer(bundle["base_model"]))
    yield
    STATE.clear()


app = FastAPI(title="Retail Churn API", version="1.0", lifespan=lifespan)


class ScoreRequest(BaseModel):
    customer_id: str | None = Field(None, pattern=r"^C\d{6}$", examples=["C000123"])
    features: dict[str, Any] | None = Field(None, description="Raw feature values; missing ones are imputed")

    @model_validator(mode="after")
    def one_of(self):
        if (self.customer_id is None) == (self.features is None):
            raise ValueError("send exactly one of customer_id or features")
        return self


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float
    probabilities: dict[str, float]
    risk_segment: str
    model_version: str
    snapshot_date: str | None


class Explanation(Prediction):
    base_value_logit: float
    top_reasons: list[str]
    contributions: dict[str, float]


def _frame(req: ScoreRequest) -> tuple[pd.DataFrame, str | None]:
    meta = STATE["bundle"]["meta"]
    feats = meta["features_numeric"] + meta["features_categorical"]
    if req.customer_id:
        if req.customer_id not in STATE["latest"].index:
            raise HTTPException(404, f"{req.customer_id} not in the latest scoring snapshot")
        row = STATE["latest"].loc[[req.customer_id]]
        return row[feats].reset_index(drop=True), str(row.snapshot_date.iloc[0])[:10]
    unknown = set(req.features) - set(feats)
    if unknown:
        raise HTTPException(422, f"unknown features: {sorted(unknown)}")
    X = pd.DataFrame([{f: req.features.get(f) for f in feats}])
    for c in meta["features_numeric"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in meta["features_categorical"]:
        X[c] = X[c].fillna("Unknown").astype(str)
    return X, None


def _predict(req: ScoreRequest) -> tuple[pd.DataFrame, dict]:
    bundle, cfg = STATE["bundle"], STATE["cfg"]
    X, snap_date = _frame(req)
    probs = {f"{h}d": float(m.predict_proba(X)[0, 1]) for h, m in bundle["models"].items()}
    p = probs[f"{cfg.churn.primary_horizon}d"]
    return X, {
        "customer_id": req.customer_id,
        "churn_probability": p,
        "probabilities": probs,
        "risk_segment": decision.risk_segment([p]).iloc[0],
        "model_version": bundle["meta"]["model_version"],
        "snapshot_date": snap_date,
    }


@app.get("/health")
def health():
    meta = STATE["bundle"]["meta"]
    return {"status": "ok", "model_version": meta["model_version"], "customers_loaded": len(STATE["latest"])}


@app.post("/predict", response_model=Prediction)
def predict(req: ScoreRequest):
    return _predict(req)[1]


@app.post("/explain", response_model=Explanation)
def explain(req: ScoreRequest):
    X, pred = _predict(req)
    ex: Explainer = STATE["explainer"]
    sv = ex.shap_frame(X).iloc[0]
    sv = sv[[c for c in sv.index if not c.startswith("missingindicator_")]]
    reasons = ex.top_reasons(X, k=3).iloc[0]
    return {
        **pred,
        "base_value_logit": ex.base_value,
        "top_reasons": [reasons[f"reason_{i}"] for i in (1, 2, 3) if f"reason_{i}" in reasons and pd.notna(reasons[f"reason_{i}"])],
        "contributions": sv.sort_values(key=abs, ascending=False).round(5).to_dict(),
    }
