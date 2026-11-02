import pytest

from churn.config import ROOT

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "production" / "churn_model.joblib").exists()
    or not (ROOT / "data" / "gold" / "snapshots.parquet").exists(),
    reason="needs a trained model and gold snapshots",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from churn.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_and_explain_by_customer(client):
    from churn.api import STATE

    cid = STATE["latest"].index[0]
    p = client.post("/predict", json={"customer_id": cid}).json()
    assert 0 <= p["churn_probability"] <= 1
    assert set(p["probabilities"]) == {"30d", "60d", "90d"}
    e = client.post("/explain", json={"customer_id": cid}).json()
    assert e["churn_probability"] == pytest.approx(p["churn_probability"])
    assert len(e["contributions"]) > 10


def test_predict_raw_features_and_validation(client):
    r = client.post("/predict", json={"features": {"recency_days": 45, "baskets_90d": 2, "AGE_DESC": "45-54"}})
    assert r.status_code == 200
    assert client.post("/predict", json={"features": {"not_a_feature": 1}}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"features": {}}).status_code == 422
    assert client.post("/predict", json={"features": {"recency_days": "abc"}}).status_code == 422
    assert client.post("/predict", json={"features": {"recency_days": "45"}}).status_code == 200
    assert client.post("/predict", json={"customer_id": "C999999"}).status_code == 404
