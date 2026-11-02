FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY pyproject.toml .
COPY configs ./configs
COPY src ./src
RUN pip install --no-deps .
COPY models/production ./models/production
# the API looks customers up in the latest feature snapshot; mount data/gold in production instead
COPY app/data/latest_snapshot.parquet ./data/gold/snapshots.parquet

RUN useradd -m api && chown -R api /srv
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
