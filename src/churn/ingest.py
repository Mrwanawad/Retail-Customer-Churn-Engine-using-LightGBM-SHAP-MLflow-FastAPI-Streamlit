"""Ingestion: source exports -> bronze parquet.

Bronze keeps the data exactly as delivered (every column read as text) plus lineage columns:
_source_file, _row_hash, _ingested_at. Nothing is fixed here; fixing is the cleaning layer's job,
so bad loads can always be replayed from bronze.

Sources:
  csv       (default) files dropped by the source systems under data/exports
  postgres  same tables pulled with the SQL in sql/extract/*.sql (needs DATABASE_URL)
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from churn.config import ROOT, get_logger, load_config

log = get_logger(__name__)

# bronze table -> glob under exports/ (POS has two layouts after the vendor upgrade)
CSV_SOURCES = {
    "crm_customers": "crm/customers.csv",
    "crm_complaints": "crm/complaints.csv",
    "pos_legacy": "pos/pos_legacy_*.csv",
    "pos_v2": "pos/pos_v2_*.csv",
    "loyalty_ledger": "loyalty/points_ledger.csv",
    "mkt_campaigns": "marketing/campaigns.csv",
    "mkt_targets": "marketing/campaign_targets.csv",
    "mkt_redemptions": "marketing/coupon_redemptions.csv",
    "mkt_coupons": "marketing/coupons.csv",
    "products": "catalog/products.csv",
}


def _with_lineage(df: pl.DataFrame, source: str, ingested_at: datetime) -> pl.DataFrame:
    data_cols = df.columns
    return df.with_columns(
        pl.lit(source).alias("_source_file"),
        pl.concat_str([pl.col(c).fill_null("<null>") for c in data_cols], separator="|").hash().alias("_row_hash"),
        pl.lit(ingested_at).alias("_ingested_at"),
    )


def read_csv_source(exports: Path, pattern: str, ingested_at: datetime) -> pl.DataFrame:
    files = sorted(exports.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no export matches {pattern}")
    frames = [
        _with_lineage(pl.read_csv(f, infer_schema=False, encoding="utf8-lossy"), f.name, ingested_at)
        for f in files
    ]
    return pl.concat(frames, how="diagonal")


def read_postgres_source(table: str, ingested_at: datetime) -> pl.DataFrame:
    url = os.environ["DATABASE_URL"]
    query = (ROOT / "sql" / "extract" / f"{table}.sql").read_text()
    df = pl.read_database_uri(query, url).cast(pl.Utf8)
    return _with_lineage(df, f"postgres:{table}", ingested_at)


def write_bronze(df: pl.DataFrame, bronze: Path, table: str, extract_date: str) -> Path:
    """One partition per extract date; re-running the same date replaces that partition."""
    part = bronze / table / f"extract_date={extract_date}"
    part.mkdir(parents=True, exist_ok=True)
    out = part / "part-0.parquet"
    df.write_parquet(out, compression="zstd")
    return out


def run(source: str = "csv", extract_date: str | None = None) -> dict[str, int]:
    cfg = load_config()
    extract_date = extract_date or datetime.now(UTC).strftime("%Y-%m-%d")
    ingested_at = datetime.now(UTC).replace(tzinfo=None)
    exports, bronze = cfg.path("exports"), cfg.path("bronze")
    counts = {}
    for table, pattern in CSV_SOURCES.items():
        df = read_csv_source(exports, pattern, ingested_at) if source == "csv" else read_postgres_source(table, ingested_at)
        write_bronze(df, bronze, table, extract_date)
        counts[table] = df.height
        log.info("bronze %-16s %9d rows from %d file(s)", table, df.height, df["_source_file"].n_unique())
    return counts


def read_bronze(table: str, extract_date: str | None = None) -> pl.DataFrame:
    """Latest partition unless a date is given."""
    base = load_config().path("bronze") / table
    parts = sorted(base.glob("extract_date=*"))
    if not parts:
        raise FileNotFoundError(f"no bronze partitions for {table}; run ingestion first")
    part = base / f"extract_date={extract_date}" if extract_date else parts[-1]
    return pl.read_parquet(part / "part-0.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "postgres"], default="csv")
    ap.add_argument("--extract-date")
    a = ap.parse_args()
    run(a.source, a.extract_date)
