"""Bronze -> silver. Writes typed, validated tables and a cleaning report."""

from __future__ import annotations

import json

import pandas as pd
import polars as pl

from churn.cleaning import customers as cust
from churn.cleaning import transactions as txn
from churn.config import get_logger, load_config
from churn.ingest import read_bronze
from churn.validation import check_silver

log = get_logger(__name__)


def _map_cards(df: pl.DataFrame, card_map: pl.DataFrame, card_col: str = "card_id") -> tuple[pl.DataFrame, int]:
    out = df.rename({card_col: "card_id"}) if card_col != "card_id" else df
    out = out.with_columns(pl.col("card_id").str.strip_chars()).join(card_map, on="card_id", how="left")
    unknown = out["customer_id"].null_count()
    return out.filter(pl.col("customer_id").is_not_null()), unknown


def clean_complaints(raw: pl.DataFrame, card_map: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    df, unknown = _map_cards(raw, card_map)
    df = df.select(
        "ticket_id", "customer_id",
        pl.col("created_at").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
        # "OUT OF STOCK" and "out_of_stock" are the same category typed by different agents
        pl.col("category").str.strip_chars().str.to_lowercase().str.replace_all(r"[\s\-]+", "_"),
        pl.col("channel").str.to_lowercase(),
        (pl.col("resolved") == "True").alias("resolved"),
    ).unique("ticket_id")
    return df, unknown


def clean_ledger(raw: pl.DataFrame, card_map: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    df, unknown = _map_cards(raw, card_map)
    return df.select(
        pl.col("ledger_id").cast(pl.Int64), "customer_id",
        pl.col("event_date").str.to_date("%Y-%m-%d"),
        "event_type", pl.col("points").cast(pl.Int64),
    ), unknown


def clean_marketing(camp: pl.DataFrame, targets: pl.DataFrame, red: pl.DataFrame, card_map: pl.DataFrame):
    camp = camp.select(
        pl.col("campaign_id").cast(pl.Int32), "campaign_type", "channel",
        pl.col("start_date").str.to_date(), pl.col("end_date").str.to_date(),
        pl.col("offer_cost_per_customer").cast(pl.Float64),
    )
    t, _ = _map_cards(targets, card_map)
    t = t.select(pl.col("campaign_id").cast(pl.Int32), "customer_id").unique()
    r, _ = _map_cards(red, card_map)
    r = r.select(
        "customer_id", pl.col("redeemed_date").str.to_date(), pl.col("coupon_upc").cast(pl.Int64),
        pl.col("campaign_id").cast(pl.Int32),
    ).unique()
    return camp, t, r


def run() -> dict:
    cfg = load_config()
    silver = cfg.path("silver")
    report: dict = {}

    # customers + identity resolution
    raw_c = read_bronze("crm_customers").to_pandas()
    golden, card_map, links = cust.resolve(raw_c, cfg.cleaning.fuzzy_name_threshold)
    check_silver("customers", golden)
    check_silver("card_map", card_map)
    golden.to_parquet(silver / "customers.parquet", index=False)
    card_map.to_parquet(silver / "card_map.parquet", index=False)
    links.to_parquet(silver / "identity_links.parquet", index=False)
    report["customers"] = {
        "cards_in": len(raw_c), "customers_out": len(golden), "links": len(links),
        "multi_card_customers": int((golden.n_cards > 1).sum()),
        "invalid_phone": int(raw_c.phone.notna().sum() - golden.phone_e164.notna().sum()),
    }
    cm = pl.from_pandas(card_map)

    # transactions
    lines, rep = txn.clean(
        read_bronze("pos_legacy"), read_bronze("pos_v2"), cm, read_bronze("products"),
        cfg.simulation.test_store_id, cfg.cleaning.outlier_quantile,
    )
    report["transactions"] = rep
    check_silver("transactions", lines.select(
        "invoice_id", "line_no", "txn_ts", "store_id", "customer_id", "product_id", "quantity",
        "sales_value", "discount_total", "is_return", "payment_method",
    ).to_pandas())
    lines.write_parquet(silver / "transactions.parquet", compression="zstd")
    baskets = txn.to_baskets(lines)
    baskets.write_parquet(silver / "baskets.parquet")
    report["baskets"] = baskets.height

    comp, unk = clean_complaints(read_bronze("crm_complaints"), cm)
    comp.write_parquet(silver / "complaints.parquet")
    report["complaints"] = {"rows": comp.height, "unknown_card": unk}

    led, unk = clean_ledger(read_bronze("loyalty_ledger"), cm)
    led.write_parquet(silver / "points_ledger.parquet")
    report["points_ledger"] = {"rows": led.height, "unknown_card": unk}

    camp, tgt, red = clean_marketing(
        read_bronze("mkt_campaigns"), read_bronze("mkt_targets"), read_bronze("mkt_redemptions"), cm
    )
    camp.write_parquet(silver / "campaigns.parquet")
    tgt.write_parquet(silver / "campaign_targets.parquet")
    red.write_parquet(silver / "coupon_redemptions.parquet")
    report["marketing"] = {"campaigns": camp.height, "targets": tgt.height, "redemptions": red.height}

    # identity resolution quality against simulator ground truth (evaluation only)
    truth_path = cfg.path("exports") / "_truth" / "card_household.csv"
    if truth_path.exists():
        report["dedup_quality"] = cust.dedup_quality(card_map, pd.read_csv(truth_path))
        log.info("dedup quality %s", report["dedup_quality"])

    out = cfg.path("reports") / "cleaning_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("silver written; report -> %s", out)
    return report


if __name__ == "__main__":
    run()
