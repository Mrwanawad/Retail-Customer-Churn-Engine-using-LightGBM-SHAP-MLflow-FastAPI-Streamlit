"""POS line cleaning (polars).

Steps, each counted in the cleaning report:
  harmonise    legacy + v2 layouts -> one typed schema (timestamps, branch codes, text amounts)
  dedup        lines loaded twice (reloaded monthly files) -> keep one per (invoice_id, line_no)
  exclude      test/training store, test card, lines with no loyalty card, unknown cards
  returns      flag (legacy TXN_TYPE / v2 negative qty), link to original sale, mark orphans
  fix          positive discounts (sign error) -> 0; fuel lines (litres, not units) -> qty 1;
               price keyed x100 -> /100 when unit price is > 50x the product's median
  cap          non-fuel quantity at the configured quantile
"""

from __future__ import annotations

import polars as pl

from churn.config import get_logger

log = get_logger(__name__)

TEST_CARD = "LC00000000"
FUEL_DEPARTMENTS = ["KIOSK-GAS", "MISC. TRANS."]


def _num(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all(",", "").str.strip_chars().cast(pl.Float64, strict=False)


def harmonise(legacy: pl.DataFrame, v2: pl.DataFrame) -> pl.DataFrame:
    hhmm = pl.col("TXN_TIME").str.zfill(4)
    a = legacy.select(
        pl.col("INVOICE_NO").cast(pl.Int64).alias("invoice_id"),
        pl.col("LINE_NO").cast(pl.Int32).alias("line_no"),
        (pl.col("TXN_DATE") + " " + hhmm.str.slice(0, 2) + ":" + hhmm.str.slice(2, 2))
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False).alias("txn_ts"),
        pl.col("STORE_ID").cast(pl.Int32).alias("store_id"),
        pl.col("CARD_ID").str.strip_chars().alias("card_id"),
        pl.col("PRODUCT_ID").cast(pl.Int64).alias("product_id"),
        _num("QTY").alias("quantity"),
        _num("SALES_VALUE").alias("sales_value"),
        _num("RETAIL_DISC").alias("retail_disc"),
        _num("COUPON_DISC").alias("coupon_disc"),
        _num("COUPON_MATCH_DISC").alias("coupon_match_disc"),
        (pl.col("TXN_TYPE") == "RETURN").alias("is_return"),
        pl.col("ORIG_INVOICE_NO").cast(pl.Int64, strict=False).alias("orig_invoice_id"),
        pl.col("PAYMENT_METHOD").str.to_lowercase().alias("payment_method"),
        pl.col("_source_file"),
    )
    b = v2.select(
        pl.col("invoice_id").str.strip_prefix("INV-").cast(pl.Int64).alias("invoice_id"),
        pl.col("line").cast(pl.Int32).alias("line_no"),
        pl.col("txn_ts").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False).alias("txn_ts"),
        pl.col("branch_code").str.strip_prefix("BR").cast(pl.Int32).alias("store_id"),
        pl.col("loyalty_card").str.strip_chars().alias("card_id"),
        pl.col("sku").cast(pl.Int64).alias("product_id"),
        _num("quantity").alias("quantity"),
        _num("net_amount").alias("sales_value"),
        _num("promo_disc").alias("retail_disc"),
        _num("coupon_disc").alias("coupon_disc"),
        _num("coupon_match_disc").alias("coupon_match_disc"),
        # v2 has no transaction type: a return is a negative line
        ((_num("quantity") < 0) | (_num("net_amount") < 0)).alias("is_return"),
        pl.col("ref_invoice").str.strip_prefix("INV-").cast(pl.Int64, strict=False).alias("orig_invoice_id"),
        pl.col("tender").replace_strict(
            {"Cash": "cash", "Credit/Debit": "card", "Wallet": "mobile"}, default=None
        ).alias("payment_method"),
        pl.col("_source_file"),
    )
    return pl.concat([a, b])


def clean(
    legacy: pl.DataFrame,
    v2: pl.DataFrame,
    card_map: pl.DataFrame,
    products: pl.DataFrame,
    test_store_id: int,
    outlier_quantile: float,
) -> tuple[pl.DataFrame, dict[str, int]]:
    rep: dict[str, int] = {}
    df = harmonise(legacy, v2)
    rep["lines_in"] = df.height
    rep["unparseable_ts"] = df["txn_ts"].null_count()
    df = df.filter(pl.col("txn_ts").is_not_null())

    n = df.height
    df = df.unique(["invoice_id", "line_no"], keep="first", maintain_order=True)
    rep["duplicate_lines_dropped"] = n - df.height

    test = ((pl.col("store_id") == test_store_id) | (pl.col("card_id") == TEST_CARD)).fill_null(False)
    rep["test_lines_dropped"] = df.filter(test).height
    df = df.filter(~test)

    rep["unscanned_card_lines_dropped"] = df["card_id"].null_count()
    df = df.filter(pl.col("card_id").is_not_null())

    df = df.join(card_map, on="card_id", how="left")
    rep["unknown_card_lines_dropped"] = df["customer_id"].null_count()
    df = df.filter(pl.col("customer_id").is_not_null())

    # returns: link to the original sale; orphans stay (refund happened) but are flagged
    sale_invoices = df.filter(~pl.col("is_return")).select(pl.col("invoice_id").unique().alias("orig_invoice_id"))
    df = df.with_columns(
        pl.when(pl.col("is_return"))
        .then(pl.col("orig_invoice_id").is_in(sale_invoices["orig_invoice_id"].implode()).fill_null(False))
        .otherwise(None)
        .alias("return_linked")
    )
    rep["return_lines"] = df.filter(pl.col("is_return")).height
    rep["orphan_return_lines"] = df.filter(pl.col("is_return") & ~pl.col("return_linked")).height

    # discounts are stored negative; a positive one is a sign error
    disc_cols = ["retail_disc", "coupon_disc", "coupon_match_disc"]
    rep["positive_discount_fixed"] = df.filter(pl.any_horizontal([pl.col(c) > 0 for c in disc_cols])).height
    df = df.with_columns([pl.when(pl.col(c) > 0).then(0.0).otherwise(pl.col(c)).alias(c) for c in disc_cols])
    df = df.with_columns((-(pl.sum_horizontal(disc_cols))).alias("discount_total"))

    # fuel/misc lines record litres or odometer-style counters as quantity
    prod = products.select(
        pl.col("PRODUCT_ID").cast(pl.Int64).alias("product_id"),
        pl.col("DEPARTMENT").str.strip_chars().alias("department"),
        pl.col("COMMODITY_DESC").str.strip_chars().alias("commodity"),
    )
    df = df.join(prod, on="product_id", how="left")
    fuel = pl.col("department").is_in(FUEL_DEPARTMENTS)
    rep["fuel_qty_normalised"] = df.filter(fuel & (pl.col("quantity").abs() > 1)).height
    df = df.with_columns(pl.when(fuel).then(pl.col("quantity").sign()).otherwise(pl.col("quantity")).alias("quantity"))

    # price keyed x100: unit price far above the product's typical unit price
    unit = pl.col("sales_value") / pl.col("quantity").abs().clip(lower_bound=1)
    df = df.with_columns(unit.alias("_unit"))
    med = (
        df.filter(~pl.col("is_return") & (pl.col("sales_value") > 0))
        .group_by("product_id").agg(pl.col("_unit").median().alias("_med"), pl.len().alias("_n"))
    )
    df = df.join(med, on="product_id", how="left")
    typo = (pl.col("_n") >= 5) & (pl.col("_unit").abs() > 50 * pl.col("_med")) & (pl.col("_unit").abs() > 20)
    rep["price_typo_fixed"] = df.filter(typo).height
    df = df.with_columns(
        pl.when(typo).then(pl.col("sales_value") / 100).otherwise(pl.col("sales_value")).alias("sales_value")
    ).drop("_unit", "_med", "_n")

    cap = df.filter(~fuel & (pl.col("quantity") > 0))["quantity"].quantile(outlier_quantile)
    rep["quantity_capped"] = df.filter(~fuel & (pl.col("quantity") > cap)).height
    df = df.with_columns(
        pl.when(~fuel & (pl.col("quantity") > cap)).then(cap).otherwise(pl.col("quantity")).alias("quantity")
    )

    rep["lines_out"] = df.height
    for k, v in rep.items():
        log.info("transactions %-30s %9d", k, v)
    return df.sort("txn_ts", "invoice_id", "line_no"), rep


def to_baskets(lines: pl.DataFrame) -> pl.DataFrame:
    """One row per sale invoice (returns are kept as their own invoices with is_return=True)."""
    return (
        lines.group_by("invoice_id", "customer_id", "is_return")
        .agg(
            pl.col("txn_ts").min(),
            pl.col("store_id").first(),
            pl.col("payment_method").first(),
            pl.col("sales_value").sum().alias("basket_value"),
            pl.col("discount_total").sum(),
            (pl.col("coupon_disc") < 0).any().alias("coupon_used"),
            pl.col("quantity").sum().alias("n_items"),
            pl.len().alias("n_lines"),
            pl.col("department").n_unique().alias("n_departments"),
            pl.col("commodity").n_unique().alias("n_commodities"),
        )
        .sort("txn_ts")
    )
