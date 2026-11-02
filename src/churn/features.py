"""Point-in-time snapshot features and labels.

For each cutoff day c (every 28 days), a customer is in the snapshot if they bought in (c-90, c] and
have at least `min_history_days` of history. Features use only rows dated <= c. Labels look at (c, c+h]
and are left null when that window is not fully observed yet (immature label).

Survival target: days from c until the customer's last purchase before their first gap of
`primary_horizon` days; censored when the data ends before such a gap is seen.
"""

from __future__ import annotations

import polars as pl

from churn.config import get_logger, load_config

log = get_logger(__name__)

DEMO_COLS = [
    "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC", "HOMEOWNER_DESC", "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC",
]
ID_COLS = ["customer_id", "cutoff_day", "snapshot_date"]


def _day(col: str, day_one) -> pl.Expr:
    return (pl.col(col).cast(pl.Date) - pl.lit(day_one)).dt.total_days().cast(pl.Int32) + 1


def load_silver(cfg) -> dict[str, pl.DataFrame]:
    s = cfg.path("silver")
    d1 = cfg.day_one
    baskets = pl.read_parquet(s / "baskets.parquet").with_columns(_day("txn_ts", d1).alias("day"))
    return {
        "baskets": baskets,
        "sales": baskets.filter(~pl.col("is_return")),
        "returns": baskets.filter(pl.col("is_return")),
        "complaints": pl.read_parquet(s / "complaints.parquet").with_columns(_day("created_at", d1).alias("day")),
        "ledger": pl.read_parquet(s / "points_ledger.parquet").with_columns(_day("event_date", d1).alias("day")),
        "targets": pl.read_parquet(s / "campaign_targets.parquet").join(
            pl.read_parquet(s / "campaigns.parquet").with_columns(_day("start_date", d1).alias("day")),
            on="campaign_id",
        ),
        "redemptions": pl.read_parquet(s / "coupon_redemptions.parquet").with_columns(
            _day("redeemed_date", d1).alias("day")
        ),
        "customers": pl.read_parquet(s / "customers.parquet"),
    }


def _window(df: pl.DataFrame, c: int, days: int) -> pl.DataFrame:
    return df.filter((pl.col("day") > c - days) & (pl.col("day") <= c))


def snapshot_features(t: dict[str, pl.DataFrame], c: int, cfg) -> pl.DataFrame:
    ch = cfg.churn
    sales = t["sales"].filter(pl.col("day") <= c)
    first = sales.group_by("customer_id").agg(pl.col("day").min().alias("first_day"))
    active = _window(sales, c, ch.activity_window).select("customer_id").unique()
    pop = active.join(first, on="customer_id").filter(pl.col("first_day") <= c - ch.min_history_days)

    s = sales.join(pop.select("customer_id"), on="customer_id")
    f = pop.with_columns((pl.lit(c) - pl.col("first_day")).alias("tenure_days"))

    # recency, cadence
    visits = s.select("customer_id", "day").unique().sort("customer_id", "day")
    gaps = visits.with_columns(pl.col("day").diff().over("customer_id").alias("gap")).drop_nulls("gap")
    g = gaps.group_by("customer_id").agg(
        pl.col("gap").mean().alias("gap_mean"),
        pl.col("gap").std().alias("gap_std"),
        pl.col("gap").max().alias("gap_max"),
        pl.col("gap").filter(pl.col("day") > c - 90).mean().alias("gap_mean_90"),
    )
    last = visits.group_by("customer_id").agg(pl.col("day").max().alias("last_day"))
    f = f.join(last, on="customer_id").join(g, on="customer_id", how="left").with_columns(
        (pl.lit(c) - pl.col("last_day")).alias("recency_days"),
    ).with_columns(
        (pl.col("recency_days") / pl.col("gap_mean").clip(lower_bound=1)).alias("overdue_ratio"),
        (pl.col("gap_mean_90") / pl.col("gap_mean").clip(lower_bound=1)).alias("gap_trend"),
    ).drop("last_day", "first_day")

    # frequency / monetary over several windows
    for w in (28, 90, 180):
        sw = _window(s, c, w).group_by("customer_id").agg(
            pl.len().alias(f"baskets_{w}d"),
            pl.col("day").n_unique().alias(f"visit_days_{w}d"),
            pl.col("basket_value").sum().alias(f"spend_{w}d"),
        )
        f = f.join(sw, on="customer_id", how="left")
    prev = s.filter((pl.col("day") > c - 180) & (pl.col("day") <= c - 90)).group_by("customer_id").agg(
        pl.len().alias("baskets_prev_90d"), pl.col("basket_value").sum().alias("spend_prev_90d")
    )
    f = f.join(prev, on="customer_id", how="left")
    w90 = _window(s, c, 90).group_by("customer_id").agg(
        pl.col("basket_value").mean().alias("avg_basket_90d"),
        pl.col("n_items").mean().alias("avg_items_90d"),
        pl.col("n_departments").mean().alias("avg_departments_90d"),
        pl.col("n_commodities").sum().alias("commodities_90d"),
        (pl.col("discount_total").sum() / (pl.col("basket_value").sum() + pl.col("discount_total").sum()).clip(lower_bound=1e-6)).alias("discount_share_90d"),
        pl.col("coupon_used").mean().alias("coupon_basket_share_90d"),
        pl.col("store_id").n_unique().alias("stores_90d"),
        (pl.col("payment_method") == "cash").mean().alias("cash_share_90d"),
        (pl.col("payment_method") == "mobile").mean().alias("mobile_share_90d"),
        (pl.col("txn_ts").dt.weekday() >= 6).mean().alias("weekend_share_90d"),
        pl.col("txn_ts").dt.hour().mean().alias("avg_hour_90d"),
    )
    f = f.join(w90, on="customer_id", how="left")
    f = f.with_columns(
        (pl.col("spend_90d") / pl.col("spend_prev_90d").clip(lower_bound=1)).alias("spend_trend"),
        (pl.col("baskets_90d") / pl.col("baskets_prev_90d").clip(lower_bound=1)).alias("visit_trend"),
        (pl.col("spend_28d") * 90 / 28 / pl.col("spend_90d").clip(lower_bound=1)).alias("spend_momentum_28d"),
    )

    # returns
    r = _window(t["returns"], c, 90).group_by("customer_id").agg(
        pl.len().alias("returns_90d"), (-pl.col("basket_value").sum()).alias("return_value_90d")
    )
    # complaints
    comp = t["complaints"].filter(pl.col("day") <= c)
    cpl = comp.group_by("customer_id").agg(
        pl.col("day").filter(pl.col("day") > c - 90).len().alias("complaints_90d"),
        pl.col("day").filter(pl.col("day") > c - 180).len().alias("complaints_180d"),
        (~pl.col("resolved")).filter(pl.col("day") > c - 180).sum().alias("unresolved_complaints_180d"),
        (pl.lit(c) - pl.col("day").max()).alias("days_since_complaint"),
    )
    # loyalty points
    led = t["ledger"].filter(pl.col("day") <= c).group_by("customer_id").agg(
        pl.col("points").sum().alias("points_balance"),
        (pl.col("event_type") == "REDEEM").filter(pl.col("day") > c - 180).sum().alias("redemptions_180d"),
    )
    # marketing exposure / response
    tg = t["targets"].filter((pl.col("day") <= c) & (pl.col("day") > c - 180)).group_by("customer_id").agg(
        pl.len().alias("campaigns_targeted_180d")
    )
    rd = _window(t["redemptions"], c, 180).group_by("customer_id").agg(pl.len().alias("coupons_redeemed_180d"))

    for extra in (r, cpl, led, tg, rd):
        f = f.join(extra, on="customer_id", how="left")

    zero_fill = [
        c_ for c_ in f.columns
        if c_.startswith(("baskets_", "visit_days_", "spend_", "returns_", "return_value", "complaints_",
                          "unresolved_", "redemptions_", "campaigns_", "coupons_", "points_"))
        and c_ not in ("spend_trend", "spend_momentum_28d")
    ]
    f = f.with_columns([pl.col(z).fill_null(0) for z in zero_fill])

    # static attributes known at the cutoff
    cust = t["customers"].select(
        "customer_id", "n_cards", "signup_date", *DEMO_COLS
    ).with_columns([pl.col(d).fill_null("Unknown") for d in DEMO_COLS])
    f = f.join(cust, on="customer_id", how="left").with_columns(
        (pl.lit(cfg.day_to_date(c)) - pl.col("signup_date").cast(pl.Date)).dt.total_days().alias("days_since_signup"),
        pl.lit(c).alias("cutoff_day"),
        pl.lit(cfg.day_to_date(c)).alias("snapshot_date"),
    ).drop("signup_date")
    return f


def labels(sales: pl.DataFrame, pop: pl.DataFrame, c: int, horizons: list[int], data_end: int, gap: int) -> pl.DataFrame:
    future = sales.filter(pl.col("day") > c).select("customer_id", "day").unique()
    out = pop.select("customer_id")
    for h in horizons:
        bought = future.filter(pl.col("day") <= c + h).select("customer_id").unique().with_columns(pl.lit(1).alias("_b"))
        col = out.join(bought, on="customer_id", how="left").select(
            pl.when(pl.lit(c + h > data_end)).then(None).otherwise(pl.col("_b").is_null().cast(pl.Int8)).alias(f"churn_{h}d")
        )
        out = out.with_columns(col)

    # survival: time from c to the last purchase before the first `gap`-day silence
    seq = pl.concat([
        pop.select("customer_id", pl.lit(c).cast(pl.Int32).alias("day")),
        future.with_columns(pl.col("day").cast(pl.Int32)),
    ]).sort("customer_id", "day")
    seq = seq.with_columns(
        pl.col("day").shift(-1).over("customer_id").fill_null(data_end).alias("next_day")
    ).with_columns((pl.col("next_day") - pl.col("day")).alias("gap"))
    ev = seq.filter(pl.col("gap") >= gap).group_by("customer_id").agg(pl.col("day").min().alias("leave_day"))
    out = out.join(ev, on="customer_id", how="left").with_columns(
        # leaving is observed only if the gap after leave_day is fully inside the data
        pl.when(pl.col("leave_day").is_not_null() & (pl.col("leave_day") + gap <= data_end))
        .then(pl.col("leave_day") - c).otherwise(data_end - gap - c).clip(lower_bound=0).alias("surv_duration"),
        (pl.col("leave_day").is_not_null() & (pl.col("leave_day") + gap <= data_end)).cast(pl.Int8).alias("surv_event"),
    ).drop("leave_day")
    return out


def build() -> pl.DataFrame:
    cfg = load_config()
    ch = cfg.churn
    t = load_silver(cfg)
    data_end = int(t["sales"]["day"].max())
    cutoffs = list(range(ch.first_cutoff_day, data_end + 1, ch.cutoff_step_days))
    if cutoffs[-1] != data_end:
        cutoffs.append(data_end)  # latest snapshot = the one we score this week
    frames = []
    for c in cutoffs:
        f = snapshot_features(t, c, cfg)
        lab = labels(t["sales"], f, c, ch.horizons, data_end, ch.primary_horizon)
        f = f.join(lab, on="customer_id", how="left")
        frames.append(f)
        log.info("cutoff %d (%s): %d customers, churn_%dd=%s", c, cfg.day_to_date(c), f.height,
                 ch.primary_horizon, f[f"churn_{ch.primary_horizon}d"].mean())
    out = pl.concat(frames, how="diagonal").sort("cutoff_day", "customer_id")
    path = cfg.path("gold") / "snapshots.parquet"
    out.write_parquet(path)
    log.info("snapshots: %d rows, %d cols -> %s", out.height, out.width, path)
    return out


if __name__ == "__main__":
    build()
