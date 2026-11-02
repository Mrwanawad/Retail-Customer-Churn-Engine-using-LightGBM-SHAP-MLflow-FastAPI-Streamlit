from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

from churn.config import load_config
from churn.decision import expected_profit, risk_segment, select_targets
from churn.features import DEMO_COLS, labels, snapshot_features
from churn.modeling import out_of_time_split


def _tables(max_day: int) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(0)
    rows = []
    for cust, every in (("C000001", 3), ("C000002", 10), ("C000003", 7)):
        for day in range(1, 401, every):
            if cust == "C000003" and day > 150:
                break  # goes silent after day 150
            rows.append({
                "invoice_id": len(rows), "customer_id": cust, "day": day, "is_return": False,
                "txn_ts": datetime(2024, 1, 1) + timedelta(days=day - 1, hours=10),
                "store_id": 1, "payment_method": "card", "basket_value": float(rng.uniform(10, 80)),
                "discount_total": 1.0, "coupon_used": False, "n_items": 5.0, "n_lines": 5,
                "n_departments": 2, "n_commodities": 4,
            })
    sales = pl.DataFrame(rows).with_columns(pl.col("day").cast(pl.Int32))
    sales = sales.filter(pl.col("day") <= max_day)
    empty = lambda schema: pl.DataFrame(schema=schema)  # noqa: E731
    cust = pl.DataFrame({
        "customer_id": ["C000001", "C000002", "C000003"], "n_cards": [1, 2, 1],
        "signup_date": [datetime(2023, 6, 1)] * 3, **{d: [None, "x", None] for d in DEMO_COLS},
    }, schema_overrides={d: pl.Utf8 for d in DEMO_COLS})
    return {
        "sales": sales, "returns": sales.head(0),
        "complaints": empty({"customer_id": pl.Utf8, "day": pl.Int32, "resolved": pl.Boolean}),
        "ledger": empty({"customer_id": pl.Utf8, "day": pl.Int32, "points": pl.Int64, "event_type": pl.Utf8}),
        "targets": empty({"customer_id": pl.Utf8, "day": pl.Int32}),
        "redemptions": empty({"customer_id": pl.Utf8, "day": pl.Int32}),
        "customers": cust,
    }


def test_features_are_point_in_time():
    """Adding data after the cutoff must not change any feature value."""
    cfg = load_config()
    c = 140
    full = snapshot_features(_tables(400), c, cfg).sort("customer_id")
    trunc = snapshot_features(_tables(c), c, cfg).sort("customer_id")
    assert full.equals(trunc)


def test_labels_and_survival():
    t = _tables(400)
    pop = pl.DataFrame({"customer_id": ["C000001", "C000002", "C000003"]})
    lab = labels(t["sales"], pop, c=150, horizons=[30, 60, 300], data_end=400, gap=60).sort("customer_id")
    d = dict(zip(lab["customer_id"], lab.iter_rows(named=True), strict=True))
    assert d["C000001"]["churn_60d"] == 0
    assert d["C000003"]["churn_60d"] == 1        # last purchase day 148, nothing in (150, 210]
    assert d["C000001"]["churn_300d"] is None    # window not fully observed
    assert d["C000003"]["surv_event"] == 1
    assert d["C000001"]["surv_event"] == 0


def test_out_of_time_split_purges_overlapping_labels():
    cut = np.repeat(np.arange(100, 600, 28), 10)
    snap = pd.DataFrame({"customer_id": np.tile([f"C{i:06d}" for i in range(10)], len(cut) // 10),
                         "cutoff_day": cut, "churn_60d": 0.0})
    snap.loc[snap.cutoff_day > 600 - 60, "churn_60d"] = np.nan
    s = out_of_time_split(snap, "churn_60d", 60, n_valid=2, n_test=2)
    assert s.train.cutoff_day.max() + 60 <= s.valid.cutoff_day.min()
    assert s.valid.cutoff_day.max() + 60 <= s.test.cutoff_day.min()
    assert s.score.cutoff_day.unique().tolist() == [cut.max()]


def test_decision_layer():
    assert expected_profit([0.5], [100], 0.2, 5)[0] == pytest.approx(5.0)
    df = pd.DataFrame({"churn_probability": [0.6, 0.05, 0.3], "clv": [200, 1000, 10]})
    out = select_targets(df, success=0.25, cost=5).set_index("clv")
    assert out.loc[200, "targeted"] and out.loc[1000, "targeted"] and not out.loc[10, "targeted"]
    assert risk_segment([0.01, 0.1, 0.3, 0.9]).tolist() == ["Low", "Medium", "High", "Critical"]
