import pandas as pd
import polars as pl
import pytest

from churn.cleaning import customers as cust
from churn.cleaning import transactions as txn


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("John Smith", ("John", "Smith")),
        ("SMITH, JOHN", ("John", "Smith")),
        ("  john   smith ", ("John", "Smith")),
        ("Mary Ann Lee", ("Mary", "Ann Lee")),
        ("", (None, None)),
        (None, (None, None)),
    ],
)
def test_split_name(raw, expected):
    assert cust.split_name(raw) == expected


@pytest.mark.parametrize("raw", ["(212) 456-7890", "212.456.7890", "+1 212 456 7890", "12124567890", "212 4567890"])
def test_phone_formats_normalise_to_same_e164(raw):
    assert cust.normalise_phone(raw) == "+12124567890"


def test_invalid_phone_is_dropped():
    assert cust.normalise_phone("123") is None
    assert cust.normalise_phone(None) is None


def test_email_and_dates():
    assert cust.normalise_email("  John.Smith@GMAIL.com ") == "john.smith@gmail.com"
    assert cust.normalise_email("not-an-email") is None
    for raw in ["2024-03-05", "03/05/2024", "05-Mar-2024"]:
        assert cust.parse_date(raw) == pd.Timestamp("2024-03-05")
    assert pd.isna(cust.parse_date("31/31/2024"))


def _crm(rows):
    base = {c: None for c in cust.DEMO_COLS}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_resolve_links_reformatted_duplicate_but_not_namesake():
    raw = _crm([
        {"card_id": "LC00000001", "customer_name": "William Turner", "phone": "(212) 456-7890",
         "email": "w.turner@gmail.com", "signup_date": "2024-01-02", "home_store": "10"},
        # same person: nickname, other phone format, uppercase email, later card
        {"card_id": "LC00000002", "customer_name": "TURNER, BILL", "phone": "212.456.7890",
         "email": "W.TURNER@GMAIL.COM", "signup_date": "06/01/2024", "home_store": "10"},
        # namesake: same name, different contact details and store
        {"card_id": "LC00000003", "customer_name": "William Turner", "phone": "(415) 555-0199",
         "email": "bill.t@yahoo.com", "signup_date": "2024-02-01", "home_store": "77"},
    ])
    golden, card_map, _ = cust.resolve(raw)
    ids = card_map.set_index("card_id").customer_id
    assert ids["LC00000001"] == ids["LC00000002"]
    assert ids["LC00000003"] != ids["LC00000001"]
    merged = golden.set_index("customer_id").loc[ids["LC00000001"]]
    assert merged.n_cards == 2
    assert merged.first_name == "William"  # earliest signup card survives


def _legacy(rows):
    cols = ["INVOICE_NO", "LINE_NO", "TXN_DATE", "TXN_TIME", "STORE_ID", "CARD_ID", "PRODUCT_ID", "QTY",
            "SALES_VALUE", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC", "TXN_TYPE", "ORIG_INVOICE_NO",
            "PAYMENT_METHOD", "_source_file"]
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows], schema={c: pl.Utf8 for c in cols})


def _v2(rows):
    cols = ["invoice_id", "line", "txn_ts", "branch_code", "loyalty_card", "sku", "quantity", "net_amount",
            "promo_disc", "coupon_disc", "coupon_match_disc", "ref_invoice", "tender", "_source_file"]
    return pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows], schema={c: pl.Utf8 for c in cols})


def test_transaction_cleaning_end_to_end():
    sale = ["100", "1", "2024-01-05", "905", "1", "LC00000001", "1", "2", "4.00", "-0.5", "0", "0", "SALE", None, "CASH", "a.csv"]
    legacy = _legacy([
        sale,
        sale[:-1] + ["a_reload.csv"],                                                   # double-loaded
        ["101", "1", "2024-01-05", "1200", "9999", "LC00000001", "1", "1", "2.00", "0", "0", "0", "SALE", None, "CASH", "a.csv"],  # test store
        ["102", "1", "2024-01-06", "1200", "1", None, "1", "1", "2.00", "0", "0", "0", "SALE", None, "CASH", "a.csv"],  # no card
        ["103", "1", "2024-01-07", "1200", "1", "LC00000001", "1", "1", "200.00", "0.3", "0", "0", "SALE", None, "CARD", "a.csv"],  # x100, +disc
    ] + [[str(200 + i), "1", "2024-01-08", "1000", "1", "LC00000001", "1", "1", "2.00", "0", "0", "0", "SALE", None, "CARD", "a.csv"] for i in range(5)])
    v2 = _v2([
        ["INV-300", "1", "10/02/2024 09:05", "BR00001", "LC00000001", "1", "-1", "-2.00", "0", "0", "0", "INV-100", "Wallet", "b.csv"],
        ["INV-301", "1", "11/02/2024 09:05", "BR00001", "LC00000001", "2", "1", "1,234.50", "0", "0", "0", None, "Cash", "b.csv"],
    ])
    card_map = pl.DataFrame({"card_id": ["LC00000001"], "customer_id": ["C000001"]})
    products = pl.DataFrame({"PRODUCT_ID": ["1"], "DEPARTMENT": ["GROCERY"], "COMMODITY_DESC": ["MILK"]})
    lines, rep = txn.clean(legacy, v2, card_map, products, test_store_id=9999, outlier_quantile=1.0)

    assert rep["duplicate_lines_dropped"] == 1
    assert rep["test_lines_dropped"] == 1
    assert rep["unscanned_card_lines_dropped"] == 1
    assert rep["price_typo_fixed"] == 1
    assert rep["positive_discount_fixed"] == 1
    assert rep["return_lines"] == 1 and rep["orphan_return_lines"] == 0

    by_inv = {r["invoice_id"]: r for r in lines.iter_rows(named=True)}
    assert by_inv[100]["txn_ts"].strftime("%H:%M") == "09:05"   # "905" -> 09:05
    assert by_inv[103]["sales_value"] == pytest.approx(2.0)
    assert by_inv[300]["is_return"] and by_inv[300]["store_id"] == 1
    assert by_inv[300]["txn_ts"].month == 2                        # dd/mm/yyyy, not mm/dd
    assert by_inv[301]["sales_value"] == pytest.approx(1234.5)
    assert by_inv[301]["payment_method"] == "cash"
