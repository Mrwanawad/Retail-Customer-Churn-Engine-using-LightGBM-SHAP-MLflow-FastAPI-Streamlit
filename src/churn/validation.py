"""Data contracts (pandera).

Two layers:
  raw contracts   on bronze text columns: shape of each value as delivered. Row failures are counted and
                  reported, not fixed. A table fails only if a required column is missing or the failure
                  share passes `max_fail_share`: that means the source changed, not that rows are dirty.
  silver schemas  on cleaned, typed tables: hard guarantees the feature layer relies on
                  (types, ranges, keys unique, no nulls where none allowed). Any failure stops the run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd
import pandera.pandas as pa
import pandera.polars as papl
import polars as pl

from churn.config import get_logger, load_config

log = get_logger(__name__)

INT = r"^-?\d+$"
DEC = r"^-?\d+(\.\d+)?$"
ISO_DATE = r"^\d{4}-\d{2}-\d{2}$"
CARD = r"^LC\d{8}$"


def _col(pattern: str | None = None, nullable: bool = False, isin: list[str] | None = None, **kw) -> papl.Column:
    checks = []
    if pattern:
        checks.append(papl.Check.str_matches(pattern))
    if isin:
        checks.append(papl.Check.isin(isin))
    return papl.Column(pl.Utf8, checks=checks, nullable=nullable, **kw)


RAW_CONTRACTS: dict[str, papl.DataFrameSchema] = {
    "crm_customers": papl.DataFrameSchema({
        "card_id": _col(CARD, unique=True),
        "customer_name": _col(),
        "phone": _col(r"^[\d\s\(\)\.\-\+]{10,17}$", nullable=True),
        "email": _col(r"^\s*[^@\s]+@[^@\s]+\.[A-Za-z]{2,}\s*$", nullable=True),
        "signup_date": _col(r"^(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-[A-Za-z]{3}-\d{4})$", nullable=True),
        "home_store": _col(INT, nullable=True),
        "loyalty_tier": _col(nullable=True),
        "account_status": _col(isin=["ACTIVE", "DORMANT", "CLOSED"], nullable=True),
    }),
    "pos_legacy": papl.DataFrameSchema({
        "INVOICE_NO": _col(INT),
        "LINE_NO": _col(INT),
        "TXN_DATE": _col(ISO_DATE),
        "TXN_TIME": _col(r"^\d{1,4}$"),
        "STORE_ID": _col(INT),
        "CARD_ID": _col(CARD, nullable=True),
        "PRODUCT_ID": _col(INT),
        "QTY": _col(INT),
        "SALES_VALUE": _col(DEC),
        "RETAIL_DISC": _col(DEC),
        "COUPON_DISC": _col(DEC),
        "COUPON_MATCH_DISC": _col(DEC),
        "TXN_TYPE": _col(isin=["SALE", "RETURN"]),
        "PAYMENT_METHOD": _col(isin=["CASH", "CARD", "MOBILE"], nullable=True),
    }),
    "pos_v2": papl.DataFrameSchema({
        "invoice_id": _col(r"^INV-\d+$"),
        "line": _col(INT),
        "txn_ts": _col(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$"),
        "branch_code": _col(r"^BR\d{5}$"),
        "loyalty_card": _col(CARD, nullable=True),
        "sku": _col(INT),
        "quantity": _col(INT),
        "net_amount": _col(r"^-?[\d,]+\.\d{2}$"),
        "promo_disc": _col(DEC),
        "coupon_disc": _col(DEC),
        "coupon_match_disc": _col(DEC),
        "tender": _col(isin=["Cash", "Credit/Debit", "Wallet"], nullable=True),
    }),
    "crm_complaints": papl.DataFrameSchema({
        "ticket_id": _col(r"^TCK-\d{6}$", unique=True),
        "card_id": _col(CARD),
        "category": _col(),
        "channel": _col(isin=["phone", "store", "email", "app"]),
        "created_at": _col(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
        "resolved": _col(isin=["True", "False"]),
    }),
    "loyalty_ledger": papl.DataFrameSchema({
        "ledger_id": _col(INT, unique=True),
        "card_id": _col(CARD),
        "event_date": _col(ISO_DATE),
        "event_type": _col(isin=["EARN", "REDEEM", "EXPIRE"]),
        "points": _col(INT),
    }),
    "mkt_campaigns": papl.DataFrameSchema({
        "campaign_id": _col(INT, unique=True),
        "campaign_type": _col(isin=["TypeA", "TypeB", "TypeC"]),
        "start_date": _col(ISO_DATE),
        "end_date": _col(ISO_DATE),
        "offer_cost_per_customer": _col(DEC),
    }),
    "mkt_targets": papl.DataFrameSchema({"campaign_id": _col(INT), "card_id": _col(CARD)}),
    "mkt_redemptions": papl.DataFrameSchema({
        "card_id": _col(CARD), "redeemed_date": _col(ISO_DATE), "coupon_upc": _col(INT), "campaign_id": _col(INT),
    }),
    "products": papl.DataFrameSchema({"PRODUCT_ID": _col(INT, unique=True), "DEPARTMENT": _col(nullable=True)}),
}


@dataclass
class ContractResult:
    table: str
    rows: int
    status: str  # pass | warn | fail
    missing_columns: list[str] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)  # "column:check" -> failing rows
    examples: dict[str, list] = field(default_factory=dict)


def check_raw(table: str, df: pl.DataFrame, max_fail_share: float = 0.02) -> ContractResult:
    schema = RAW_CONTRACTS[table]
    missing = [c for c in schema.columns if c not in df.columns]
    if missing:
        return ContractResult(table, df.height, "fail", missing_columns=missing)
    data = df.select(list(schema.columns))
    res = ContractResult(table, df.height, "pass")
    try:
        schema.validate(data, lazy=True)
    except papl.errors.SchemaErrors as err:
        fc = err.failure_cases
        fc = fc.to_pandas() if isinstance(fc, pl.DataFrame) else fc
        for (col, check), g in fc.groupby(["column", "check"], dropna=False):
            key = f"{col}:{check}"
            res.failures[key] = len(g)
            res.examples[key] = g["failure_case"].astype(str).head(5).tolist()
        worst = max(res.failures.values()) / max(df.height, 1)
        res.status = "fail" if worst > max_fail_share else "warn"
    return res


def run_raw_contracts(read_bronze) -> list[ContractResult]:
    cfg = load_config()
    out_dir = cfg.path("reports") / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for table in RAW_CONTRACTS:
        r = check_raw(table, read_bronze(table))
        results.append(r)
        log.info("contract %-16s %-4s %s", table, r.status, r.failures or "")
    (out_dir / "raw_contracts.json").write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    failed = [r.table for r in results if r.status == "fail"]
    if failed:
        raise RuntimeError(f"raw contract failed for {failed}; see data/reports/validation/raw_contracts.json")
    return results


# ---------------------------------------------------------------- silver (typed, strict)
class Customers(pa.DataFrameModel):
    customer_id: str = pa.Field(unique=True, str_matches=r"^C\d{6}$")
    first_name: str = pa.Field(nullable=True)
    last_name: str = pa.Field(nullable=True)
    phone_e164: str = pa.Field(nullable=True, str_matches=r"^\+1\d{10}$")
    email: str = pa.Field(nullable=True, str_matches=r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$")
    signup_date: pd.Timestamp = pa.Field(nullable=True)
    home_store: pd.Int64Dtype = pa.Field(nullable=True, ge=0)
    n_cards: int = pa.Field(ge=1)

    class Config:
        coerce = True


class CardMap(pa.DataFrameModel):
    card_id: str = pa.Field(unique=True, str_matches=r"^LC\d{8}$")
    customer_id: str = pa.Field(str_matches=r"^C\d{6}$")


class Transactions(pa.DataFrameModel):
    invoice_id: int = pa.Field(ge=0)
    line_no: int = pa.Field(ge=1)
    txn_ts: pd.Timestamp
    store_id: int = pa.Field(ge=0)
    customer_id: str = pa.Field(str_matches=r"^C\d{6}$")
    product_id: int
    quantity: float
    sales_value: float = pa.Field(ge=-5000, le=5000)
    discount_total: float = pa.Field(ge=0)
    is_return: bool
    payment_method: str = pa.Field(nullable=True, isin=["cash", "card", "mobile"])

    class Config:
        coerce = True

    @pa.dataframe_check
    def unique_line(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(["invoice_id", "line_no"]).any()

    @pa.dataframe_check
    def sign_matches_type(cls, df: pd.DataFrame) -> pd.Series:
        return (df.sales_value <= 0) | ~df.is_return


SILVER_SCHEMAS = {"customers": Customers, "card_map": CardMap, "transactions": Transactions}


def check_silver(name: str, df: pd.DataFrame) -> pd.DataFrame:
    return SILVER_SCHEMAS[name].validate(df, lazy=True)


if __name__ == "__main__":
    from churn.ingest import read_bronze

    run_raw_contracts(read_bronze)
