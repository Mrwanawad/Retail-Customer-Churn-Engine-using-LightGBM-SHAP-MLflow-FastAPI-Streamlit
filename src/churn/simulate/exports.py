"""Turn the Dunnhumby Complete Journey into the exports a real retailer's systems would hand us.

Dunnhumby is already a curated research panel: one clean household key, integer days, no duplicates.
Production feeds never look like that. This module rebuilds the data as it would land from four
source systems (CRM, POS, loyalty, marketing), with the defects those systems really produce:

CRM (customer master)
  - free-text names in mixed formats ("SMITH, JOHN", " john  smith ")
  - phones in every format, emails with stray case/whitespace, dates in 3 formats
  - ~6% of households registered twice (second loyalty card, name typo/nickname,
    reformatted phone), so their baskets are split across two card ids
  - fields written after churn (account_status, closed_date) and a current-state tier: leakage traps
POS (monthly line-level exports)
  - vendor upgrade mid-history: column names, timestamp and branch formats change
  - two months loaded twice, card not scanned on some baskets, test-store lines
  - returns as negative lines (some without a matching sale), price keyed x100
Loyalty: points ledger (earn / redeem / expire)
Marketing: campaigns with offer cost, targets, coupon redemptions (real Dunnhumby campaign data)
Complaints: synthetic tickets (Dunnhumby has none)

Real behaviour (what, when, where, how much each household bought) is untouched. Only identity,
formats and system defects are simulated. Ground truth for identity resolution is written to
exports/_truth so dedup quality can be measured; the pipeline never reads it.
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from churn.config import get_logger, load_config

log = get_logger(__name__)

AREA_CODES = [
    212, 312, 415, 617, 713, 214, 404, 305, 206, 303, 602, 702, 816, 913, 314, 502, 615, 919, 704,
    513, 614, 216, 412, 215, 410, 202, 813, 407, 504, 210, 512, 801, 503, 916, 619, 858, 949, 714,
]
NICKNAMES = {
    "William": "Bill", "Robert": "Bob", "Richard": "Rick", "James": "Jim", "Joseph": "Joe",
    "Thomas": "Tom", "Michael": "Mike", "Elizabeth": "Liz", "Jennifer": "Jen", "Katherine": "Kate",
    "Margaret": "Peggy", "Christopher": "Chris", "Daniel": "Dan", "Matthew": "Matt", "Anthony": "Tony",
    "Patricia": "Pat", "Rebecca": "Becky", "Steven": "Steve", "Jonathan": "Jon", "Samantha": "Sam",
}
CAMPAIGN_COST = {"TypeA": 8.0, "TypeB": 4.0, "TypeC": 2.5}
CAMPAIGN_CHANNEL = {"TypeA": "direct_mail", "TypeB": "email", "TypeC": "in_store"}


# ---------------------------------------------------------------- helpers
def _read_dunnhumby(src: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(src / f"{name}.csv", low_memory=False)


def _phone_digits(rng: np.random.Generator, n: int) -> np.ndarray:
    area = rng.choice(AREA_CODES, n)
    exch = rng.integers(200, 1000, n)
    exch = np.where(exch == 555, 556, exch)
    line = rng.integers(0, 10000, n)
    return np.array([f"{a}{e:03d}{ln:04d}" for a, e, ln in zip(area, exch, line, strict=True)])


def _format_phone(d: str, style: int) -> str:
    a, e, ln = d[:3], d[3:6], d[6:]
    return [
        f"({a}) {e}-{ln}", f"{a}-{e}-{ln}", f"{a}.{e}.{ln}", f"+1 {a} {e} {ln}", d, f"1{d}", f"{a} {e}{ln}",
    ][style]


def _format_name(first: str, last: str, style: int) -> str:
    return [
        f"{first} {last}", f"{first} {last}", f"{last.upper()}, {first.upper()}", f" {first.lower()}  {last.lower()}",
        f"{first.upper()} {last.upper()}", f"{last}, {first}",
    ][style]


def _typo(rng: np.random.Generator, s: str) -> str:
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 1))
    op = rng.integers(0, 3)
    if op == 0:  # drop a letter
        return s[:i] + s[i + 1 :]
    if op == 1:  # swap neighbours
        return s[: i - 1] + s[i] + s[i - 1] + s[i + 1 :]
    return s[:i] + rng.choice(list(string.ascii_lowercase)) + s[i + 1 :]  # substitute


def _format_date(d: pd.Timestamp, style: int) -> str:
    return [d.strftime("%Y-%m-%d"), d.strftime("%m/%d/%Y"), d.strftime("%d-%b-%Y")][style]


# ---------------------------------------------------------------- CRM
def build_customers(cfg, tx: pd.DataFrame, demo: pd.DataFrame, rng: np.random.Generator):
    sim = cfg.simulation
    fake = Faker("en_US")
    Faker.seed(cfg.seed)

    hh = (
        tx.groupby("household_key")
        .agg(first_day=("DAY", "min"), last_day=("DAY", "max"))
        .reset_index()
    )
    first_store = (
        tx[tx.DAY <= tx.groupby("household_key").DAY.transform("min") + 30]
        .groupby("household_key").STORE_ID.agg(lambda s: s.value_counts().index[0])
    )
    hh["home_store"] = hh.household_key.map(first_store)
    n = len(hh)

    hh["first_name"] = [fake.first_name() for _ in range(n)]
    hh["last_name"] = [fake.last_name() for _ in range(n)]
    hh["phone"] = _phone_digits(rng, n)
    hh["email"] = [
        f"{f.lower()}.{la.lower()}{rng.integers(1, 99)}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'aol.com', 'icloud.com'])}"
        for f, la in zip(hh.first_name, hh.last_name, strict=True)
    ]
    card_pool = rng.choice(np.arange(10_000_000, 99_999_999), int(n * 1.2), replace=False)
    hh["card_id"] = [f"LC{x:08d}" for x in card_pool[:n]]
    # signup happens before the first observed basket (panel recruitment)
    hh["signup_day"] = hh.first_day - rng.integers(0, 420, n)

    # current-state fields written at export time -> must NOT be used as features
    spend12 = tx[tx.DAY > tx.DAY.max() - 365].groupby("household_key").SALES_VALUE.sum()
    s = hh.household_key.map(spend12).fillna(0)
    hh["loyalty_tier"] = np.select([s > 4000, s > 1500, s > 300], ["Platinum", "Gold", "Silver"], "Bronze")
    gap_at_end = tx.DAY.max() - hh.last_day
    closed = (gap_at_end > 60) & (rng.random(n) < 0.35)
    hh["account_status"] = np.where(closed, "CLOSED", np.where(gap_at_end > 60, "DORMANT", "ACTIVE"))
    hh["closed_day"] = np.where(closed, hh.last_day + rng.integers(20, 60, n), np.nan)

    hh = hh.merge(demo, on="household_key", how="left")

    # ---- duplicate registrations (same household, second card)
    dup_mask = rng.random(n) < sim.duplicate_customer_rate
    d = hh[dup_mask].copy()
    m = len(d)
    d["card_id"] = [f"LC{x:08d}" for x in card_pool[n : n + m]]
    d["signup_day"] = d.first_day + rng.integers(30, 300, m)
    variant = rng.integers(0, 4, m)
    d["first_name"] = [
        NICKNAMES.get(f, f) if v == 0 else (_typo(rng, f) if v == 1 else f)
        for f, v in zip(d.first_name, variant, strict=True)
    ]
    d["last_name"] = [
        _typo(rng, la) if v == 2 else la for la, v in zip(d.last_name, variant, strict=True)
    ]
    email_mode = rng.random(m)
    d["email"] = np.where(
        email_mode < 0.5, d.email.str.upper(),
        np.where(email_mode < 0.8, [f"{f.lower()}{rng.integers(100, 999)}@gmail.com" for f in d.first_name], None),
    )
    d["phone"] = np.where(rng.random(m) < 0.8, d.phone, None)
    d["loyalty_tier"] = "Bronze"
    d["is_duplicate_registration"] = True
    hh["is_duplicate_registration"] = False

    crm = pd.concat([hh, d], ignore_index=True)
    N = len(crm)

    # ---- format noise
    name_style = rng.choice(6, N, p=[0.55, 0.15, 0.1, 0.08, 0.07, 0.05])
    crm["customer_name"] = [
        _format_name(f, la, st) for f, la, st in zip(crm.first_name, crm.last_name, name_style, strict=True)
    ]
    phone_style = rng.integers(0, 7, N)
    crm["phone"] = [
        None if p is None or rng.random() < 0.03 else _format_phone(p, st)
        for p, st in zip(crm.phone, phone_style, strict=True)
    ]
    crm["email"] = [
        None if e is None or rng.random() < 0.05 else (f"  {e} " if rng.random() < 0.05 else e) for e in crm.email
    ]
    date_style = rng.choice(3, N, p=[0.6, 0.3, 0.1])
    signup = pd.Timestamp(cfg.day_one) + pd.to_timedelta(crm.signup_day - 1, unit="D")
    crm["signup_date"] = [
        None if rng.random() < 0.005 else _format_date(dt, st) for dt, st in zip(signup, date_style, strict=True)
    ]
    closed_dt = pd.Timestamp(cfg.day_one) + pd.to_timedelta(crm.closed_day - 1, unit="D")
    crm["closed_date"] = closed_dt.dt.strftime("%Y-%m-%d")
    tier_noise = rng.random(N)
    crm["loyalty_tier"] = np.where(
        tier_noise < 0.1, crm.loyalty_tier.str.upper(),
        np.where(tier_noise < 0.15, crm.loyalty_tier.str.lower() + " ", np.where(tier_noise < 0.17, None, crm.loyalty_tier)),
    )
    crm["home_store"] = crm.home_store.astype("Int64").astype(str)
    crm["created_by_system"] = rng.choice(["POS_ENROLL", "WEB", "CALL_CENTER"], N, p=[0.7, 0.2, 0.1])

    cols = [
        "card_id", "customer_name", "phone", "email", "signup_date", "home_store", "loyalty_tier",
        "account_status", "closed_date", "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC", "HOMEOWNER_DESC",
        "HH_COMP_DESC", "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC", "created_by_system",
    ]
    out = crm.sample(frac=1, random_state=cfg.seed)[cols]
    truth = crm[["card_id", "household_key", "is_duplicate_registration", "signup_day"]]
    return out, truth


# ---------------------------------------------------------------- POS
def build_pos(cfg, tx: pd.DataFrame, truth: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sim = cfg.simulation
    tx = tx.copy()

    # card attribution: duplicate households split baskets across their two cards
    primary = truth[~truth.is_duplicate_registration].set_index("household_key").card_id
    dup = truth[truth.is_duplicate_registration].set_index("household_key")
    tx["CARD_ID"] = tx.household_key.map(primary)
    baskets = tx[["BASKET_ID", "household_key", "DAY"]].drop_duplicates("BASKET_ID")
    b_dup = baskets.household_key.isin(dup.index)
    b_dup &= baskets.DAY >= baskets.household_key.map(dup.signup_day).fillna(10**6)
    b_dup &= rng.random(len(baskets)) < sim.card_split_share
    tx.loc[tx.BASKET_ID.isin(baskets.BASKET_ID[b_dup]), "CARD_ID"] = tx.household_key.map(dup.card_id)
    # card not scanned
    unscanned = baskets.BASKET_ID[rng.random(len(baskets)) < sim.walk_in_line_rate]
    tx.loc[tx.BASKET_ID.isin(unscanned), "CARD_ID"] = None

    # payment method: household habit, basket-level draw
    hh_ids = tx.household_key.unique()
    pref = pd.Series(rng.dirichlet([3, 4, 1], len(hh_ids)).tolist(), index=hh_ids)
    b_pay = [
        rng.choice(["CASH", "CARD", "MOBILE"], p=pref[h]) for h in baskets.household_key
    ]
    tx["PAYMENT_METHOD"] = tx.BASKET_ID.map(pd.Series(b_pay, index=baskets.BASKET_ID.values))
    tx["TXN_TYPE"] = "SALE"
    tx["LINE_NO"] = tx.groupby("BASKET_ID").cumcount() + 1

    # price keyed x100
    typo = rng.random(len(tx)) < sim.price_typo_rate
    tx.loc[typo, "SALES_VALUE"] = tx.loc[typo, "SALES_VALUE"] * 100

    # returns: a later negative line, mostly at the same store
    tx["ORIG_BASKET_ID"] = np.nan
    sold = tx[(tx.QUANTITY > 0) & (tx.QUANTITY <= 20) & (tx.SALES_VALUE > 0)]
    ret = sold.sample(frac=sim.return_line_rate, random_state=cfg.seed).copy()
    ret["DAY"] = (ret.DAY + rng.integers(1, 31, len(ret))).clip(upper=tx.DAY.max())
    unit_price = ret.SALES_VALUE / ret.QUANTITY
    ret["QUANTITY"] = -ret.QUANTITY.clip(upper=3)
    ret["SALES_VALUE"] = (unit_price * ret.QUANTITY).round(2)
    ret[["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]] = 0.0
    other_store = rng.random(len(ret)) < 0.2
    ret.loc[other_store, "STORE_ID"] = rng.choice(tx.STORE_ID.unique(), other_store.sum())
    ret["TXN_TYPE"] = "RETURN"
    ret["ORIG_BASKET_ID"] = ret.BASKET_ID
    ret["BASKET_ID"] = 90_000_000_000 + np.arange(len(ret))
    ret["LINE_NO"] = 1
    orphan = sold.sample(n=int(len(tx) * sim.orphan_return_rate), random_state=cfg.seed + 1).copy()
    orphan["QUANTITY"] = -1
    orphan["SALES_VALUE"] = -orphan.SALES_VALUE.round(2)
    orphan["TXN_TYPE"] = "RETURN"
    orphan["ORIG_BASKET_ID"] = np.nan
    orphan["BASKET_ID"] = 91_000_000_000 + np.arange(len(orphan))
    orphan["LINE_NO"] = 1

    # test / training-mode store
    test = tx.sample(n=600, random_state=cfg.seed + 2).copy()
    test["STORE_ID"] = sim.test_store_id
    test["CARD_ID"] = "LC00000000"
    test["BASKET_ID"] = 99_000_000_000 + rng.integers(0, 40, len(test))

    pos = pd.concat([tx, ret, orphan, test], ignore_index=True)
    pos["TRANS_TIME"] = pos.TRANS_TIME.astype(int)
    pos["date"] = pd.Timestamp(cfg.day_one) + pd.to_timedelta(pos.DAY - 1, unit="D")
    return pos


def write_pos_exports(cfg, pos: pd.DataFrame, out: Path, rng: np.random.Generator) -> list[Path]:
    """Monthly CSV files as the POS ETL drops them. The vendor upgrade changes the file layout."""
    sim = cfg.simulation
    pos_dir = out / "pos"
    pos_dir.mkdir(parents=True, exist_ok=True)
    for f in pos_dir.glob("*.csv"):
        f.unlink()
    pos["month"] = pos.date.dt.strftime("%Y_%m")
    months = sorted(pos.month.unique())
    reloads = set(rng.choice(months[1:-1], sim.double_load_months, replace=False))
    written = []
    for month, g in pos.groupby("month", sort=True):
        g = g.sort_values(["date", "TRANS_TIME", "BASKET_ID", "LINE_NO"])
        old = g[g.DAY < sim.schema_change_day]
        new = g[g.DAY >= sim.schema_change_day]
        parts = []
        if len(old):
            parts.append(("legacy", pd.DataFrame({
                "INVOICE_NO": old.BASKET_ID.astype("int64"),
                "LINE_NO": old.LINE_NO,
                "TXN_DATE": old.date.dt.strftime("%Y-%m-%d"),
                "TXN_TIME": old.TRANS_TIME,  # HHMM integer, leading zeros lost
                "STORE_ID": old.STORE_ID,
                "CARD_ID": old.CARD_ID,
                "PRODUCT_ID": old.PRODUCT_ID,
                "QTY": old.QUANTITY,
                "SALES_VALUE": old.SALES_VALUE.round(2),
                "RETAIL_DISC": old.RETAIL_DISC,
                "COUPON_DISC": old.COUPON_DISC,
                "COUPON_MATCH_DISC": old.COUPON_MATCH_DISC,
                "TXN_TYPE": old.TXN_TYPE,
                "ORIG_INVOICE_NO": old.ORIG_BASKET_ID.astype("Int64"),
                "PAYMENT_METHOD": old.PAYMENT_METHOD,
            })))
        if len(new):
            hhmm = new.TRANS_TIME.astype(int)
            ts = new.date + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
            parts.append(("v2", pd.DataFrame({
                "invoice_id": "INV-" + new.BASKET_ID.astype("int64").astype(str),
                "line": new.LINE_NO,
                "txn_ts": ts.dt.strftime("%d/%m/%Y %H:%M"),
                "branch_code": "BR" + new.STORE_ID.astype(int).astype(str).str.zfill(5),
                "loyalty_card": new.CARD_ID,
                "sku": new.PRODUCT_ID,
                "quantity": new.QUANTITY,
                "net_amount": new.SALES_VALUE.map(lambda v: f"{v:,.2f}"),  # thousands separator in text
                "promo_disc": new.RETAIL_DISC,
                "coupon_disc": new.COUPON_DISC,
                "coupon_match_disc": new.COUPON_MATCH_DISC,
                "ref_invoice": ("INV-" + new.ORIG_BASKET_ID.astype("Int64").astype(str)).where(
                    new.ORIG_BASKET_ID.notna()
                ),
                "tender": new.PAYMENT_METHOD.str.lower().map({"cash": "Cash", "card": "Credit/Debit", "mobile": "Wallet"}),
            })))
        for layout, df in parts:
            p = pos_dir / f"pos_{layout}_{month}.csv"
            df.to_csv(p, index=False)
            written.append(p)
            if month in reloads:
                p2 = pos_dir / f"pos_{layout}_{month}_reload.csv"
                df.to_csv(p2, index=False)
                written.append(p2)
    return written


# ---------------------------------------------------------------- loyalty, complaints, marketing
def build_points_ledger(cfg, pos: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sales = pos[(pos.TXN_TYPE == "SALE") & pos.CARD_ID.notna() & (pos.STORE_ID != cfg.simulation.test_store_id)]
    b = sales.groupby(["CARD_ID", "BASKET_ID", "date"]).SALES_VALUE.sum().reset_index().sort_values(["CARD_ID", "date"])
    b["points"] = np.floor(b.SALES_VALUE).astype(int)
    rows = []
    for card, g in b.groupby("CARD_ID", sort=False):
        bal = 0
        for dt, pts in zip(g.date.to_numpy(), g.points.to_numpy(), strict=True):
            rows.append((card, dt, "EARN", pts))
            bal += pts
            if bal >= 500 and rng.random() < 0.3:
                rows.append((card, dt, "REDEEM", -500))
                bal -= 500
    led = pd.DataFrame(rows, columns=["card_id", "event_date", "event_type", "points"])
    led["event_date"] = pd.to_datetime(led.event_date).dt.strftime("%Y-%m-%d")
    led.insert(0, "ledger_id", np.arange(1, len(led) + 1))
    return led


def build_complaints(cfg, tx: pd.DataFrame, truth: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Weekly ticket draw per household. Rate rises in the 8 weeks before a long lapse begins,
    which is how complaints relate to defection in practice (dissatisfaction precedes leaving)."""
    sim = cfg.simulation
    days = tx[["household_key", "DAY"]].drop_duplicates().sort_values(["household_key", "DAY"])
    days["next_gap"] = days.groupby("household_key").DAY.shift(-1) - days.DAY
    lapse = days[days.next_gap > 60].groupby("household_key").DAY.apply(list)
    span = tx.groupby("household_key").DAY.agg(["min", "max"])
    primary = truth[~truth.is_duplicate_registration].set_index("household_key").card_id
    cats = ["product_quality", "out_of_stock", "checkout_wait", "staff", "pricing", "loyalty_points", "refund"]
    rows = []
    for h, (lo, hi) in span.iterrows():
        lapse_days = np.array(lapse.get(h, []))
        for wk_start in range(int(lo), int(hi) + 1, 7):
            rate = sim.complaint_base_rate
            if len(lapse_days) and np.any((lapse_days - wk_start >= 0) & (lapse_days - wk_start < 56)):
                rate *= 6
            if rng.random() < rate * 7:
                day = wk_start + int(rng.integers(0, 7))
                rows.append((primary[h], day, rng.choice(cats), rng.choice(["phone", "store", "email", "app"])))
    c = pd.DataFrame(rows, columns=["card_id", "day", "category", "channel"])
    c["created_at"] = (
        pd.Timestamp(cfg.day_one) + pd.to_timedelta(c.day - 1, unit="D")
        + pd.to_timedelta(rng.integers(8 * 60, 21 * 60, len(c)), unit="m")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    c["resolved"] = rng.random(len(c)) < 0.85
    c["category"] = np.where(rng.random(len(c)) < 0.1, c.category.str.upper().str.replace("_", " "), c.category)
    c.insert(0, "ticket_id", [f"TCK-{i:06d}" for i in range(1, len(c) + 1)])
    return c.drop(columns="day")


def build_marketing(cfg, src: Path, truth: pd.DataFrame):
    day = lambda s: (pd.Timestamp(cfg.day_one) + pd.to_timedelta(s - 1, unit="D")).dt.strftime("%Y-%m-%d")  # noqa: E731
    desc = _read_dunnhumby(src, "campaign_desc")
    camp = pd.DataFrame({
        "campaign_id": desc.CAMPAIGN,
        "campaign_type": desc.DESCRIPTION,
        "channel": desc.DESCRIPTION.map(CAMPAIGN_CHANNEL),
        "start_date": day(desc.START_DAY),
        "end_date": day(desc.END_DAY),
        "offer_cost_per_customer": desc.DESCRIPTION.map(CAMPAIGN_COST),
    }).sort_values("campaign_id")
    primary = truth[~truth.is_duplicate_registration].set_index("household_key").card_id
    tgt = _read_dunnhumby(src, "campaign_table")
    targets = pd.DataFrame({"campaign_id": tgt.CAMPAIGN, "card_id": tgt.household_key.map(primary)})
    red = _read_dunnhumby(src, "coupon_redempt")
    redemptions = pd.DataFrame({
        "card_id": red.household_key.map(primary),
        "redeemed_date": day(red.DAY),
        "coupon_upc": red.COUPON_UPC,
        "campaign_id": red.CAMPAIGN,
    })
    coupons = _read_dunnhumby(src, "coupon").rename(columns=str.lower)
    return camp, targets, redemptions, coupons


# ---------------------------------------------------------------- entry point
def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg.seed)
    src, out = cfg.path("dunnhumby"), cfg.path("exports")
    for sub in ("crm", "pos", "loyalty", "marketing", "catalog", "_truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    log.info("reading Dunnhumby")
    tx = _read_dunnhumby(src, "transaction_data")
    demo = _read_dunnhumby(src, "hh_demographic")

    log.info("CRM customer master")
    crm, truth = build_customers(cfg, tx, demo, rng)
    crm.to_csv(out / "crm" / "customers.csv", index=False)
    truth.to_csv(out / "_truth" / "card_household.csv", index=False)

    log.info("POS exports")
    pos = build_pos(cfg, tx, truth, rng)
    files = write_pos_exports(cfg, pos, out, rng)
    log.info("  %d POS files", len(files))

    log.info("loyalty ledger")
    build_points_ledger(cfg, pos, rng).to_csv(out / "loyalty" / "points_ledger.csv", index=False)

    log.info("complaints")
    build_complaints(cfg, tx, truth, rng).to_csv(out / "crm" / "complaints.csv", index=False)

    log.info("marketing")
    camp, targets, redemptions, coupons = build_marketing(cfg, src, truth)
    camp.to_csv(out / "marketing" / "campaigns.csv", index=False)
    targets.to_csv(out / "marketing" / "campaign_targets.csv", index=False)
    redemptions.to_csv(out / "marketing" / "coupon_redemptions.csv", index=False)
    coupons.to_csv(out / "marketing" / "coupons.csv", index=False)

    _read_dunnhumby(src, "product").to_csv(out / "catalog" / "products.csv", index=False)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
