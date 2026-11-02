"""Customer master cleaning + identity resolution.

One person can hold several loyalty cards (re-registered at another branch, lost card, web signup).
If those cards stay separate, the person looks like two half-active customers and the older card
looks churned. Resolution:

  1. normalise   name (handles "LAST, FIRST", case, spacing), phone -> E.164, email, signup date
  2. block       candidate pairs share phone, email, or (last-name prefix + home store)
  3. score       rapidfuzz name similarity with nickname folding
  4. link        phone/email match + name sim >= 70, or name sim >= threshold with same home store
                 and same demographics; union-find over links -> customer_id
  5. survive     golden record: earliest signup card wins, later cards fill its gaps
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import phonenumbers
from rapidfuzz import fuzz

from churn.config import get_logger

log = get_logger(__name__)

NICK_TO_FORMAL = {
    "bill": "william", "bob": "robert", "rick": "richard", "jim": "james", "joe": "joseph", "tom": "thomas",
    "mike": "michael", "liz": "elizabeth", "jen": "jennifer", "kate": "katherine", "peggy": "margaret",
    "chris": "christopher", "dan": "daniel", "matt": "matthew", "tony": "anthony", "pat": "patricia",
    "becky": "rebecca", "steve": "steven", "jon": "jonathan", "sam": "samantha",
}
EMAIL_RE = re.compile(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$")
DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"]
DEMO_COLS = [
    "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC", "HOMEOWNER_DESC", "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC",
]
# written after (or because of) churn, or describe the customer at export time, not at the snapshot
LEAKAGE_COLS = ["account_status", "closed_date", "loyalty_tier"]


def split_name(raw: str | None) -> tuple[str | None, str | None]:
    if raw is None or not str(raw).strip():
        return None, None
    s = re.sub(r"\s+", " ", str(raw)).strip()
    if "," in s:
        last, first = (p.strip() for p in s.split(",", 1))
    else:
        parts = s.split(" ")
        first, last = parts[0], " ".join(parts[1:]) or None
    fix = lambda x: x.title() if x else None  # noqa: E731
    return fix(first), fix(last)


def normalise_phone(raw: str | None) -> str | None:
    if raw is None or pd.isna(raw):
        return None
    try:
        num = phonenumbers.parse(str(raw), "US")
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(num):
        return None
    return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164)


def normalise_email(raw: str | None) -> str | None:
    if raw is None or pd.isna(raw):
        return None
    e = str(raw).strip().lower()
    return e if EMAIL_RE.match(e) else None


def parse_date(raw: str | None) -> pd.Timestamp:
    if raw is None or pd.isna(raw):
        return pd.NaT
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(raw, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def _name_key(first: str | None, last: str | None) -> str:
    f = (first or "").lower()
    return f"{NICK_TO_FORMAL.get(f, f)} {(last or '').lower()}".strip()


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    names = df.customer_name.map(split_name)
    df["first_name"] = names.str[0]
    df["last_name"] = names.str[1]
    df["phone_e164"] = df.phone.map(normalise_phone)
    df["email"] = df.email.map(normalise_email)
    df["signup_date"] = df.signup_date.map(parse_date)
    df["home_store"] = pd.to_numeric(df.home_store, errors="coerce")
    df["name_key"] = [_name_key(f, la) for f, la in zip(df.first_name, df.last_name, strict=True)]
    for c in DEMO_COLS:
        df[c] = df[c].str.strip().replace({"": None})
    return df


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def candidate_pairs(df: pd.DataFrame) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    block_keys = {
        "phone": df.phone_e164,
        "email": df.email,
        "name_store": df.last_name.fillna("").str.lower().str[:3] + "|" + df.home_store.astype(str),
    }
    for key in block_keys.values():
        for _, idx in df.groupby(key, dropna=True).groups.items():
            idx = list(idx)
            if 1 < len(idx) <= 50:  # oversized blocks are junk keys (shared family phone etc.)
                pairs.update((a, b) for i, a in enumerate(idx) for b in idx[i + 1 :])
    return pairs


def link(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Score candidate pairs and return the accepted links with their evidence."""
    rows = []
    for a, b in candidate_pairs(df):
        ra, rb = df.loc[a], df.loc[b]
        sim = fuzz.token_sort_ratio(ra.name_key, rb.name_key)
        phone = ra.phone_e164 is not None and ra.phone_e164 == rb.phone_e164
        email = ra.email is not None and ra.email == rb.email
        same_store = ra.home_store == rb.home_store
        demo_equal = all(
            (pd.isna(ra[c]) and pd.isna(rb[c])) or ra[c] == rb[c] for c in DEMO_COLS
        )
        if ((phone or email) and sim >= 70) or (sim >= threshold and same_store and demo_equal):
            rows.append({"a": a, "b": b, "name_sim": sim, "phone": phone, "email": email, "same_store": same_store})
    return pd.DataFrame(rows, columns=["a", "b", "name_sim", "phone", "email", "same_store"])


def resolve(raw: pd.DataFrame, threshold: int = 88) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (customers golden records, card_map, accepted links)."""
    df = normalise(raw).reset_index(drop=True)
    links = link(df, threshold)
    uf = _UnionFind(df.index)
    for a, b in zip(links.a, links.b, strict=True):
        uf.union(a, b)
    df["cluster"] = [uf.find(i) for i in df.index]

    # deterministic ids: cluster ordered by its earliest card
    order = df.groupby("cluster").card_id.min().sort_values()
    cid = {c: f"C{i:06d}" for i, c in enumerate(order.index, start=1)}
    df["customer_id"] = df.cluster.map(cid)

    # golden record: earliest signup first, then fill gaps from the other cards
    df = df.sort_values(["customer_id", "signup_date"], na_position="last")
    keep = ["first_name", "last_name", "phone_e164", "email", "signup_date", "home_store", *DEMO_COLS]
    golden = df.groupby("customer_id")[keep].first()
    golden["n_cards"] = df.groupby("customer_id").card_id.size()
    golden = golden.reset_index()
    golden["n_cards"] = golden.n_cards.astype(int)

    card_map = df[["card_id", "customer_id"]].reset_index(drop=True)
    log.info(
        "customers: %d cards -> %d customers (%d links, %d multi-card customers); dropped leakage cols %s",
        len(df), len(golden), len(links), int((golden.n_cards > 1).sum()), LEAKAGE_COLS,
    )
    return golden, card_map, links


def dedup_quality(card_map: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Pairwise precision/recall of the clustering against the simulator's ground truth."""
    m = card_map.merge(truth[["card_id", "household_key"]], on="card_id")

    def pairs(col):
        out = set()
        for _, g in m.groupby(col).card_id:
            cards = sorted(g)
            out.update((a, b) for i, a in enumerate(cards) for b in cards[i + 1 :])
        return out

    pred, true = pairs("customer_id"), pairs("household_key")
    tp = len(pred & true)
    return {
        "true_pairs": len(true),
        "predicted_pairs": len(pred),
        "precision": tp / len(pred) if pred else 1.0,
        "recall": tp / len(true) if true else 1.0,
        "unresolved_cards_share": float(np.mean([p not in pred for p in true])) if true else 0.0,
    }
