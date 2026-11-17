"""Transaction de-duplication.

An order confirmation, a payment receipt and a shipping notice often describe
the SAME purchase. Two transactions are considered the same purchase when:

1. Both carry an order/invoice id -> duplicate iff the normalised ids are equal
   (and merchants don't contradict). Different ids are never merged, even if the
   amount and merchant match (two genuine Rs.349 Swiggy orders stay separate).
2. Otherwise, same normalised merchant AND same currency AND same amount AND
   effective dates within `window_days`.
3. A same-merchant record with NO amount within the window (e.g. "Your order
   has been delivered") is folded into the priced record instead of being
   reported as a separate, unpriced transaction.

Merging keeps the record with the most information as primary, fills its null
fields from the duplicate, and records all `source_email_ids`.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from utils import normalize_merchant, normalize_order_id, parse_iso_date

_INFO_FIELDS = ("amount", "currency", "merchant", "date", "category", "payment_method", "order_id", "description")


def effective_date(txn: dict) -> Optional[date]:
    """Transaction date if the email stated one, else the email's own date."""
    return parse_iso_date(txn.get("date")) or parse_iso_date(txn.get("email_date"))


def _within(a: dict, b: dict, window_days: int) -> bool:
    da, db = effective_date(a), effective_date(b)
    if da is None or db is None:
        return False
    return abs((da - db).days) <= window_days


def is_duplicate(a: dict, b: dict, window_days: int = 3) -> bool:
    ma, mb = normalize_merchant(a.get("merchant")), normalize_merchant(b.get("merchant"))
    merchants_match = bool(ma) and ma == mb
    oa, ob = normalize_order_id(a.get("order_id")), normalize_order_id(b.get("order_id"))

    if oa and ob:
        # Same id at the same (or an unknown) merchant is the same purchase.
        return oa == ob and (merchants_match or not ma or not mb)

    if not merchants_match:
        return False

    amt_a, amt_b = a.get("amount"), b.get("amount")
    if amt_a is not None and amt_b is not None:
        return (
            abs(float(amt_a) - float(amt_b)) < 0.005
            and (a.get("currency") or "") == (b.get("currency") or "")
            and _within(a, b, window_days)
        )

    # One side has no amount: a follow-up notice about the same purchase.
    if (amt_a is None) != (amt_b is None):
        return _within(a, b, window_days)
    return False


def _richness(txn: dict) -> tuple[int, int]:
    filled = sum(1 for f in _INFO_FIELDS if txn.get(f) not in (None, ""))
    # Prefer records with an amount, then the most complete one.
    return (1 if txn.get("amount") is not None else 0, filled)


def _merge(primary: dict, secondary: dict) -> dict:
    if _richness(secondary) > _richness(primary):
        primary, secondary = secondary, primary
    merged = dict(primary)
    for field in _INFO_FIELDS:
        if merged.get(field) in (None, "") and secondary.get(field) not in (None, ""):
            merged[field] = secondary[field]
    merged["source_email_ids"] = sorted(
        set(primary.get("source_email_ids", [primary["email_id"]]))
        | set(secondary.get("source_email_ids", [secondary["email_id"]]))
    )
    notes = list(primary.get("notes", [])) + [n for n in secondary.get("notes", []) if n not in primary.get("notes", [])]
    if (
        primary.get("amount") is not None
        and secondary.get("amount") is not None
        and abs(primary["amount"] - secondary["amount"]) >= 0.005
    ):
        notes.append(
            f"amount conflict with {secondary['email_id']} ({secondary['amount']}); kept {primary['amount']}"
        )
    merged["notes"] = notes
    return merged


def deduplicate(transactions: list[dict], window_days: int = 3) -> list[dict]:
    """Collapse transactions that describe the same purchase. Input order does not matter."""
    ordered = sorted(
        transactions,
        key=lambda t: (effective_date(t) or date.max, t.get("email_id", "")),
    )
    clusters: list[dict] = []
    for txn in ordered:
        txn = {**txn, "source_email_ids": txn.get("source_email_ids") or [txn["email_id"]]}
        for i, cluster in enumerate(clusters):
            if is_duplicate(cluster, txn, window_days):
                clusters[i] = _merge(cluster, txn)
                break
        else:
            clusters.append(txn)
    for c in clusters:
        c["txn_id"] = "txn-" + "-".join(c["source_email_ids"])
    return clusters
