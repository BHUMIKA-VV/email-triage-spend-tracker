"""Spend aggregation - pure functions, no LLM, fully unit-testable.

Amounts are summed per currency and never converted: summing INR and USD with a
guessed exchange rate would itself be an "invented amount".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Optional

from dedup import effective_date
from schemas import EMAIL_CATEGORIES
from utils import parse_iso_date


def _add(bucket: dict, key: str, currency: str, amount: float) -> None:
    bucket[key][currency] = round(bucket[key].get(currency, 0.0) + amount, 2)


def _plain(d: dict) -> dict:
    return {k: dict(sorted(v.items())) for k, v in sorted(d.items())}


def resolve_as_of(as_of: Optional[str], emails: list[dict]) -> date:
    """Reference date for "this month": explicit value, else the newest email's date."""
    parsed = parse_iso_date(as_of)
    if parsed:
        return parsed
    dates = [d for d in (parse_iso_date(e.get("date")) for e in emails) if d]
    return max(dates) if dates else date.today()


def in_month(txn: dict, year: int, month: int, up_to: Optional[date] = None) -> bool:
    d = effective_date(txn)
    if d is None or (d.year, d.month) != (year, month):
        return False
    return up_to is None or d <= up_to


def summarize(
    emails: list[dict],
    classifications: list[dict],
    transactions: list[dict],
    as_of: Optional[str] = None,
    month: Optional[str] = None,
) -> dict:
    """Build the spend summary.

    `month` ("YYYY-MM") reports a whole calendar month; otherwise the window is
    month-to-date relative to `as_of`.
    """
    ref = resolve_as_of(as_of, emails)
    if month:
        year, mon = (int(x) for x in month.split("-"))
        up_to: Optional[date] = None
        period = f"{year:04d}-{mon:02d} (full month)"
    else:
        year, mon, up_to = ref.year, ref.month, ref
        period = f"{year:04d}-{mon:02d}-01 to {ref.isoformat()} (month-to-date)"

    total: dict[str, float] = defaultdict(float)
    by_category: dict[str, dict] = defaultdict(dict)
    by_email_category: dict[str, dict] = defaultdict(dict)
    by_merchant: dict[str, dict] = defaultdict(dict)
    counted: list[str] = []
    unpriced: list[dict] = []
    outside_period: list[str] = []

    for txn in transactions:
        if txn.get("amount") is None:
            unpriced.append(
                {"txn_id": txn["txn_id"], "merchant": txn.get("merchant"), "emails": txn["source_email_ids"]}
            )
            continue
        if not in_month(txn, year, mon, up_to):
            outside_period.append(txn["txn_id"])
            continue
        currency = txn.get("currency") or "UNKNOWN"
        amount = float(txn["amount"])
        total[currency] = round(total[currency] + amount, 2)
        _add(by_category, txn.get("category") or "Uncategorised", currency, amount)
        _add(by_email_category, txn.get("email_category") or "Unknown", currency, amount)
        _add(by_merchant, txn.get("merchant") or "Unknown merchant", currency, amount)
        counted.append(txn["txn_id"])

    counts = Counter(c["category"] for c in classifications)
    email_counts = {cat: counts.get(cat, 0) for cat in EMAIL_CATEGORIES}

    return {
        "as_of": ref.isoformat(),
        "period": period,
        "total_spend": dict(sorted(total.items())),
        "spend_by_category": _plain(by_category),
        "spend_by_email_category": _plain(by_email_category),
        "spend_by_merchant": _plain(by_merchant),
        "email_counts": email_counts,
        "emails_processed": len(emails),
        "transactions_counted": counted,
        "transactions_without_amount": unpriced,
        "transactions_outside_period": outside_period,
    }
