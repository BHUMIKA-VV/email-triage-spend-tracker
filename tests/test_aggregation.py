"""Aggregation: month-to-date totals, per category / merchant, email counts."""

from __future__ import annotations

from aggregation import summarize

EMAILS = [{"id": f"e{i}", "date": "2026-11-10"} for i in range(1, 6)]
CLASSIFICATIONS = [
    {"email_id": "e1", "category": "Online Orders"},
    {"email_id": "e2", "category": "Online Orders"},
    {"email_id": "e3", "category": "Subscriptions"},
    {"email_id": "e4", "category": "Work"},
    {"email_id": "e5", "category": "Subscriptions"},
]


def t(tid, amount, merchant, category, date, currency="INR", email_category="Online Orders"):
    return {"txn_id": tid, "source_email_ids": [tid], "amount": amount, "currency": currency, "merchant": merchant,
            "category": category, "date": date, "email_date": date, "email_category": email_category}


TXNS = [
    t("t1", 2499.0, "Amazon", "Electronics", "2026-11-02"),
    t("t2", 486.0, "Swiggy", "Food & Dining", "2026-11-04"),
    t("t3", 649.0, "Netflix", "Entertainment", "2026-11-05", email_category="Subscriptions"),
    t("t4", 649.0, "Netflix", "Entertainment", "2026-10-05", email_category="Subscriptions"),  # last month
    t("t5", 10.0, "GitHub", "Software & SaaS", "2026-11-06", currency="USD", email_category="Subscriptions"),
    t("t6", None, "BigBasket", "Groceries", "2026-11-07"),  # unknown amount
    t("t7", 999.0, "Airtel", "Utilities & Bills", "2026-11-12"),  # after as_of
]


def test_month_to_date_totals_per_currency():
    s = summarize(EMAILS, CLASSIFICATIONS, TXNS, as_of="2026-11-10")
    assert s["total_spend"] == {"INR": 2499.0 + 486.0 + 649.0, "USD": 10.0}
    assert set(s["transactions_counted"]) == {"t1", "t2", "t3", "t5"}
    assert set(s["transactions_outside_period"]) == {"t4", "t7"}


def test_spend_by_category_and_merchant():
    s = summarize(EMAILS, CLASSIFICATIONS, TXNS, as_of="2026-11-10")
    assert s["spend_by_category"]["Electronics"] == {"INR": 2499.0}
    assert s["spend_by_category"]["Software & SaaS"] == {"USD": 10.0}
    assert s["spend_by_merchant"]["Netflix"] == {"INR": 649.0}  # October renewal excluded
    assert s["spend_by_email_category"]["Subscriptions"] == {"INR": 649.0, "USD": 10.0}


def test_unpriced_transactions_are_reported_not_counted():
    s = summarize(EMAILS, CLASSIFICATIONS, TXNS, as_of="2026-11-10")
    assert [u["txn_id"] for u in s["transactions_without_amount"]] == ["t6"]
    assert "BigBasket" not in s["spend_by_merchant"]


def test_email_counts_include_every_category():
    s = summarize(EMAILS, CLASSIFICATIONS, [], as_of="2026-11-10")
    assert s["email_counts"] == {"Work": 1, "Online Orders": 2, "Subscriptions": 2, "Miscellaneous": 0}
    assert s["total_spend"] == {}


def test_full_month_query():
    s = summarize(EMAILS, CLASSIFICATIONS, TXNS, month="2026-10")
    assert s["total_spend"] == {"INR": 649.0}


def test_as_of_defaults_to_newest_email():
    s = summarize([{"id": "a", "date": "2026-11-03"}, {"id": "b", "date": "2026-11-16"}], [], [])
    assert s["as_of"] == "2026-11-16"
