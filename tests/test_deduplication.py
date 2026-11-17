"""De-duplication of transactions that describe the same purchase."""

from __future__ import annotations

from dedup import deduplicate, is_duplicate


def txn(email_id, merchant="Amazon", amount=2499.0, currency="INR", order_id=None, email_date="2026-11-03",
        date=None, payment_method=None):
    return {
        "txn_id": f"txn-{email_id}", "email_id": email_id, "source_email_ids": [email_id],
        "amount": amount, "currency": currency, "merchant": merchant, "date": date,
        "category": "Electronics", "payment_method": payment_method, "order_id": order_id,
        "description": None, "email_date": email_date, "email_category": "Online Orders", "notes": [],
    }


def test_order_confirmation_and_shipping_email_merge_on_order_id():
    confirm = txn("e004", order_id="#402-118", email_date="2026-11-02", date="2026-11-02", payment_method="UPI")
    shipped = txn("e001", order_id="402-118", email_date="2026-11-03")
    [merged] = deduplicate([shipped, confirm])
    assert merged["source_email_ids"] == ["e001", "e004"]
    assert merged["amount"] == 2499.0
    assert merged["payment_method"] == "UPI" and merged["date"] == "2026-11-02"  # nulls filled from duplicate


def test_same_merchant_amount_within_window_without_ids_merge():
    a = txn("a", order_id=None, email_date="2026-11-02")
    b = txn("b", merchant="Amazon.in", order_id=None, email_date="2026-11-04")
    assert len(deduplicate([a, b], window_days=3)) == 1


def test_different_order_ids_are_never_merged():
    # Two genuine Rs.349 Zomato orders on consecutive days.
    a = txn("e006", merchant="Zomato", amount=349, order_id="ZO-55102", email_date="2026-11-07")
    b = txn("e007", merchant="Zomato", amount=349, order_id="ZO-55187", email_date="2026-11-08")
    assert len(deduplicate([a, b])) == 2


def test_monthly_renewals_are_not_duplicates():
    oct_ = txn("e024", merchant="Netflix", amount=649, email_date="2026-10-05")
    nov = txn("e002", merchant="Netflix", amount=649, email_date="2026-11-05")
    assert len(deduplicate([oct_, nov], window_days=3)) == 2


def test_different_amounts_or_currencies_are_not_merged():
    assert not is_duplicate(txn("a", amount=100), txn("b", amount=101))
    assert not is_duplicate(txn("a", currency="INR"), txn("b", currency="USD"))
    assert not is_duplicate(txn("a", merchant="Amazon"), txn("b", merchant="Flipkart"))


def test_unpriced_delivery_notice_folds_into_priced_order():
    placed = txn("e012", merchant="Flipkart", amount=1299, email_date="2026-11-09")
    delivered = txn("e011", merchant="Flipkart", amount=None, currency=None, email_date="2026-11-12")
    [merged] = deduplicate([delivered, placed], window_days=3)
    assert merged["amount"] == 1299 and merged["source_email_ids"] == ["e011", "e012"]


def test_dedup_is_order_independent():
    items = [txn("a", order_id="X1"), txn("b", order_id="X1"), txn("c", merchant="Swiggy", amount=486)]
    forward = deduplicate(items)
    backward = deduplicate(list(reversed(items)))
    assert sorted(t["txn_id"] for t in forward) == sorted(t["txn_id"] for t in backward)


def test_full_run_dedups_sample_inbox(run_graph, emails):
    graph, config = run_graph(emails, settings_override={"review_confidence_threshold": 0.0,
                                                         "spend_review_threshold_inr": 10**9})
    state = graph.get_state(config).values
    groups = {tuple(t["source_email_ids"]) for t in state["deduped_transactions"]}
    assert ("e001", "e004") in groups   # Amazon confirmation + shipping
    assert ("e011", "e012") in groups   # Flipkart placed + delivered
    assert ("e022", "e023") in groups   # Myntra confirmed + shipped
    assert ("e006",) in groups and ("e007",) in groups  # distinct Zomato orders
    amazon = state["summary"]["spend_by_merchant"]["Amazon"]
    assert amazon == {"INR": 2499.0}    # counted once, not twice
