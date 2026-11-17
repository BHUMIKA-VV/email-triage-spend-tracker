"""Extraction: correct amounts, no transaction for ordinary emails, never invent amounts."""

from __future__ import annotations

from conftest import FailingClient, ScriptedClient, make_email

from nodes import make_nodes, postprocess_extraction
from rules import rule_extract
from schemas import Email, ExtractedTransaction
from utils import amount_is_grounded, numbers_in_text


def test_amount_currency_merchant_and_payment_method(email_by_id):
    t = rule_extract(email_by_id("e001"))
    assert t.is_transaction
    assert (t.amount, t.currency, t.merchant, t.payment_method) == (2499.0, "INR", "Amazon", "UPI")
    assert t.order_id == "402-118"


def test_grand_total_is_preferred_over_line_items(email_by_id):
    # Item total 520, delivery 25, discount 59 -> paid 486.
    assert rule_extract(email_by_id("e005")).amount == 486.0


def test_foreign_currency(email_by_id):
    t = rule_extract(email_by_id("e016"))
    assert (t.amount, t.currency) == (10.0, "USD")


def test_ordinary_email_has_no_transaction(email_by_id):
    assert rule_extract(email_by_id("e003")).is_transaction is False


def test_future_charge_is_not_a_transaction(email_by_id):
    # "you will be charged Rs.299 after the trial" - nothing has been paid yet.
    assert rule_extract(email_by_id("e025")).is_transaction is False


def test_missing_amount_stays_null(email_by_id):
    # BigBasket: "final amount will be charged on delivery" - no number to extract.
    t = rule_extract(email_by_id("e021"))
    assert t.is_transaction is True
    assert t.amount is None and t.currency is None


def test_vague_amount_in_words_is_not_converted(email_by_id):
    # "around two and a half thousand" must never become 2500.
    t = rule_extract(email_by_id("e019"))
    assert t.amount is None


def test_delivery_date_is_not_used_as_transaction_date(email_by_id):
    assert rule_extract(email_by_id("e001")).date is None          # only "Expected delivery" date
    assert rule_extract(email_by_id("e004")).date == "2026-11-02"  # "placed on 2026-11-02"


# ---------------------------------------------------------------- anti-hallucination guard
def test_hallucinated_amount_is_discarded():
    email = Email.model_validate(make_email(subject="Order shipped", body="Your order #A123 has shipped."))
    llm_output = ExtractedTransaction(is_transaction=True, amount=1999.0, currency="INR", merchant="Shop")
    txn = postprocess_extraction(email, llm_output, "Online Orders", "llm")
    assert txn["amount"] is None and txn["currency"] is None
    assert any("not found in email text" in n for n in txn["notes"])


def test_grounded_amount_is_kept_and_currency_normalised():
    email = Email.model_validate(make_email(body="Total paid: ₹1,23,456.50 by card."))
    llm_output = ExtractedTransaction(is_transaction=True, amount=123456.5, currency="₹")
    txn = postprocess_extraction(email, llm_output, "Online Orders", "llm")
    assert txn["amount"] == 123456.5 and txn["currency"] == "INR"


def test_non_transaction_returns_nothing():
    email = Email.model_validate(make_email(body="Flat 50% off, prices from Rs. 499"))
    assert postprocess_extraction(email, ExtractedTransaction(is_transaction=False), "Miscellaneous", "llm") is None


def test_header_date_is_not_invented_when_llm_returns_garbage_date():
    email = Email.model_validate(make_email(body="Paid Rs. 500"))
    txn = postprocess_extraction(
        email, ExtractedTransaction(is_transaction=True, amount=500, date="last Tuesday"), "Online Orders", "llm"
    )
    assert txn["date"] is None
    assert txn["email_date"] == "2026-11-10"  # kept separately, never copied into `date`


def test_number_parsing_handles_indian_grouping():
    assert {2499.0, 123456.5} <= numbers_in_text("Rs. 2,499 and ₹1,23,456.50")
    assert amount_is_grounded(649, "We charged Rs. 649")
    assert not amount_is_grounded(650, "We charged Rs. 649")


# ---------------------------------------------------------------- graph-level
def test_extract_node_failure_falls_back_to_rules(settings, email_by_id):
    nodes = make_nodes(FailingClient(), settings)
    email = email_by_id("e001")
    out = nodes.extract_transaction({
        "email": email.as_state(),
        "classification": {"email_id": "e001", "category": "Online Orders", "confidence": 0.9,
                           "reason": "order", "may_contain_transaction": True},
    })
    [txn] = out["transactions"]
    assert txn["amount"] == 2499.0 and txn["extraction_method"] == "fallback_rules"
    assert out["errors"][0]["stage"] == "extract"


def test_llm_saying_no_transaction_produces_no_row(run_graph):
    client = ScriptedClient(extractions={"x1": ExtractedTransaction(is_transaction=False)})
    graph, config = run_graph(
        [make_email(subject="Your order has shipped", body="Order #ZX-9981 total Rs. 300 paid")], client=client
    )
    assert client.extract_calls == ["x1"]
    assert graph.get_state(config).values["transactions"] == []
