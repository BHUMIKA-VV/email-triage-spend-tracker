"""Conditional routing: which emails reach the extractor, fan-out, and the review loop."""

from __future__ import annotations

from conftest import ScriptedClient, make_email
from langgraph.graph import END
from langgraph.types import Send

from nodes import dispatch_emails, needs_extraction, route_after_classify, route_review
from schemas import Classification, Email


def _cls(category, may_txn, confidence=0.95):
    return {"category": category, "confidence": confidence, "reason": "r", "may_contain_transaction": may_txn}


def test_work_invoice_is_routed_to_extraction_even_if_llm_says_no(email_by_id):
    # The LLM classifies the invoice as Work and (wrongly) says no transaction:
    # the deterministic "invoice" signal still sends it to the extractor.
    assert needs_extraction(email_by_id("e008"), _cls("Work", False)) is True


def test_ordinary_work_email_skips_extraction(email_by_id):
    assert needs_extraction(email_by_id("e003"), _cls("Work", False)) is False


def test_newsletter_and_promo_skip_extraction(email_by_id):
    assert needs_extraction(email_by_id("e013"), _cls("Subscriptions", False)) is False
    assert needs_extraction(email_by_id("e010"), _cls("Miscellaneous", False)) is False


def test_online_orders_always_checked(email_by_id):
    assert needs_extraction(email_by_id("e011"), _cls("Online Orders", False)) is True


def test_route_after_classify_returns_graph_targets(email_by_id):
    state = {"email": email_by_id("e003").as_state(), "classification": _cls("Work", False)}
    assert route_after_classify(state) == END
    state = {"email": email_by_id("e002").as_state(), "classification": _cls("Subscriptions", True)}
    assert route_after_classify(state) == "extract_transaction"


def test_dispatch_fans_out_one_send_per_email(emails):
    sends = dispatch_emails({"emails": emails[:3]})
    assert all(isinstance(s, Send) and s.node == "process_email" for s in sends)
    assert [s.arg["email"]["id"] for s in sends] == ["e001", "e002", "e003"]
    assert dispatch_emails({"emails": []}) == "deduplicate"


def test_review_loop_routing():
    item = {"review_id": "review-e1"}
    assert route_review({"review_queue": [item], "review_decisions": []}) == "human_review"
    assert route_review({"review_queue": [item], "review_decisions": [{"review_id": "review-e1"}]}) == "aggregate"
    assert route_review({"review_queue": []}) == "aggregate"


def test_graph_only_calls_extractor_for_transaction_emails(run_graph, emails):
    """End to end: the extractor (LLM) is never called for conversations/newsletters/promos."""
    client = ScriptedClient()
    by_id = {e["id"]: e for e in emails}
    picked = [by_id[i] for i in ("e003", "e008", "e010", "e013", "e001", "e027")]
    run_graph(picked, client=client)

    assert sorted(client.classify_calls) == sorted(e["id"] for e in picked)
    assert sorted(client.extract_calls) == ["e001", "e008"]  # order + work invoice only


def test_classifier_flag_alone_triggers_extraction(run_graph):
    client = ScriptedClient(classifications={
        "x1": Classification(category="Miscellaneous", confidence=0.9, reason="r", may_contain_transaction=True)
    })
    run_graph([make_email(body="Thanks for visiting our store today.")], client=client)
    assert client.extract_calls == ["x1"]


def test_needs_extraction_uses_email_model():
    email = Email.model_validate(make_email(body="Invoice attached for consulting"))
    assert needs_extraction(email, _cls("Work", False))
