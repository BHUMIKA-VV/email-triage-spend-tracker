"""Human-in-the-loop: review triggers, interrupt(), and resume via checkpointer + thread id."""

from __future__ import annotations

from conftest import ScriptedClient, make_email
from langgraph.types import Command

from config import Settings
from graph import build_graph, sqlite_checkpointer
from review import review_reasons
from schemas import Classification

BIG_ORDER = make_email(
    id="big", sender="tickets@makemytrip.com", subject="Booking confirmed",
    body="Booking ID NF7Y2K93. Total fare paid: Rs. 18,750 with your HDFC Credit Card.",
)
SMALL_ORDER = make_email(
    id="small", sender="noreply@swiggy.in", subject="Your Swiggy order was delivered",
    body="Order #SW-1 delivered. Grand total paid: Rs. 486 via UPI.",
)
VAGUE = make_email(id="vague", sender="someone@example.com", subject="hi", body="ok see you then")


# ---------------------------------------------------------------- trigger rules
def test_review_triggers():
    s = Settings(review_confidence_threshold=0.7, spend_review_threshold_inr=10_000)
    confident = {"confidence": 0.9}
    assert review_reasons(confident, {"amount": 500, "currency": "INR"}, s) == []
    assert review_reasons({"confidence": 0.69}, None, s)[0].startswith("classification confidence")
    assert "exceeds review threshold" in review_reasons(confident, {"amount": 18_750, "currency": "INR"}, s)[0]
    # $200 ~ Rs.17,600 -> over the INR threshold after conversion
    assert review_reasons(confident, {"amount": 200, "currency": "USD"}, s)
    assert review_reasons(confident, {"amount": 10, "currency": "XYZ"}, s)  # unknown currency
    assert review_reasons(confident, {"amount": None, "currency": None}, s) == []


def test_threshold_is_configurable():
    low = Settings(spend_review_threshold_inr=100)
    assert review_reasons({"confidence": 0.9}, {"amount": 486, "currency": "INR"}, low)


# ---------------------------------------------------------------- interrupt / resume
def _confident_client():
    return ScriptedClient(classifications={
        "big": Classification(category="Online Orders", confidence=0.95, reason="flight booking",
                              may_contain_transaction=True),
        "small": Classification(category="Online Orders", confidence=0.95, reason="food order",
                                may_contain_transaction=True),
        "vague": Classification(category="Miscellaneous", confidence=0.4, reason="too short to tell",
                                may_contain_transaction=False),
    })


def test_interrupt_then_approve_resumes_to_completion(run_graph):
    graph, config = run_graph([BIG_ORDER, SMALL_ORDER], client=_confident_client())

    snap = graph.get_state(config)
    assert snap.next == ("human_review",)
    [intr] = snap.interrupts
    assert intr.value["email_id"] == "big"
    assert intr.value["transaction"]["amount"] == 18750.0
    assert "summary" not in snap.values  # aggregation has NOT run yet

    graph.invoke(Command(resume={"action": "approve"}), config)
    final = graph.get_state(config)
    assert not final.interrupts and final.next == ()
    assert final.values["summary"]["total_spend"] == {"INR": 18750.0 + 486.0}
    assert final.values["review_decisions"][0]["action"] == "approve"


def test_reject_excludes_transaction_from_spend(run_graph):
    graph, config = run_graph([BIG_ORDER, SMALL_ORDER], client=_confident_client())
    graph.invoke(Command(resume={"action": "reject", "note": "refunded"}), config)
    summary = graph.get_state(config).values["summary"]
    assert summary["total_spend"] == {"INR": 486.0}
    assert "MakeMyTrip" not in summary["spend_by_merchant"]


def test_correct_patches_amount_and_category(run_graph):
    graph, config = run_graph([BIG_ORDER, SMALL_ORDER], client=_confident_client())
    graph.invoke(Command(resume={"action": "correct", "amount": 17500, "category": "Work",
                                 "spend_category": "Travel"}), config)
    values = graph.get_state(config).values
    big_txn = next(t for t in values["final_transactions"] if t["email_id"] == "big")
    assert big_txn["amount"] == 17500 and big_txn["email_category"] == "Work"
    big_cls = next(c for c in values["final_classifications"] if c["email_id"] == "big")
    assert big_cls["category"] == "Work" and big_cls["original_category"] == "Online Orders"
    assert values["summary"]["email_counts"]["Work"] == 1
    # Raw model output is preserved for audit.
    assert next(c for c in values["classifications"] if c["email_id"] == "big")["category"] == "Online Orders"


def test_multiple_reviews_are_answered_one_interrupt_at_a_time(run_graph):
    graph, config = run_graph([BIG_ORDER, SMALL_ORDER, VAGUE], client=_confident_client())
    seen = []
    while (snap := graph.get_state(config)).interrupts:
        item = snap.interrupts[0].value
        seen.append((item["email_id"], item["pending_count"]))
        graph.invoke(Command(resume="approve"), config)  # bare string is accepted too
    assert seen == [("big", 2), ("vague", 1)]
    assert len(graph.get_state(config).values["review_decisions"]) == 2


def test_invalid_resume_value_re_prompts_instead_of_crashing(run_graph):
    graph, config = run_graph([BIG_ORDER], client=_confident_client())
    graph.invoke(Command(resume={"action": "maybe"}), config)
    snap = graph.get_state(config)
    assert snap.interrupts and "invalid decision" in snap.interrupts[0].value["error"]
    graph.invoke(Command(resume={"action": "approve"}), config)
    assert graph.get_state(config).values["summary"]["total_spend"] == {"INR": 18750.0}


def test_correction_can_add_a_missing_transaction(run_graph):
    # Reviewer knows the vague email was actually a Rs.2,340 payment.
    graph, config = run_graph([VAGUE], client=_confident_client())
    graph.invoke(Command(resume={"action": "correct", "amount": 2340, "currency": "INR",
                                 "merchant": "Toit"}), config)
    summary = graph.get_state(config).values["summary"]
    assert summary["spend_by_merchant"] == {"Toit": {"INR": 2340.0}}


def test_resume_from_new_process_with_sqlite_checkpointer(tmp_path, settings):
    """Simulates `python main.py` exiting at the interrupt and `python main.py resume`
    running later: a brand-new graph object + new DB connection, same thread id."""
    db = str(tmp_path / "ckpt.sqlite")
    config = {"configurable": {"thread_id": "thread-42"}}

    first = build_graph(client=_confident_client(), settings=settings, checkpointer=sqlite_checkpointer(db))
    first.invoke({"raw_emails": [BIG_ORDER, SMALL_ORDER]}, config)
    assert first.get_state(config).interrupts
    del first

    client2 = _confident_client()
    second = build_graph(client=client2, settings=settings, checkpointer=sqlite_checkpointer(db))
    assert second.get_state(config).interrupts[0].value["email_id"] == "big"
    second.invoke(Command(resume={"action": "approve"}), config)

    assert second.get_state(config).values["summary"]["total_spend"] == {"INR": 19236.0}
    assert client2.classify_calls == [] and client2.extract_calls == []  # no LLM work was redone
