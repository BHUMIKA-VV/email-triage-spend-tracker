"""Classification: categories, ambiguous edge cases, structured output, retry and fallback."""

from __future__ import annotations

import pytest
from conftest import FailingClient, ScriptedClient, make_email

from config import Settings
from llm import LangChainLLMClient
from nodes import FALLBACK_CONFIDENCE_CAP, make_nodes
from prompts import CLASSIFY_SYSTEM
from rules import rule_classify
from schemas import Classification


# ---------------------------------------------------------------- all four categories
@pytest.mark.parametrize(
    "email_id, expected",
    [
        ("e003", "Work"),            # sprint review
        ("e001", "Online Orders"),   # Amazon shipped
        ("e002", "Subscriptions"),   # Netflix renewal
        ("e017", "Miscellaneous"),   # bank security alert
    ],
)
def test_each_category_is_recognised(email_by_id, email_id, expected):
    result = rule_classify(email_by_id(email_id))
    assert result.category == expected
    assert 0.0 <= result.confidence <= 1.0
    assert result.reason and "\n" not in result.reason


# ---------------------------------------------------------------- ambiguous emails
def test_work_invoice_is_work_but_flags_a_transaction(email_by_id):
    result = rule_classify(email_by_id("e008"))
    assert result.category == "Work"
    assert result.may_contain_transaction is True


def test_subscription_renewal_that_charges_card_is_subscription(email_by_id):
    assert rule_classify(email_by_id("e002")).category == "Subscriptions"


def test_promo_from_store_user_ordered_from_is_miscellaneous(email_by_id):
    result = rule_classify(email_by_id("e010"))  # Flipkart sale; user has a Flipkart order
    assert result.category == "Miscellaneous"
    assert result.may_contain_transaction is False


def test_ambiguous_email_gets_low_confidence(email_by_id):
    # "Payment successful" from a payment gateway: no clear category.
    result = rule_classify(email_by_id("e018"))
    assert result.confidence < 0.7


def test_confidence_bounds_are_enforced_by_schema():
    with pytest.raises(ValueError):
        Classification(category="Work", confidence=1.5, reason="x", may_contain_transaction=False)
    with pytest.raises(ValueError):
        Classification(category="Spam", confidence=0.9, reason="x", may_contain_transaction=False)


# ---------------------------------------------------------------- structured output wiring
class _StubStructured:
    def __init__(self, schema, sink):
        self.schema, self.sink = schema, sink

    def invoke(self, messages):
        self.sink.append((self.schema, messages))
        if self.schema is Classification:
            # Providers sometimes hand back a dict; the client must validate it.
            return {"category": "Work", "confidence": 0.93, "reason": "Colleague asks for a deck.",
                    "may_contain_transaction": False}
        return self.schema(is_transaction=False)


class _StubChatModel:
    def __init__(self):
        self.calls = []

    def with_structured_output(self, schema):
        return _StubStructured(schema, self.calls)


def test_langchain_client_uses_structured_output_schema(email_by_id):
    chat = _StubChatModel()
    client = LangChainLLMClient(chat)
    result = client.classify(email_by_id("e003"))

    assert isinstance(result, Classification) and result.category == "Work"
    schema, messages = chat.calls[0]
    assert schema is Classification
    assert messages[0].content == CLASSIFY_SYSTEM
    assert "Sprint review moved to Friday" in messages[1].content
    assert "<email>" in messages[1].content  # email is delimited as untrusted data


# ---------------------------------------------------------------- retry + fallback
def test_llm_failure_falls_back_after_retries(settings, email_by_id):
    client = FailingClient()
    nodes = make_nodes(client, settings)
    out = nodes.classify({"email": email_by_id("e003").as_state()})

    assert client.classify_attempts == settings.llm_max_retries  # retried, then gave up
    cls = out["classification"]
    assert cls["method"] == "fallback_rules"
    assert cls["category"] == "Work"  # rules still produce a useful category
    assert cls["confidence"] <= FALLBACK_CONFIDENCE_CAP < settings.review_confidence_threshold
    assert out["errors"][0]["stage"] == "classify"


def test_transient_failure_is_retried_and_succeeds(settings, email_by_id):
    client = FailingClient(fail_times=1)
    nodes = make_nodes(client, settings)
    out = nodes.classify({"email": email_by_id("e003").as_state()})

    assert client.classify_attempts == 2
    assert out["classification"]["method"] == "failing"
    assert out["errors"] == []


def test_fallback_survives_even_if_rules_crash(settings, monkeypatch):
    import nodes as nodes_module

    monkeypatch.setattr(nodes_module, "rule_classify", lambda e: (_ for _ in ()).throw(RuntimeError("bug")))
    out = make_nodes(FailingClient(), settings).classify({"email": make_email(body="hello")})
    assert out["classification"]["category"] == "Miscellaneous"
    assert out["classification"]["confidence"] == 0.0


def test_whole_run_survives_a_dead_llm(run_graph, emails):
    """Every LLM call fails: the run still completes, every email is classified,
    and every email is sent to human review because fallback confidence < 0.7."""
    graph, config = run_graph(emails[:5], client=FailingClient(), settings_override={"llm_max_retries": 2})
    state = graph.get_state(config)
    assert len(state.values["classifications"]) == 5
    assert all(c["method"] == "fallback_rules" for c in state.values["classifications"])
    assert len(state.values["review_queue"]) == 5
    assert state.interrupts  # paused for a human, not crashed


def test_scripted_llm_output_is_used_verbatim(run_graph):
    scripted = ScriptedClient(classifications={
        "x1": Classification(category="Subscriptions", confidence=0.95, reason="Weekly digest.",
                             may_contain_transaction=False)
    })
    graph, config = run_graph([make_email(subject="Weekly digest")], client=scripted)
    final = graph.get_state(config).values["final_classifications"][0]
    assert (final["category"], final["confidence"], final["reason"]) == ("Subscriptions", 0.95, "Weekly digest.")


def test_settings_threshold_is_configurable():
    assert Settings(review_confidence_threshold=0.8).review_confidence_threshold == 0.8


# ---------------------------------------------------------------- permanent vs transient errors
class _HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _WrapperError(Exception):
    """Mimics LangChain wrapping an SDK error: the status lives on __cause__."""


def _raise_wrapped(status):
    try:
        raise _HTTPError(status)
    except _HTTPError as inner:
        raise _WrapperError("model call failed") from inner


@pytest.mark.parametrize("status, retryable", [(404, False), (401, False), (400, False), (429, True), (503, True)])
def test_permanent_errors_are_not_retried(status, retryable):
    from utils import call_with_retry, is_retryable

    calls = []

    def fail():
        calls.append(1)
        _raise_wrapped(status)

    with pytest.raises(_WrapperError) as info:
        call_with_retry(fail, attempts=3, base_delay=0, sleep=lambda s: None)
    assert is_retryable(info.value) is retryable
    assert len(calls) == (3 if retryable else 1)


def test_network_and_validation_errors_are_retryable():
    from utils import is_retryable

    assert is_retryable(ConnectionError("getaddrinfo failed"))
    assert is_retryable(TimeoutError())
    with pytest.raises(ValueError) as info:
        Classification(category="Spam", confidence=0.9, reason="x", may_contain_transaction=False)
    assert is_retryable(info.value)


def test_daily_quota_exhaustion_is_permanent_but_per_minute_limit_is_not():
    from utils import is_retryable

    daily = _HTTPError(429)
    daily.args = ("429 RESOURCE_EXHAUSTED: Generate content requests per day limit exceeded on Free Tier",)
    per_minute = _HTTPError(429)
    per_minute.args = ("429 RESOURCE_EXHAUSTED: requests per minute limit exceeded",)
    assert is_retryable(daily) is False
    assert is_retryable(per_minute) is True
