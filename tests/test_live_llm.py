"""Optional checks against a REAL LLM. Skipped unless an API key is configured.

    LLM_PROVIDER=gemini GOOGLE_API_KEY=...    pytest -m live     (free tier)
    LLM_PROVIDER=groq   GROQ_API_KEY=...      pytest -m live     (free tier)
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=... pytest -m live

Calls are wrapped in the same retry helper the graph uses, with a longer backoff,
so free-tier rate limits (HTTP 429) are waited out instead of failing the test.
"""

from __future__ import annotations

import os

import pytest

from config import load_settings
from llm import _KEY_ENV, build_llm_client, resolve_provider
from nodes import postprocess_extraction
from utils import call_with_retry

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not any(os.getenv(env) for _, env in _KEY_ENV), reason="no LLM API key configured"),
]

LIVE_RETRY = dict(attempts=4, base_delay=10.0)  # 10s, 20s, 40s


@pytest.fixture(scope="module")
def client():
    settings = load_settings()
    if resolve_provider(settings) == "rules":
        pytest.skip("LLM_PROVIDER resolves to rules")
    return build_llm_client(settings)


def _classify(client, email):
    return call_with_retry(lambda: client.classify(email), label="live classify", **LIVE_RETRY)


def _extract(client, email, cls):
    return call_with_retry(lambda: client.extract(email, cls), label="live extract", **LIVE_RETRY)


@pytest.mark.parametrize(
    "email_id, expected",
    [("e003", "Work"), ("e001", "Online Orders"), ("e002", "Subscriptions"), ("e010", "Miscellaneous"),
     ("e008", "Work")],
)
def test_live_classification(client, email_by_id, email_id, expected):
    result = _classify(client, email_by_id(email_id))
    assert result.category == expected, result.reason
    assert 0.0 <= result.confidence <= 1.0


def test_live_extraction_never_invents_amount(client, email_by_id):
    email = email_by_id("e021")  # BigBasket: amount charged later
    cls = _classify(client, email)
    txn = postprocess_extraction(email, _extract(client, email, cls), cls.category, "llm")
    assert txn is None or txn["amount"] is None


def test_live_extraction_amount(client, email_by_id):
    email = email_by_id("e005")
    cls = _classify(client, email)
    txn = postprocess_extraction(email, _extract(client, email, cls), cls.category, "llm")
    assert txn["amount"] == 486.0 and txn["currency"] == "INR"
