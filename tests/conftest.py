"""Shared fixtures. Nothing here needs an API key: LLMs are replaced by scripted fakes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from config import Settings
from graph import build_graph
from rules import rule_classify, rule_extract
from schemas import Classification, Email, ExtractedTransaction

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def emails() -> list[dict]:
    return json.loads((ROOT / "emails.json").read_text(encoding="utf-8"))


@pytest.fixture
def email_by_id(emails) -> Callable[[str], Email]:
    index = {e["id"]: e for e in emails}
    return lambda eid: Email.model_validate(index[eid])


@pytest.fixture
def settings() -> Settings:
    # No sleeping between retries in tests.
    return Settings(llm_provider="rules", llm_max_retries=3, llm_retry_base_delay=0.0)


def make_email(id="x1", sender="someone@example.com", subject="", body="", date="2026-11-10") -> dict:
    return {"id": id, "from": sender, "subject": subject, "date": date, "body": body}


class ScriptedClient:
    """Fake LLM: returns canned structured outputs per email id and records every call.

    Emails without a script fall back to the rule engine so whole-graph tests stay
    deterministic.
    """

    name = "scripted"

    def __init__(
        self,
        classifications: Optional[dict[str, Classification]] = None,
        extractions: Optional[dict[str, ExtractedTransaction]] = None,
    ):
        self.classifications = classifications or {}
        self.extractions = extractions or {}
        self.classify_calls: list[str] = []
        self.extract_calls: list[str] = []

    def classify(self, email: Email) -> Classification:
        self.classify_calls.append(email.id)
        return self.classifications.get(email.id) or rule_classify(email)

    def extract(self, email: Email, classification: Classification) -> ExtractedTransaction:
        self.extract_calls.append(email.id)
        return self.extractions.get(email.id) or rule_extract(email)


class FailingClient:
    """Fake LLM that fails the first `fail_times` calls of each kind (all, by default)."""

    name = "failing"

    def __init__(self, fail_times: int = 10**9):
        self.fail_times = fail_times
        self.classify_attempts = 0
        self.extract_attempts = 0

    def classify(self, email: Email) -> Classification:
        self.classify_attempts += 1
        if self.classify_attempts <= self.fail_times:
            raise TimeoutError("simulated provider timeout")
        return rule_classify(email)

    def extract(self, email: Email, classification: Classification) -> ExtractedTransaction:
        self.extract_attempts += 1
        if self.extract_attempts <= self.fail_times:
            raise ConnectionError("simulated connection reset")
        return rule_extract(email)


@pytest.fixture
def run_graph(settings):
    """Build a graph with an in-memory checkpointer and run it on a list of emails."""

    def _run(raw_emails, client=None, thread_id="t1", settings_override: Optional[dict] = None, **extra):
        s = replace(settings, **(settings_override or {}))
        graph = build_graph(client=client or ScriptedClient(), settings=s, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": thread_id}}
        graph.invoke({"raw_emails": raw_emails, **extra}, config)
        return graph, config

    return _run
