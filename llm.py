"""LLM provider abstraction.

Nodes depend on the small `LLMClient` protocol, not on a vendor SDK. That gives:
* provider switching via LLM_PROVIDER (anthropic | openai | rules | auto),
* dependency injection in tests (scripted / failing clients, no API key needed).

`LangChainLLMClient` binds each Pydantic schema with `with_structured_output`,
so the provider returns a validated object via tool / JSON-schema calling.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings
from prompts import CLASSIFY_SYSTEM, CLASSIFY_USER, EXTRACT_SYSTEM, EXTRACT_USER
from rules import rule_classify, rule_extract
from schemas import Classification, Email, ExtractedTransaction

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.8-flash",          # free tier via Google AI Studio
    "groq": "openai/gpt-oss-120b",     # free tier via console.groq.com
}

# Checked in this order when LLM_PROVIDER=auto.
_KEY_ENV = [
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("gemini", "GOOGLE_API_KEY"),
    ("groq", "GROQ_API_KEY"),
]


class LLMClient(Protocol):
    name: str

    def classify(self, email: Email) -> Classification: ...

    def extract(self, email: Email, classification: Classification) -> ExtractedTransaction: ...


class LangChainLLMClient:
    """Wraps any LangChain chat model that supports `with_structured_output`."""

    def __init__(self, chat_model: Any, name: str = "langchain"):
        self.name = name
        self._classifier = chat_model.with_structured_output(Classification)
        self._extractor = chat_model.with_structured_output(ExtractedTransaction)

    @staticmethod
    def _fields(email: Email) -> dict[str, str]:
        return {
            "sender": email.sender,
            "date": email.date.isoformat(),
            "subject": email.subject,
            "body": email.body,
        }

    def classify(self, email: Email) -> Classification:
        result = self._classifier.invoke(
            [SystemMessage(CLASSIFY_SYSTEM), HumanMessage(CLASSIFY_USER.format(**self._fields(email)))]
        )
        if not isinstance(result, Classification):  # some providers return dicts
            result = Classification.model_validate(result)
        return result

    def extract(self, email: Email, classification: Classification) -> ExtractedTransaction:
        result = self._extractor.invoke(
            [
                SystemMessage(EXTRACT_SYSTEM),
                HumanMessage(EXTRACT_USER.format(category=classification.category, **self._fields(email))),
            ]
        )
        if not isinstance(result, ExtractedTransaction):
            result = ExtractedTransaction.model_validate(result)
        return result


class RuleBasedClient:
    """Offline, deterministic stand-in for an LLM (keyword + regex rules)."""

    name = "rules"

    def classify(self, email: Email) -> Classification:
        return rule_classify(email)

    def extract(self, email: Email, classification: Classification) -> ExtractedTransaction:
        return rule_extract(email)


def resolve_provider(settings: Settings) -> str:
    provider = settings.llm_provider
    if provider != "auto":
        return provider
    for name, env in _KEY_ENV:
        if os.getenv(env):
            return name
    return "rules"


def build_llm_client(settings: Settings) -> LLMClient:
    provider = resolve_provider(settings)
    model = settings.llm_model or DEFAULT_MODELS.get(provider)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(
            model=model, temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds, max_retries=0,  # retries are handled in nodes
        )
        return LangChainLLMClient(chat, name=f"anthropic:{model}")
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            model=model, temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds, max_retries=0,
        )
        return LangChainLLMClient(chat, name=f"openai:{model}")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat = ChatGoogleGenerativeAI(
            model=model, temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds, max_retries=0,
        )
        return LangChainLLMClient(chat, name=f"gemini:{model}")
    if provider == "groq":
        from langchain_groq import ChatGroq

        chat = ChatGroq(
            model=model, temperature=settings.llm_temperature,
            request_timeout=settings.llm_timeout_seconds, max_retries=0,
        )
        return LangChainLLMClient(chat, name=f"groq:{model}")
    if provider == "rules":
        logger.warning("No LLM API key configured - using the offline rule-based classifier/extractor.")
        return RuleBasedClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use anthropic, openai, gemini, groq, rules or auto.")
