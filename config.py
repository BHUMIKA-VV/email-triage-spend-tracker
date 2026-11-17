"""Runtime configuration, loaded from environment variables / a .env file.

Every knob that affects behaviour (LLM provider, thresholds, dedup window,
retry policy) lives here so the graph code never reads os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


# Approximate conversion rates used ONLY to compare a foreign-currency amount
# against the INR review threshold. Aggregation never converts currencies.
DEFAULT_FX_TO_INR: dict[str, float] = {
    "INR": 1.0,
    "USD": 88.0,
    "EUR": 102.0,
    "GBP": 117.0,
    "SGD": 68.0,
    "AED": 24.0,
}


@dataclass(frozen=True)
class Settings:
    # "auto" picks anthropic -> openai -> rules depending on which API key exists.
    llm_provider: str = "auto"
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_timeout_seconds: float = 60.0

    # Human-in-the-loop gates.
    review_confidence_threshold: float = 0.7
    spend_review_threshold_inr: float = 10_000.0

    # Two transactions with the same merchant + amount + currency whose dates
    # are at most this many days apart are treated as the same purchase.
    dedup_window_days: int = 3

    # Max parallel Send workers (bounds concurrent LLM calls / rate limits).
    max_concurrency: int = 4

    checkpoint_db: str = "checkpoints.sqlite"
    fx_to_inr: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FX_TO_INR))


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
        llm_model=os.getenv("LLM_MODEL") or None,
        llm_temperature=_float("LLM_TEMPERATURE", 0.0),
        llm_max_retries=_int("LLM_MAX_RETRIES", 3),
        llm_retry_base_delay=_float("LLM_RETRY_BASE_DELAY", 1.0),
        llm_timeout_seconds=_float("LLM_TIMEOUT_SECONDS", 60.0),
        review_confidence_threshold=_float("REVIEW_CONFIDENCE_THRESHOLD", 0.7),
        spend_review_threshold_inr=_float("SPEND_REVIEW_THRESHOLD_INR", 10_000.0),
        dedup_window_days=_int("DEDUP_WINDOW_DAYS", 3),
        max_concurrency=_int("MAX_CONCURRENCY", 4),
        checkpoint_db=os.getenv("CHECKPOINT_DB", "checkpoints.sqlite"),
    )
