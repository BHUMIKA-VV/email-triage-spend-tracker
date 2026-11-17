"""Small deterministic helpers: normalisation, grounding, retry, date handling."""

from __future__ import annotations

import logging
import re
import time
from datetime import date
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

# --------------------------------------------------------------------------- #
# Currency / numbers
# --------------------------------------------------------------------------- #
_CURRENCY_ALIASES = {
    "rs": "INR", "rs.": "INR", "inr": "INR", "₹": "INR", "rupees": "INR",
    "$": "USD", "usd": "USD", "us$": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
    "sgd": "SGD", "aed": "AED",
}

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}

_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def normalize_currency(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    if key in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[key]
    return key.upper() if len(key) == 3 and key.isalpha() else raw.strip().upper()


def numbers_in_text(text: str) -> set[float]:
    """Every number written in `text`, with Indian/Western thousands separators removed."""
    values: set[float] = set()
    for match in _NUMBER_RE.findall(text or ""):
        try:
            values.add(round(float(match.replace(",", "")), 2))
        except ValueError:
            continue
    return values


def amount_is_grounded(amount: Optional[float], text: str) -> bool:
    """True if `amount` literally appears in the email text.

    This is the main anti-hallucination guard: an LLM-extracted amount that
    cannot be found in the source text is discarded (set to null).
    """
    if amount is None:
        return True
    target = round(float(amount), 2)
    return any(abs(target - n) < 0.005 for n in numbers_in_text(text))


def format_money(amount: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency)
    return f"{symbol}{amount:,.2f}" if symbol else f"{amount:,.2f} {currency}"


# --------------------------------------------------------------------------- #
# Merchants / ids / dates
# --------------------------------------------------------------------------- #
_MERCHANT_SUFFIX_RE = re.compile(
    r"\b(inc|ltd|llc|pvt|private|limited|india|technologies|retail|online|\.com|\.in)\b\.?", re.I
)

KNOWN_DOMAINS: dict[str, str] = {
    "amazon": "Amazon", "flipkart": "Flipkart", "myntra": "Myntra", "swiggy": "Swiggy",
    "zomato": "Zomato", "bigbasket": "BigBasket", "blinkit": "Blinkit", "nykaa": "Nykaa",
    "netflix": "Netflix", "spotify": "Spotify", "hotstar": "Disney+ Hotstar",
    "jiocinema": "JioCinema", "youtube": "YouTube", "google": "Google", "apple": "Apple",
    "airtel": "Airtel", "jio": "Jio", "makemytrip": "MakeMyTrip", "irctc": "IRCTC",
    "uber": "Uber", "olacabs": "Ola", "ola": "Ola", "github": "GitHub", "notion": "Notion",
    "linkedin": "LinkedIn", "croma": "Croma", "tatacliq": "Tata CLiQ", "ajio": "AJIO",
    "bookmyshow": "BookMyShow", "cult": "cult.fit", "hdfcbank": "HDFC Bank", "icicibank": "ICICI Bank",
    "substack": "Substack", "the-ken": "The Ken", "figma": "Figma", "canva": "Canva",
}


def normalize_merchant(name: Optional[str]) -> str:
    """'Amazon.in' / 'AMAZON India Pvt Ltd' / 'amazon' -> 'amazon'."""
    if not name:
        return ""
    cleaned = _MERCHANT_SUFFIX_RE.sub(" ", name.lower())
    return re.sub(r"[^a-z0-9+]", "", cleaned)


# Mailbox providers and payment gateways are never the merchant.
NON_MERCHANT_DOMAINS = {
    "gmail", "yahoo", "outlook", "hotmail", "icloud", "proton", "company",
    "razorpay", "paytm", "phonepe", "payu", "cashfree", "stripe", "paypal",
}


def merchant_from_sender(sender: str) -> Optional[str]:
    """Best-effort merchant name from the sender domain, e.g. orders@amazon.in -> Amazon."""
    domain = sender.split("@")[-1].lower()
    parts = [p for p in re.split(r"[.]", domain) if p]
    for part in parts:
        if part in KNOWN_DOMAINS:
            return KNOWN_DOMAINS[part]
    for key, value in KNOWN_DOMAINS.items():
        if re.search(rf"(^|[.\-]){re.escape(key)}([.\-]|$)", domain):
            return value
    label = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
    if not label or label in NON_MERCHANT_DOMAINS or any(n in domain for n in NON_MERCHANT_DOMAINS):
        return None
    return label.replace("-", " ").title()


def normalize_order_id(order_id: Optional[str]) -> str:
    if not order_id:
        return ""
    return re.sub(r"[^a-z0-9]", "", order_id.lower())


def parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
PERMANENT_HTTP_STATUSES = {400, 401, 403, 404}


def _http_status(exc: BaseException) -> Optional[int]:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    """False for client errors that will fail identically on every attempt.

    Walks the exception chain because LangChain wraps SDK errors (the HTTP status
    lives on the `__cause__`). Rate limits (429), 5xx, timeouts, network errors and
    schema-validation errors stay retryable.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = _http_status(current)
        if status in PERMANENT_HTTP_STATUSES:
            return False
        # A 429 for an exhausted DAILY quota won't clear in seconds (unlike a per-minute limit).
        if status == 429 and re.search(r"per ?day|perday", str(current), re.I):
            return False
        current = current.__cause__ or current.__context__
    return True


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    label: str = "llm call",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` up to `attempts` times with exponential backoff; re-raise the last error.

    Permanent errors (bad key, unknown model, no credit) are raised immediately:
    retrying them only wastes time before the fallback kicks in.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider/network/validation errors
            last_exc = exc
            if not is_retryable(exc):
                logger.warning("%s failed with a permanent error, not retrying: %s", label, exc)
                raise
            logger.warning("%s failed (attempt %d/%d): %s", label, attempt, attempts, exc)
            if attempt < attempts:
                sleep(base_delay * (2 ** (attempt - 1)))
    assert last_exc is not None
    raise last_exc
