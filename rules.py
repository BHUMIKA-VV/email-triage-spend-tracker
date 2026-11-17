"""Deterministic, rule-based classifier and extractor.

Used in three places:
1. As the *fallback* when the LLM fails after all retries (the run never crashes).
2. As the `rules` provider so the whole graph runs offline without an API key.
3. `has_transaction_signal` is a cheap recall-oriented check that the router
   combines with the LLM's own `may_contain_transaction` flag.

It is intentionally simple and transparent; it is NOT meant to match LLM quality.
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import Classification, Email, ExtractedTransaction
from utils import merchant_from_sender, normalize_currency

# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
_CURRENCY_TOKEN = r"(?:Rs\.?|INR|₹|USD|US\$|\$|EUR|€|GBP|£)"
_AMOUNT_RE = re.compile(
    rf"(?P<cur>{_CURRENCY_TOKEN})\s?(?P<num>\d[\d,]*(?:\.\d+)?)"
    rf"|(?P<num2>\d[\d,]*(?:\.\d+)?)\s?(?P<cur2>INR|USD|EUR|GBP)\b",
    re.I,
)
_PAID_RE = re.compile(
    r"\b(paid|charged|debited|payment (?:of|received|successful)|order total|grand total|"
    r"amount (?:paid|due|payable)|total(?: amount)?|invoice|receipt|billed|renewed|auto-?debit|"
    r"purchase|booking confirmed|fare)\b",
    re.I,
)
_PROMO_RE = re.compile(
    r"(\d+\s?% off|\bsale\b|\boffer\b|\bdeal(s)?\b|\bdiscount\b|\bcoupon\b|\buse code\b|"
    r"\bflat\b|\bstarting at\b|\bprice drop\b|\blimited time\b|\bcashback offer\b)",
    re.I,
)
_ORDER_ID_RE = re.compile(
    r"(?:order|invoice|booking|receipt|pnr|txn|transaction)\s*(?:id|no\.?|number|#)?\s*[:#]?\s*"
    r"#?(?P<id>(?=[A-Z\-/_]*\d)[A-Z0-9][A-Z0-9\-/_]{3,})",  # an id must contain a digit
    re.I,
)
# A date counts as the transaction date only when introduced as one
# ("placed on", "paid ... on", "dated"); delivery / trial / next-billing dates do not.
_TXN_DATE_RE = re.compile(
    r"\b(?:placed on|paid(?: [\w ]{0,25})? on|charged on|dated|date:)\s*(20\d{2}-\d{2}-\d{2})\b", re.I
)
_FUTURE_CHARGE_RE = re.compile(r"\b(will be charged|will charge|unless you cancel|trial ends)\b", re.I)
_ORDER_PLACED_RE = re.compile(
    r"\border\b.{0,80}\b(placed|confirmed|shipped|delivered|on its way|dispatched)\b", re.I | re.S
)
_PAID_TO_RE = re.compile(r"\b(?i:paid to):?\s*(?P<name>[A-Z][\w&.' ]{1,40}?)(?:\.|,|$)", re.M)

_PAYMENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bupi\b|gpay|phonepe|paytm upi", "UPI"),
    (r"credit card", "Credit Card"),
    (r"debit card", "Debit Card"),
    (r"card ending|\bcard\b", "Card"),
    (r"net ?banking", "Net Banking"),
    (r"wallet|amazon pay balance|paytm wallet", "Wallet"),
    (r"cash on delivery|\bcod\b", "Cash on Delivery"),
    (r"neft|rtgs|imps|bank transfer|wire transfer", "Bank Transfer"),
    (r"paypal", "PayPal"),
]

_SPEND_CATEGORY_HINTS: list[tuple[str, str]] = [
    (r"swiggy|zomato|restaurant|food|pizza|dominos|instamart", "Food & Dining"),
    (r"bigbasket|blinkit|grocer|zepto", "Groceries"),
    (r"uber|ola|irctc|makemytrip|flight|train|cab|ride|indigo|hotel", "Travel"),
    (r"netflix|spotify|hotstar|prime video|bookmyshow|movie|jiocinema|youtube premium", "Entertainment"),
    (r"github|notion|figma|canva|aws|saas|copilot|slack|software|google one|icloud", "Software & SaaS"),
    (r"airtel|jio|electricity|broadband|postpaid|recharge|bill", "Utilities & Bills"),
    (r"mouse|keyboard|laptop|phone|headphone|earbuds|charger|monitor|ssd|electronics|croma", "Electronics"),
    (r"consult|freelanc|design services|professional fee|contractor|invoice", "Professional Services"),
    (r"pharmacy|medicine|apollo|1mg|pharmeasy|cult\.fit|gym", "Health"),
    (r"course|udemy|coursera|book", "Education"),
    (r"amazon|flipkart|myntra|ajio|nykaa|shirt|shoes|kurta", "Shopping"),
]

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Work": [
        r"\bsprint\b", r"\bstand-?up\b", r"\bmeeting\b", r"\bdeck\b", r"\bclient\b", r"\bproject\b",
        r"\bdeadline\b", r"\bEOD\b", r"\bquarterly\b", r"\breimburse", r"\bexpense report\b",
        r"\bvendor\b", r"\bpurchase order\b", r"\bPO\b", r"\bpull request\b", r"\bdeploy",
        r"\bon-?call\b", r"\bteam\b", r"\boffsite\b", r"\bperformance review\b", r"\bpayroll\b",
        r"\bconsulting\b", r"\bprofessional services\b", r"\bnet 30\b",
    ],
    "Online Orders": [
        r"\border\b", r"\bshipped\b", r"\bdispatched\b", r"\bdelivered\b", r"\bout for delivery\b",
        r"\btracking\b", r"\breturn\b", r"\brefund\b", r"\bbooking\b", r"\bride\b", r"\btrip\b",
        r"\bitems?:", r"\bpnr\b",
    ],
    "Subscriptions": [
        r"\brenew", r"\bsubscription\b", r"\bmembership\b", r"\bplan\b", r"\bauto-?debit\b",
        r"\bbilling cycle\b", r"\btrial\b", r"\bnewsletter\b", r"\bdigest\b", r"\bissue #?\d+",
        r"\bmonthly\b", r"\bannual\b", r"\brecurring\b",
    ],
    "Miscellaneous": [
        r"\d+\s?% off", r"\bsale\b", r"\boffer\b", r"\bdeal\b", r"\bcoupon\b", r"\bOTP\b",
        r"\bsecurity alert\b", r"\bnew sign-?in\b", r"\bbirthday\b", r"\bcongratulations\b",
        r"\bwinner\b", r"\bstatement\b", r"\bwebinar\b",
    ],
}


def _text(email: Email) -> str:
    return f"{email.subject}\n{email.body}"


def has_transaction_signal(email: Email) -> bool:
    """Recall-oriented: True if the email mentions an amount near payment wording, or an invoice."""
    text = _text(email)
    if re.search(r"\binvoice\b", text, re.I):
        return True
    return bool(_AMOUNT_RE.search(text)) and bool(_PAID_RE.search(text))


def looks_promotional(email: Email) -> bool:
    return bool(_PROMO_RE.search(_text(email)))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
def rule_classify(email: Email) -> Classification:
    text = _text(email)
    scores: dict[str, float] = {}
    for category, patterns in _CATEGORY_KEYWORDS.items():
        scores[category] = float(sum(1 for p in patterns if re.search(p, text, re.I)))

    # Strong structural hints.
    if looks_promotional(email) and not re.search(r"\b(paid|charged|debited|shipped|renewed)\b", text, re.I):
        scores["Miscellaneous"] += 2
    if re.search(r"\binvoice\b", text, re.I) and scores["Work"] > 0:
        scores["Work"] += 1
    if re.search(r"\brenew(ed|al)\b|\bauto-?debit\b", text, re.I):
        scores["Subscriptions"] += 1.5

    total = sum(scores.values())
    if total == 0:
        return Classification(
            category="Miscellaneous", confidence=0.4,
            reason="No recognisable keywords; defaulted to Miscellaneous.",
            may_contain_transaction=has_transaction_signal(email),
        )

    best = max(scores, key=lambda c: scores[c])
    share = scores[best] / total
    # Map share-of-evidence (0.25..1.0) onto a conservative 0.45..0.9 range.
    confidence = min(0.9, 0.45 + 0.6 * (share - 0.25))
    if scores[best] < 2:  # a single keyword is weak evidence
        confidence = min(confidence, 0.75)
    confidence = round(confidence, 2)
    return Classification(
        category=best,  # type: ignore[arg-type]
        confidence=max(0.3, confidence),
        reason=f"Keyword rules matched {best} ({scores[best]:.0f} of {total:.0f} signals).",
        may_contain_transaction=has_transaction_signal(email),
    )


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
def _pick_amount(text: str) -> tuple[Optional[float], Optional[str]]:
    candidates: list[tuple[int, float, str]] = []
    for m in _AMOUNT_RE.finditer(text):
        num = m.group("num") or m.group("num2")
        cur = m.group("cur") or m.group("cur2")
        try:
            value = float(num.replace(",", ""))
        except (TypeError, ValueError):
            continue
        # Prefer amounts whose surrounding sentence talks about paying.
        start = max(0, m.start() - 60)
        window = text[start : m.end() + 20]
        priority = 2 if re.search(r"total|charged|paid|debited|amount|payable|due", window, re.I) else 1
        candidates.append((priority, value, normalize_currency(cur) or "INR"))
    if not candidates:
        return None, None
    top_priority = max(c[0] for c in candidates)
    best = [c for c in candidates if c[0] == top_priority]
    # Among equally-prioritised amounts, the last "total"-like one is usually the grand total.
    _, value, currency = best[-1]
    return value, currency


def rule_extract(email: Email) -> ExtractedTransaction:
    text = _text(email)
    order_match = _ORDER_ID_RE.search(text)
    # An order notice without a price (placed / shipped / delivered) is still a
    # purchase: report it with amount=null so dedup can attach it to the priced one.
    order_notice = bool(order_match and _ORDER_PLACED_RE.search(text))
    if _FUTURE_CHARGE_RE.search(text) and not order_notice and not re.search(r"\b(we charged|has been charged|paid)\b", text, re.I):
        return ExtractedTransaction(is_transaction=False)
    if not (has_transaction_signal(email) or order_notice) or (
        looks_promotional(email) and not re.search(r"\b(paid|charged|debited|invoice|order total)\b", text, re.I)
    ):
        return ExtractedTransaction(is_transaction=False)

    amount, currency = _pick_amount(text)
    lowered = text.lower()
    payment = next((label for pat, label in _PAYMENT_PATTERNS if re.search(pat, lowered)), None)
    paid_to = _PAID_TO_RE.search(email.body)
    merchant = paid_to.group("name").strip() if paid_to else merchant_from_sender(email.sender)
    spend_cat = next(
        (label for pat, label in _SPEND_CATEGORY_HINTS if re.search(pat, f"{lowered} {email.sender.lower()}")),
        "Other",
    )
    date_match = _TXN_DATE_RE.search(email.body)
    return ExtractedTransaction(
        is_transaction=True,
        amount=amount,
        currency=currency,
        merchant=merchant,
        date=date_match.group(1) if date_match else None,
        category=spend_cat,  # type: ignore[arg-type]
        payment_method=payment,  # type: ignore[arg-type]
        order_id=order_match.group("id") if order_match else None,
        description=email.subject[:80] or None,
    )
