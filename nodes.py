"""Graph nodes and routing functions.

Nodes are built by `make_nodes(client, settings)` so the LLM client and settings
are injected (closures) rather than read from globals. Tests inject scripted or
failing clients; production injects a LangChain chat model.

Node boundaries (justified in the README):
  ingest -> [Send per email] classify -> (route) -> extract_transaction
         -> deduplicate -> review_gate -> human_review (loops) -> aggregate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from langgraph.graph import END
from langgraph.types import Send, interrupt
from pydantic import ValidationError

from aggregation import summarize
from config import Settings
from dedup import deduplicate
from llm import LLMClient
from review import apply_decisions, build_review_queue
from rules import has_transaction_signal, rule_classify, rule_extract
from schemas import Classification, Email, ExtractedTransaction, ReviewDecision
from state import EmailState, TriageState
from utils import amount_is_grounded, call_with_retry, merchant_from_sender, normalize_currency, parse_iso_date

logger = logging.getLogger(__name__)

FALLBACK_CONFIDENCE_CAP = 0.5  # always below the review threshold -> fallback results get a human look


# --------------------------------------------------------------------------- #
# Routing functions (pure, unit-tested in tests/test_routing.py)
# --------------------------------------------------------------------------- #
def needs_extraction(email: Email, classification: dict) -> bool:
    """Recall-oriented gate: send to extraction if ANY signal says money may be involved.

    * the classifier's own `may_contain_transaction` flag,
    * the email is an Online Order (orders are purchases by definition),
    * a deterministic signal: the word "invoice", or an amount next to payment
      wording. This is what guarantees a Work invoice is still checked even if
      the classifier says "Work" with may_contain_transaction=False.
    Precision is the extractor's job: it may still return "no transaction".
    """
    return (
        bool(classification.get("may_contain_transaction"))
        or classification.get("category") == "Online Orders"
        or has_transaction_signal(email)
    )


def route_after_classify(state: EmailState) -> Literal["extract_transaction", "__end__"]:
    email = Email.model_validate(state["email"])
    return "extract_transaction" if needs_extraction(email, state["classification"]) else END


def pending_reviews(state: TriageState) -> list[dict]:
    decided = {d["review_id"] for d in state.get("review_decisions", [])}
    return [r for r in state.get("review_queue", []) if r["review_id"] not in decided]


def route_review(state: TriageState) -> Literal["human_review", "aggregate"]:
    return "human_review" if pending_reviews(state) else "aggregate"


def dispatch_emails(state: TriageState) -> list[Send] | Literal["deduplicate"]:
    """Fan out: one `Send` per email to the per-email subgraph (map step)."""
    emails = state.get("emails", [])
    if not emails:
        return "deduplicate"
    return [Send("process_email", {"email": e}) for e in emails]


# --------------------------------------------------------------------------- #
# Node factory
# --------------------------------------------------------------------------- #
@dataclass
class Nodes:
    ingest: Callable[[TriageState], dict]
    classify: Callable[[EmailState], dict]
    extract_transaction: Callable[[EmailState], dict]
    deduplicate: Callable[[TriageState], dict]
    review_gate: Callable[[TriageState], dict]
    human_review: Callable[[TriageState], dict]
    aggregate: Callable[[TriageState], dict]


def _load_raw(state: TriageState) -> list[Any]:
    if state.get("raw_emails") is not None:
        return list(state["raw_emails"])
    path = Path(state.get("source_path") or "emails.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of emails")
    return data


def make_nodes(client: LLMClient, settings: Settings) -> Nodes:
    retry_kwargs = dict(attempts=settings.llm_max_retries, base_delay=settings.llm_retry_base_delay)

    # ---------------------------------------------------------------- ingest
    def ingest(state: TriageState) -> dict:
        emails, errors, seen = [], [], set()
        for i, raw in enumerate(_load_raw(state)):
            try:
                email = Email.model_validate(raw)
            except ValidationError as exc:
                errors.append({"email_id": str(raw.get("id", f"#{i}")) if isinstance(raw, dict) else f"#{i}",
                               "stage": "ingest", "error": f"invalid email skipped: {exc.errors()[0]['msg']}"})
                continue
            if email.id in seen:
                errors.append({"email_id": email.id, "stage": "ingest", "error": "duplicate email id skipped"})
                continue
            seen.add(email.id)
            emails.append(email.as_state())
        logger.info("ingested %d emails (%d skipped)", len(emails), len(errors))
        return {"emails": emails, "errors": errors}

    # -------------------------------------------------------------- classify
    def classify(state: EmailState) -> dict:
        email = Email.model_validate(state["email"])
        errors: list[dict] = []
        try:
            result = call_with_retry(lambda: client.classify(email), label=f"classify {email.id}", **retry_kwargs)
            method = client.name
        except Exception as exc:  # noqa: BLE001 - never let one email crash the run
            errors.append({"email_id": email.id, "stage": "classify", "error": f"{type(exc).__name__}: {exc}"})
            result = fallback_classification(email)
            method = "fallback_rules"
        record = {"email_id": email.id, **result.model_dump(), "method": method}
        return {"classification": record, "classifications": [record], "errors": errors}

    # --------------------------------------------------- extract_transaction
    def extract_transaction(state: EmailState) -> dict:
        email = Email.model_validate(state["email"])
        cls_record = state["classification"]
        classification = Classification.model_validate(
            {k: cls_record[k] for k in ("category", "confidence", "reason", "may_contain_transaction")}
        )
        errors: list[dict] = []
        try:
            extracted = call_with_retry(
                lambda: client.extract(email, classification), label=f"extract {email.id}", **retry_kwargs
            )
            method = "llm" if client.name != "rules" else "rules"
        except Exception as exc:  # noqa: BLE001
            errors.append({"email_id": email.id, "stage": "extract", "error": f"{type(exc).__name__}: {exc}"})
            try:
                extracted = rule_extract(email)
            except Exception as rule_exc:  # noqa: BLE001 - last resort: record nothing rather than crash
                errors.append({"email_id": email.id, "stage": "extract", "error": f"fallback failed: {rule_exc}"})
                extracted = ExtractedTransaction(is_transaction=False)
            method = "fallback_rules"

        txn = postprocess_extraction(email, extracted, cls_record["category"], method)
        return {"transactions": [txn] if txn else [], "errors": errors}

    # ----------------------------------------------------------- deduplicate
    def deduplicate_node(state: TriageState) -> dict:
        deduped = deduplicate(state.get("transactions", []), settings.dedup_window_days)
        return {"deduped_transactions": deduped}

    # ----------------------------------------------------------- review_gate
    def review_gate(state: TriageState) -> dict:
        queue = build_review_queue(
            state.get("emails", []), state.get("classifications", []),
            state.get("deduped_transactions", []), settings,
        )
        return {"review_queue": queue}

    # ---------------------------------------------------------- human_review
    def human_review(state: TriageState) -> dict:
        """Pause for ONE pending item. The graph loops back here until the queue is empty.

        On resume LangGraph re-runs this node from the top and `interrupt()` returns
        the reviewer's value, so everything before it must be side-effect free.
        """
        pending = pending_reviews(state)
        item = pending[0]
        payload = {
            **item,
            "pending_count": len(pending),
            "allowed_actions": ["approve", "reject", "correct"],
            "correctable_fields": ["category", "amount", "currency", "merchant", "date",
                                   "spend_category", "payment_method"],
        }
        response = interrupt(payload)
        while True:
            try:
                if isinstance(response, str):
                    response = {"action": response}
                decision = ReviewDecision.model_validate(response)
                break
            except ValidationError as exc:
                # Ask again instead of crashing the run on a malformed resume value.
                response = interrupt({**payload, "error": f"invalid decision: {exc.errors()[0]['msg']}"})

        return {"review_decisions": [{
            "review_id": item["review_id"],
            "email_id": item["email_id"],
            "action": decision.action,
            "corrections": decision.corrections(),
            "note": decision.note,
        }]}

    # ------------------------------------------------------------- aggregate
    def aggregate(state: TriageState) -> dict:
        final_cls, final_txn = apply_decisions(
            state.get("classifications", []), state.get("deduped_transactions", []),
            state.get("review_queue", []), state.get("review_decisions", []),
        )
        order = {e["id"]: i for i, e in enumerate(state.get("emails", []))}
        final_cls.sort(key=lambda c: order.get(c["email_id"], 0))
        summary = summarize(state.get("emails", []), final_cls, final_txn, as_of=state.get("as_of"))
        return {"final_classifications": final_cls, "final_transactions": final_txn, "summary": summary}

    return Nodes(ingest, classify, extract_transaction, deduplicate_node, review_gate, human_review, aggregate)


def fallback_classification(email: Email) -> Classification:
    """Rule-based classification with confidence capped below the review threshold.

    If even the rules fail, return Miscellaneous with confidence 0 - the email still
    flows through the graph and is guaranteed to reach a human reviewer.
    """
    try:
        fb = rule_classify(email)
        return fb.model_copy(update={
            "confidence": min(fb.confidence, FALLBACK_CONFIDENCE_CAP),
            "reason": f"[fallback] LLM unavailable; {fb.reason}",
        })
    except Exception:  # noqa: BLE001
        return Classification(
            category="Miscellaneous", confidence=0.0,
            reason="[fallback] LLM and rules both failed; needs manual review.",
            may_contain_transaction=True,
        )


# --------------------------------------------------------------------------- #
# Extraction post-processing: the deterministic anti-hallucination layer
# --------------------------------------------------------------------------- #
def postprocess_extraction(
    email: Email, extracted: ExtractedTransaction, email_category: str, method: str
) -> dict | None:
    """Validate an extraction against the source email.

    * `is_transaction=False` -> no transaction at all (nothing is guessed).
    * The amount must literally appear in the subject/body, otherwise it is nulled.
    * A non-ISO date is nulled; currency symbols are normalised to ISO codes.
    """
    if not extracted.is_transaction:
        return None

    notes: list[str] = []
    text = f"{email.subject}\n{email.body}"
    amount = extracted.amount
    if amount is not None and not amount_is_grounded(amount, text):
        notes.append(f"extracted amount {amount} not found in email text; discarded")
        amount = None
    if amount is not None and amount <= 0:
        notes.append(f"non-positive amount {amount} discarded")
        amount = None

    currency = normalize_currency(extracted.currency) if amount is not None else None
    if amount is not None and currency is None:
        notes.append("currency not stated")

    txn_date = extracted.date if parse_iso_date(extracted.date) else None
    if extracted.date and txn_date is None:
        notes.append(f"unparseable date {extracted.date!r} discarded")

    merchant = extracted.merchant
    if not merchant:
        merchant = merchant_from_sender(email.sender)
        if merchant:
            notes.append("merchant inferred from sender domain")

    return {
        "txn_id": f"txn-{email.id}",
        "email_id": email.id,
        "source_email_ids": [email.id],
        "amount": amount,
        "currency": currency,
        "merchant": merchant,
        "date": txn_date,
        "category": extracted.category,
        "payment_method": extracted.payment_method,
        "order_id": extracted.order_id,
        "description": extracted.description,
        "email_date": email.date.isoformat(),
        "email_category": email_category,
        "extraction_method": method,
        "notes": notes,
    }
