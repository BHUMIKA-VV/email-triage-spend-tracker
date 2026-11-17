"""Human-in-the-loop review: what triggers a review, and how decisions are applied.

Review items are built per EMAIL (classification + its transaction together), so a
reviewer sees the whole picture once instead of answering two separate prompts.

Decisions are stored append-only in `review_decisions` (event-sourced) and applied
when aggregating. The raw LLM outputs are never mutated, so a run is auditable:
you can always see what the model said and what the human changed.
"""

from __future__ import annotations

from typing import Optional

from config import Settings


def amount_in_inr(amount: Optional[float], currency: Optional[str], fx: dict[str, float]) -> Optional[float]:
    if amount is None:
        return None
    rate = fx.get((currency or "INR").upper())
    return None if rate is None else amount * rate


def review_reasons(classification: dict, transaction: Optional[dict], settings: Settings) -> list[str]:
    reasons: list[str] = []
    conf = classification.get("confidence", 0.0)
    if conf < settings.review_confidence_threshold:
        reasons.append(
            f"classification confidence {conf:.2f} < {settings.review_confidence_threshold:.2f}"
        )
    if transaction and transaction.get("amount") is not None:
        inr = amount_in_inr(transaction["amount"], transaction.get("currency"), settings.fx_to_inr)
        if inr is None:
            reasons.append(f"amount in unsupported currency {transaction.get('currency')!r}")
        elif inr > settings.spend_review_threshold_inr:
            reasons.append(
                f"amount {transaction['amount']:,.2f} {transaction.get('currency') or ''} "
                f"exceeds review threshold (Rs.{settings.spend_review_threshold_inr:,.0f})".replace("  ", " ")
            )
    if transaction and transaction.get("extraction_method") == "fallback_rules":
        reasons.append("LLM extraction failed; values come from fallback rules")
    return reasons


def build_review_queue(
    emails: list[dict], classifications: list[dict], transactions: list[dict], settings: Settings
) -> list[dict]:
    """One review item per email that trips at least one gate, in email order."""
    by_email = {c["email_id"]: c for c in classifications}
    txn_by_email: dict[str, dict] = {}
    for t in transactions:
        for eid in t["source_email_ids"]:
            txn_by_email[eid] = t

    queue: list[dict] = []
    seen_txns: set[str] = set()
    for email in emails:
        cls = by_email.get(email["id"])
        if cls is None:
            continue
        txn = txn_by_email.get(email["id"])
        # A merged transaction is reviewed once, on its first source email.
        txn_for_review = txn if txn and txn["txn_id"] not in seen_txns else None
        reasons = review_reasons(cls, txn_for_review, settings)
        if not reasons:
            continue
        if txn_for_review:
            seen_txns.add(txn_for_review["txn_id"])
        queue.append(
            {
                "review_id": f"review-{email['id']}",
                "email_id": email["id"],
                "subject": email["subject"],
                "from": email["from"],
                "email_date": email["date"],
                "reasons": reasons,
                "classification": cls,
                "transaction": txn_for_review,
            }
        )
    return queue


def apply_decisions(
    classifications: list[dict], transactions: list[dict], review_queue: list[dict], decisions: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return (final_classifications, final_transactions) with human decisions applied.

    approve -> keep as-is (marked `review_status=approved`)
    reject  -> transaction is excluded from spend totals; classification kept
    correct -> patch classification category and/or transaction fields; a correction
               on an email with no transaction creates one (human-entered data)
    """
    item_by_review = {r["review_id"]: r for r in review_queue}
    final_cls = {c["email_id"]: dict(c) for c in classifications}
    final_txn = {t["txn_id"]: dict(t) for t in transactions}

    for decision in decisions:
        item = item_by_review.get(decision["review_id"])
        if item is None:
            continue
        email_id = item["email_id"]
        action = decision["action"]
        corrections = decision.get("corrections", {})
        txn_id = item["transaction"]["txn_id"] if item.get("transaction") else None

        if email_id in final_cls:
            final_cls[email_id]["review_status"] = action
            if action == "correct" and corrections.get("category"):
                final_cls[email_id]["original_category"] = final_cls[email_id]["category"]
                final_cls[email_id]["category"] = corrections["category"]
                final_cls[email_id]["confidence"] = 1.0
                final_cls[email_id]["reason"] = "Corrected by human reviewer."

        txn_fields = {
            k: v for k, v in corrections.items()
            if k in {"amount", "currency", "merchant", "date", "spend_category", "payment_method"}
        }
        if "spend_category" in txn_fields:
            txn_fields["category"] = txn_fields.pop("spend_category")

        if txn_id and txn_id in final_txn:
            final_txn[txn_id]["review_status"] = action
            if action == "reject":
                final_txn[txn_id]["rejected"] = True
            elif action == "correct" and txn_fields:
                final_txn[txn_id].update(txn_fields)
                final_txn[txn_id]["corrected_fields"] = sorted(txn_fields)
        elif action == "correct" and txn_fields.get("amount") is not None:
            new_id = f"txn-{email_id}-manual"
            final_txn[new_id] = {
                "txn_id": new_id, "email_id": email_id, "source_email_ids": [email_id],
                "amount": None, "currency": None, "merchant": None, "date": None, "category": None,
                "payment_method": None, "order_id": None, "description": item["subject"],
                "email_date": item.get("email_date"), "email_category": final_cls.get(email_id, {}).get("category"),
                "extraction_method": "human", "notes": ["entered by human reviewer"],
                "review_status": "correct", **txn_fields,
            }

    for t in final_txn.values():
        if t.get("email_id") in final_cls:
            t["email_category"] = final_cls[t["email_id"]]["category"]
    return list(final_cls.values()), [t for t in final_txn.values() if not t.get("rejected")]
