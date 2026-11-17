"""Pydantic models: the input email, and the structured outputs the LLM must return.

The LLM is bound to these schemas with `with_structured_output`, so the model
returns validated objects (tool/JSON-schema calling) — there is no free-text
parsing anywhere in the pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

EmailCategory = Literal["Work", "Online Orders", "Subscriptions", "Miscellaneous"]
EMAIL_CATEGORIES: tuple[str, ...] = ("Work", "Online Orders", "Subscriptions", "Miscellaneous")

SpendCategory = Literal[
    "Shopping",
    "Electronics",
    "Groceries",
    "Food & Dining",
    "Travel",
    "Entertainment",
    "Software & SaaS",
    "Utilities & Bills",
    "Professional Services",
    "Health",
    "Education",
    "Other",
]

PaymentMethod = Literal[
    "UPI", "Credit Card", "Debit Card", "Card", "Net Banking", "Wallet",
    "Cash on Delivery", "Bank Transfer", "PayPal", "Other",
]


class Email(BaseModel):
    """One email as loaded from emails.json (`from` is a Python keyword, hence the alias)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    sender: str = Field(alias="from")
    subject: str = ""
    date: date
    body: str = ""

    def as_state(self) -> dict:
        return {
            "id": self.id,
            "from": self.sender,
            "subject": self.subject,
            "date": self.date.isoformat(),
            "body": self.body,
        }


class Classification(BaseModel):
    """Structured output of the classifier."""

    category: EmailCategory = Field(description="Exactly one of the four inbox categories.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Calibrated probability (0-1) that `category` is correct. Use < 0.7 when genuinely unsure.",
    )
    reason: str = Field(description="One short sentence explaining the decision.")
    may_contain_transaction: bool = Field(
        description=(
            "True if the email might record money actually paid or charged "
            "(order, receipt, invoice, renewal charge, bill payment). "
            "False for pure promotions, newsletters, and conversation."
        )
    )

    @field_validator("reason")
    @classmethod
    def _one_line(cls, v: str) -> str:
        return " ".join(v.split())


class ExtractedTransaction(BaseModel):
    """Structured output of the spend extractor.

    Every field is optional on purpose: anything not stated in the email must be
    null. `is_transaction=False` means "no payment recorded here".
    """

    is_transaction: bool = Field(
        description="True only if the email records money that was paid, charged, or is owed for a purchase/service."
    )
    amount: Optional[float] = Field(
        default=None,
        description="Total amount paid/charged as a number, copied from the email. Null if not explicitly stated. Never estimate.",
    )
    currency: Optional[str] = Field(
        default=None, description="ISO 4217 code (INR for Rs./₹, USD for $). Null if no amount."
    )
    merchant: Optional[str] = Field(default=None, description="Business that was paid, e.g. 'Amazon', 'Netflix'.")
    date: Optional[str] = Field(
        default=None,
        description="Transaction date as YYYY-MM-DD only if a date is written in the email body; otherwise null.",
    )
    category: Optional[SpendCategory] = Field(default=None, description="Spending category of the purchase.")
    payment_method: Optional[PaymentMethod] = Field(
        default=None, description="How it was paid, only if stated (UPI, card, net banking, ...)."
    )
    order_id: Optional[str] = Field(
        default=None, description="Order / invoice / receipt number if present (used for de-duplication)."
    )
    description: Optional[str] = Field(default=None, description="Very short description of what was bought.")


class ReviewDecision(BaseModel):
    """What a human reviewer sends back through `Command(resume=...)`."""

    action: Literal["approve", "reject", "correct"]
    # Correctable classification field.
    category: Optional[EmailCategory] = None
    # Correctable transaction fields.
    amount: Optional[float] = None
    currency: Optional[str] = None
    merchant: Optional[str] = None
    date: Optional[str] = None
    spend_category: Optional[SpendCategory] = None
    payment_method: Optional[PaymentMethod] = None
    note: Optional[str] = None

    def corrections(self) -> dict:
        """Only the fields the reviewer actually set."""
        return self.model_dump(exclude={"action", "note"}, exclude_none=True)
