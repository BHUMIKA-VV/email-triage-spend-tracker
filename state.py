"""Typed graph state.

Two state schemas:

* `TriageState` - the parent graph. Lists written by many parallel workers
  (`classifications`, `transactions`, `errors`) and by the repeating review
  node (`review_decisions`) use the `operator.add` reducer, so concurrent
  updates are concatenated instead of overwriting each other.
* `EmailState`  - the per-email subgraph that each `Send` worker runs. It shares
  the reducer keys with the parent, which is how worker results flow back up.

All values are plain JSON-compatible dicts/strings so the SQLite checkpointer
can serialise them without custom types.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class TriageState(TypedDict, total=False):
    # ---- input -------------------------------------------------------------
    source_path: str                 # path to emails.json (read by `ingest`)
    raw_emails: list[dict]           # alternative: pass emails directly (tests)
    as_of: Optional[str]             # YYYY-MM-DD; "this month" is relative to it

    # ---- ingest ------------------------------------------------------------
    emails: list[dict]

    # ---- written by parallel per-email workers (reducers) -------------------
    classifications: Annotated[list[dict], operator.add]
    transactions: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]

    # ---- post-processing -----------------------------------------------------
    deduped_transactions: list[dict]
    review_queue: list[dict]
    review_decisions: Annotated[list[dict], operator.add]

    # ---- aggregate (human decisions applied) ---------------------------------
    final_classifications: list[dict]
    final_transactions: list[dict]
    summary: dict[str, Any]


class EmailState(TypedDict, total=False):
    """State of the per-email subgraph (classify -> route -> extract)."""

    email: dict
    classification: dict             # local to the subgraph, read by the router
    classifications: Annotated[list[dict], operator.add]
    transactions: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
