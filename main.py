"""CLI entry point.

    python main.py                       # run on emails.json; stops at the first review
    python main.py run --interactive     # answer reviews in the terminal
    python main.py run --auto-approve    # approve every review (demo / CI)
    python main.py status                # show where the last run is paused
    python main.py resume --action approve
    python main.py resume --action reject
    python main.py resume --action correct --set amount=42000 --set category=Work
    python main.py summary --month 2026-10   # spend for a whole month from a finished run
    python main.py diagram               # write diagrams/graph.mmd

State is checkpointed in SQLite (CHECKPOINT_DB), so `resume` works from a new
process using the same --thread-id (defaults to the last run's id).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph.types import Command
from pydantic import ValidationError

from aggregation import summarize
from config import load_settings
from graph import build_graph, render_mermaid, sqlite_checkpointer
from llm import build_llm_client
from schemas import ReviewDecision
from utils import format_money

LAST_THREAD_FILE = Path(".last_thread")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # ₹ on Windows consoles


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def _hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _money_map(m: dict[str, float]) -> str:
    return ", ".join(format_money(v, c) for c, v in m.items()) or "-"


def print_classifications(state: dict) -> None:
    _hr("CLASSIFICATIONS")
    rows = state.get("final_classifications") or state.get("classifications", [])
    order = {e["id"]: i for i, e in enumerate(state.get("emails", []))}
    subjects = {e["id"]: e["subject"] for e in state.get("emails", [])}
    for c in sorted(rows, key=lambda r: order.get(r["email_id"], 0)):
        flag = " (fallback)" if c.get("method") == "fallback_rules" else ""
        reviewed = f" [review: {c['review_status']}]" if c.get("review_status") else ""
        print(f"{c['email_id']}  {c['category']:<14} conf={c['confidence']:.2f}{flag}{reviewed}")
        print(f"      {subjects.get(c['email_id'], '')[:66]}")
        print(f"      -> {c['reason']}")


def print_transactions(state: dict) -> None:
    _hr("TRANSACTIONS (after de-duplication)")
    txns = state.get("final_transactions") or state.get("deduped_transactions", [])
    if not txns:
        print("none")
    for t in txns:
        amount = format_money(t["amount"], t["currency"] or "?") if t.get("amount") is not None else "amount: null"
        print(
            f"{t['txn_id']:<22} {amount:>14}  {t.get('merchant') or '?':<18} "
            f"{t.get('category') or '-':<20} {t.get('payment_method') or '-':<16} "
            f"date={t.get('date') or 'null'}"
        )
        extras = []
        if len(t["source_email_ids"]) > 1:
            extras.append(f"merged from {', '.join(t['source_email_ids'])}")
        if t.get("review_status"):
            extras.append(f"review: {t['review_status']}")
        extras += t.get("notes", [])
        for e in extras:
            print(f"{'':<23}* {e}")
    raw = len(state.get("transactions", []))
    if raw:
        print(f"\n{raw} raw extractions -> {len(state.get('deduped_transactions', []))} unique transactions")


def print_review(item: dict) -> None:
    _hr(f"HUMAN REVIEW REQUIRED  ({item.get('pending_count', '?')} pending)")
    print(f"Email   : {item['email_id']}  from {item['from']}")
    print(f"Subject : {item['subject']}")
    print("Reasons : " + "; ".join(item["reasons"]))
    c = item["classification"]
    print(f"Class.  : {c['category']} (conf {c['confidence']:.2f}) - {c['reason']}")
    t = item.get("transaction")
    if t:
        print(
            "Txn     : " + json.dumps(
                {k: t.get(k) for k in ("amount", "currency", "merchant", "date", "category", "payment_method")},
                ensure_ascii=False,
            )
        )
    if item.get("error"):
        print(f"Error   : {item['error']}")


def print_summary(summary: dict) -> None:
    _hr(f"SPEND SUMMARY  {summary['period']}")
    print(f"Total spend : {_money_map(summary['total_spend'])}")
    print("\nBy spend category:")
    for k, v in summary["spend_by_category"].items():
        print(f"  {k:<24} {_money_map(v)}")
    print("\nBy merchant:")
    for k, v in summary["spend_by_merchant"].items():
        print(f"  {k:<24} {_money_map(v)}")
    print("\nBy email category:")
    for k, v in summary["spend_by_email_category"].items():
        print(f"  {k:<24} {_money_map(v)}")
    print("\nEmails per category:")
    for k, v in summary["email_counts"].items():
        print(f"  {k:<24} {v}")
    print(f"  {'(total processed)':<24} {summary['emails_processed']}")
    if summary["transactions_without_amount"]:
        print("\nTransactions with no stated amount (not counted, never guessed):")
        for t in summary["transactions_without_amount"]:
            print(f"  {t['txn_id']} ({t['merchant']})")
    if summary["transactions_outside_period"]:
        print("\nExcluded (outside period): " + ", ".join(summary["transactions_outside_period"]))


def print_errors(state: dict) -> None:
    if state.get("errors"):
        _hr("ERRORS (handled, run continued)")
        for e in state["errors"]:
            print(f"{e['email_id']} [{e['stage']}] {e['error']}")


# --------------------------------------------------------------------------- #
# Run / resume helpers
# --------------------------------------------------------------------------- #
def _graph(args):
    settings = load_settings()
    client = build_llm_client(settings)
    graph = build_graph(client=client, settings=settings, checkpointer=sqlite_checkpointer(args.db or settings.checkpoint_db))
    return graph, settings, client


def _config(thread_id: str, settings) -> dict:
    return {"configurable": {"thread_id": thread_id}, "max_concurrency": settings.max_concurrency}


def _pending_interrupt(graph, config) -> dict | None:
    snapshot = graph.get_state(config)
    for intr in snapshot.interrupts:
        return intr.value
    return None


def _ask_reviewer(item: dict) -> dict:
    while True:
        action = input("Action [a]pprove / [r]eject / [c]orrect: ").strip().lower()
        action = {"a": "approve", "r": "reject", "c": "correct"}.get(action, action)
        if action in ("approve", "reject"):
            return {"action": action}
        if action == "correct":
            raw = input("Corrections as key=value, comma separated (e.g. amount=42000,category=Work): ")
            try:
                return _decision_from_pairs("correct", [p for p in raw.split(",") if p.strip()], None)
            except (ValueError, ValidationError) as exc:
                print(f"invalid correction: {exc}")
                continue
        print("please type approve, reject or correct")


def _decision_from_pairs(action: str, pairs: list[str], note: str | None) -> dict:
    data: dict[str, Any] = {"action": action, "note": note}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair!r}")
        key, value = (s.strip() for s in pair.split("=", 1))
        data[key] = float(value.replace(",", "")) if key == "amount" else value
    return ReviewDecision.model_validate(data).model_dump(exclude_none=True)


def drive(graph, config, settings, first_input, *, interactive: bool, auto_approve: bool) -> None:
    """Invoke the graph and keep resuming while the reviewer answers in-process."""
    graph.invoke(first_input, config)
    while (item := _pending_interrupt(graph, config)) is not None:
        print_review(item)
        if auto_approve:
            print("--auto-approve: approving")
            decision: dict = {"action": "approve", "note": "auto-approved"}
        elif interactive:
            decision = _ask_reviewer(item)
        else:
            tid = config["configurable"]["thread_id"]
            print(
                f"\nRun paused at the review gate (thread_id={tid}). Resume with one of:\n"
                f"  python main.py resume --thread-id {tid} --action approve\n"
                f"  python main.py resume --thread-id {tid} --action reject\n"
                f"  python main.py resume --thread-id {tid} --action correct --set amount=<n> --set category=<Work|...>\n"
                f"  python main.py resume --thread-id {tid} --interactive"
            )
            return
        graph.invoke(Command(resume=decision), config)
    report(graph.get_state(config).values)


def report(state: dict) -> None:
    print_classifications(state)
    print_transactions(state)
    print_errors(state)
    if state.get("summary"):
        print_summary(state["summary"])


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_run(args) -> None:
    graph, settings, client = _graph(args)
    thread_id = args.thread_id or f"run-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    LAST_THREAD_FILE.write_text(thread_id)
    print(f"LLM: {client.name} | thread_id: {thread_id} | review: conf<{settings.review_confidence_threshold}, "
          f"amount>Rs.{settings.spend_review_threshold_inr:,.0f}")
    config = _config(thread_id, settings)
    if graph.get_state(config).values:
        sys.exit(f"thread {thread_id} already exists; use `resume` or a new --thread-id")
    drive(graph, config, settings, {"source_path": args.emails, "as_of": args.as_of},
          interactive=args.interactive, auto_approve=args.auto_approve)


def _thread(args) -> str:
    if args.thread_id:
        return args.thread_id
    if LAST_THREAD_FILE.exists():
        return LAST_THREAD_FILE.read_text().strip()
    sys.exit("no --thread-id given and no previous run found")


def cmd_resume(args) -> None:
    graph, settings, _ = _graph(args)
    config = _config(_thread(args), settings)
    item = _pending_interrupt(graph, config)
    if item is None:
        print("Nothing to resume: this thread has no pending review.")
        return report(graph.get_state(config).values)
    print_review(item)
    if args.interactive and not args.action:
        decision = _ask_reviewer(item)
    elif args.action:
        try:
            decision = _decision_from_pairs(args.action, args.set or [], args.note)
        except (ValueError, ValidationError) as exc:
            sys.exit(f"invalid decision, nothing was resumed: {exc}")
    else:
        sys.exit("pass --action approve|reject|correct or --interactive")
    print(f"\nResuming with: {json.dumps(decision, ensure_ascii=False)}")
    graph.invoke(Command(resume=decision), config)
    item = _pending_interrupt(graph, config)
    if item is not None and (args.interactive or args.auto_approve):
        drive_resume(graph, config, settings, args)
    elif item is not None:
        print_review(item)
        print(f"\n{item['pending_count']} review(s) still pending. Run `python main.py resume "
              f"--thread-id {config['configurable']['thread_id']} --action ...` again.")
    else:
        report(graph.get_state(config).values)


def drive_resume(graph, config, settings, args) -> None:
    while (item := _pending_interrupt(graph, config)) is not None:
        print_review(item)
        decision = {"action": "approve", "note": "auto-approved"} if args.auto_approve else _ask_reviewer(item)
        graph.invoke(Command(resume=decision), config)
    report(graph.get_state(config).values)


def cmd_status(args) -> None:
    graph, settings, _ = _graph(args)
    thread_id = _thread(args)
    config = _config(thread_id, settings)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        sys.exit(f"unknown thread {thread_id}")
    state = snapshot.values
    print(f"thread_id={thread_id} next={list(snapshot.next)} "
          f"decisions={len(state.get('review_decisions', []))}/{len(state.get('review_queue', []))}")
    item = _pending_interrupt(graph, config)
    if item:
        print_review(item)
    elif state.get("summary"):
        print_summary(state["summary"])


def cmd_summary(args) -> None:
    """Answer e.g. "what did I spend in October?" from a finished run's checkpoint."""
    graph, settings, _ = _graph(args)
    state = graph.get_state(_config(_thread(args), settings)).values
    if not state.get("final_transactions") and not state.get("summary"):
        sys.exit("this run has not finished yet - resolve pending reviews first")
    txns = state.get("final_transactions", [])
    if args.category:
        txns = [t for t in txns if args.category.lower() in
                {(t.get("category") or "").lower(), (t.get("email_category") or "").lower()}]
    print_summary(summarize(state["emails"], state["final_classifications"], txns,
                            as_of=state.get("as_of"), month=args.month))


def cmd_diagram(args) -> None:
    from llm import RuleBasedClient

    mermaid = render_mermaid(build_graph(client=RuleBasedClient()), args.out)
    print(mermaid)
    print(f"\nwritten to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LangGraph email triage and spend tracker")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", help="SQLite checkpoint file (default: CHECKPOINT_DB or checkpoints.sqlite)")
    common.add_argument("-v", "--verbose", action="store_true", help="log retries and fallbacks")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("run", parents=[common], help="process emails.json")
    p.add_argument("--emails", default="emails.json")
    p.add_argument("--thread-id")
    p.add_argument("--as-of", help="YYYY-MM-DD reference date for month-to-date (default: newest email)")
    p.add_argument("--interactive", action="store_true", help="answer reviews in the terminal")
    p.add_argument("--auto-approve", action="store_true", help="approve all reviews (demo)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", parents=[common], help="answer the pending review and continue")
    p.add_argument("--thread-id")
    p.add_argument("--action", choices=["approve", "reject", "correct"])
    p.add_argument("--set", action="append", metavar="FIELD=VALUE",
                   help="correction, repeatable: category, amount, currency, merchant, date, spend_category, payment_method")
    p.add_argument("--note")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--auto-approve", action="store_true", help="approve the remaining reviews after this one")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("status", parents=[common], help="show the state of a run")
    p.add_argument("--thread-id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("summary", parents=[common], help="spend summary of a finished run, optionally for a month/category")
    p.add_argument("--thread-id")
    p.add_argument("--month", help="YYYY-MM, e.g. 2026-10")
    p.add_argument("--category", help="spend category or email category, e.g. Subscriptions")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("diagram", parents=[common], help="write the Mermaid graph diagram")
    p.add_argument("--out", default="diagrams/graph.mmd")
    p.set_defaults(func=cmd_diagram)

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]  # bare `python main.py [opts]` == `python main.py run [opts]`
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
