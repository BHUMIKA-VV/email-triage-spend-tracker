"""LangGraph workflow definition.

    START -> ingest --(Send x N)--> process_email --> deduplicate -> review_gate
                                     (subgraph)                          |
                                                     +-------------------+
                                                     v                   v
                                   human_review (interrupt, loops) -> aggregate -> END

process_email (one instance per email, run in parallel):
    START -> classify --(route_after_classify)--> extract_transaction -> END
                                     \\--------------------------------> END

Why Send + subgraph:
* Classification/extraction of different emails are independent, so they run in
  parallel (bounded by `max_concurrency`). On 20 emails this is several times
  faster than a sequential loop.
* Putting classify -> extract in a subgraph lets routing be a real conditional
  edge evaluated on the *per-email* state.
* The human gate runs AFTER fan-in and dedup, sequentially, one item per
  interrupt. Interrupts therefore never happen inside parallel branches, the
  reviewer never sees a duplicate, and each decision is its own checkpoint.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import Settings, load_settings
from llm import LLMClient, build_llm_client
from nodes import dispatch_emails, make_nodes, route_after_classify, route_review
from state import EmailState, TriageState


def build_email_subgraph(nodes):
    sub = StateGraph(EmailState)
    sub.add_node("classify", nodes.classify)
    sub.add_node("extract_transaction", nodes.extract_transaction)
    sub.add_edge(START, "classify")
    sub.add_conditional_edges("classify", route_after_classify, ["extract_transaction", END])
    sub.add_edge("extract_transaction", END)
    return sub.compile()


def build_graph(
    client: Optional[LLMClient] = None,
    settings: Optional[Settings] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    settings = settings or load_settings()
    client = client or build_llm_client(settings)
    nodes = make_nodes(client, settings)

    g = StateGraph(TriageState)
    g.add_node("ingest", nodes.ingest)
    g.add_node("process_email", build_email_subgraph(nodes))
    g.add_node("deduplicate", nodes.deduplicate)
    g.add_node("review_gate", nodes.review_gate)
    g.add_node("human_review", nodes.human_review)
    g.add_node("aggregate", nodes.aggregate)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", dispatch_emails, ["process_email", "deduplicate"])
    g.add_edge("process_email", "deduplicate")
    g.add_edge("deduplicate", "review_gate")
    g.add_conditional_edges("review_gate", route_review, ["human_review", "aggregate"])
    g.add_conditional_edges("human_review", route_review, ["human_review", "aggregate"])
    g.add_edge("aggregate", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: str) -> SqliteSaver:
    """Durable checkpointer: a run paused at the review gate survives process exit."""
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def render_mermaid(graph, path: str | Path = "diagrams/graph.mmd") -> str:
    mermaid = graph.get_graph(xray=True).draw_mermaid()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mermaid, encoding="utf-8")
    return mermaid


if __name__ == "__main__":
    from llm import RuleBasedClient

    print(render_mermaid(build_graph(client=RuleBasedClient())))
