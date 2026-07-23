"""
Wires the five agents into one LangGraph pipeline:
Red-Team -> Executor -> Judge -> Causal Analyst -> Report
"""
from langgraph.graph import StateGraph, END
from agents.nodes import redteam_node, executor_node, judge_node, causal_node, report_node


def build_graph():
    graph = StateGraph(dict)
    graph.add_node("redteam", redteam_node)
    graph.add_node("execute", executor_node)
    graph.add_node("judge", judge_node)
    graph.add_node("causal_analysis", causal_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("redteam")
    graph.add_edge("redteam", "execute")
    graph.add_edge("execute", "judge")
    graph.add_edge("judge", "causal_analysis")
    graph.add_edge("causal_analysis", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_audit(n_per_category: int = 40, mode: str = "synthetic",
              target_endpoint: str | None = None, timeout: float = 30.0,
              max_retries: int = 1) -> dict:
    """mode='synthetic' (default): templated generator + rule-based judge
    against the synthetic target agent -- deterministic, used for the
    ground-truth validation in eval/run_eval.py.
    mode='real': LLM generator + embedding dedup + zero-shot LLM judge
    against a real target_endpoint -- what you'd use for a full manual audit.
    mode='ci': fast templated generator (no model download) + zero-shot LLM
    judge against a real target_endpoint -- what the CI gate uses, so a PR
    check doesn't take minutes downloading a text-generation model on every
    run.
    real/ci modes require target_endpoint (the URL of the agent under test)."""
    app = build_graph()
    return app.invoke({
        "n_per_category": n_per_category,
        "mode": mode,
        "target_endpoint": target_endpoint,
        "timeout": timeout,
        "max_retries": max_retries,
    })
