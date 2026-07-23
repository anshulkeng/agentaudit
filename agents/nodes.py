"""
LangGraph node functions. Each takes and returns the shared pipeline state dict.
"""
from redteam.generator import generate_cases, generate_cases_llm
from redteam.dedup import dedup_cases, dedup_report
from execution.harness import run_batch
from execution.real_harness import run_batch_real
from execution.supportsense_harness import run_batch_supportsense
from judging.judge_ensemble import score_batch, score_batch_llm
from causal.feature_builder import build_feature_table
from causal.scm_analysis import analyze
from agents.report_agent import write_report


def redteam_node(state: dict) -> dict:
    """mode='synthetic' (default): templated cases against the synthetic
    target agent, used by eval/run_eval.py for deterministic ground-truth
    validation. mode='real', 'ci', or 'supportsense': LLM-generated cases +
    embedding dedup, for auditing an actual target agent."""
    mode = state.get("mode", "synthetic")
    if mode in ("real", "supportsense"):
        raw_cases = generate_cases_llm(n_per_category=state.get("n_per_category", 20))
        state["cases"] = dedup_cases(raw_cases)
        state["dedup_report"] = dedup_report(raw_cases, state["cases"])
    elif mode == "ci":
        # Templated cases (fast, no model download) but still exercised
        # against a REAL target endpoint -- see executor_node. Keeps CI runs
        # fast/deterministic while still catching real agent regressions.
        state["cases"] = generate_cases(n_per_category=state.get("n_per_category", 15))
    else:
        state["cases"] = generate_cases(n_per_category=state.get("n_per_category", 40))
    return state


def executor_node(state: dict) -> dict:
    """mode='synthetic' (default): the in-process synthetic target agent
    with known causal ground truth. mode='real'/'ci': a generic real HTTP
    call to state['target_endpoint'] via execution/real_harness.py.
    mode='supportsense': the SupportSense-specific submit/poll adapter."""
    mode = state.get("mode", "synthetic")
    if mode == "supportsense":
        endpoint = state.get("target_endpoint")
        if not endpoint:
            raise ValueError(
                "mode='supportsense' requires state['target_endpoint'] -- SupportSense's "
                "base URL, e.g. http://localhost:8001. Pass it via run_audit(target_endpoint=...)."
            )
        state["traces"] = run_batch_supportsense(
            state["cases"], endpoint=endpoint, timeout=state.get("timeout", 30.0),
        )
    elif mode in ("real", "ci"):
        endpoint = state.get("target_endpoint")
        if not endpoint:
            raise ValueError(
                "mode='real'/'ci' requires state['target_endpoint'] -- the URL of the "
                "agent under test. Pass it via run_audit(target_endpoint=...)."
            )
        state["traces"] = run_batch_real(
            state["cases"], endpoint=endpoint,
            timeout=state.get("timeout", 30.0),
            max_retries=state.get("max_retries", 1),
        )
    else:
        state["traces"] = run_batch(state["cases"])
    return state


def judge_node(state: dict) -> dict:
    """mode='real', 'ci', or 'supportsense' uses the generative LLM-judge
    (no ground-truth expected_tool available against a real target agent).
    mode='synthetic' (default) uses the rule-based judge validated in
    eval/run_eval.py."""
    mode = state.get("mode", "synthetic")
    if mode in ("real", "ci", "supportsense"):
        state["scored"] = score_batch_llm(state["traces"])
    else:
        state["scored"] = score_batch(state["traces"])
    return state


def causal_node(state: dict) -> dict:
    df = build_feature_table(state["scored"])
    state["feature_table"] = df
    state["causal_results"] = analyze(df)
    state["overall_failure_rate"] = 1 - df["correct"].mean()
    return state


def report_node(state: dict) -> dict:
    state["report"] = write_report(state["causal_results"], state["overall_failure_rate"])
    return state
