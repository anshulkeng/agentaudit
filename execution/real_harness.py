"""
Runs test cases against an ACTUAL target agent over HTTP instead of the
synthetic in-process target_agent.py. Kept as a separate module rather than
editing harness.py in place, so the synthetic path (used by eval/run_eval.py)
never breaks while I'm iterating on real-agent instrumentation.

What this assumes about your target agent's API:
  - POST {endpoint} with JSON body {"input": "<text>"}
  - Response JSON has at least a text field, checked in this priority order:
    "output" -> "response" -> "answer" -> "text"
  - Optionally, if your agent exposes them, the response can include
    "tool_used", "num_tools_available", "retry_count" for richer causal
    features. If it doesn't, sensible defaults are used (see below) -
    just be upfront in your README about which fields are real vs.
    approximated.

Usage:
    from execution.real_harness import run_batch_real
    traces = run_batch_real(cases, endpoint="https://your-agent.example.com/chat")
"""
import time
import requests
from execution.trace_schema import Trace

# Adjust to match your actual agent's declared toolset size. This is a
# per-deployment constant in most agents (the tools it COULD call), not
# something observable per-request unless your agent logs it -- so unless
# your target logs it, treat it as a config value you set here, and say so
# in the README rather than silently guessing.
DEFAULT_NUM_TOOLS_AVAILABLE = 5

RESPONSE_TEXT_KEYS = ("output", "response", "answer", "text")


def _extract_output_text(response_json: dict) -> str:
    for key in RESPONSE_TEXT_KEYS:
        if key in response_json and response_json[key]:
            return str(response_json[key])
    return ""


def _estimate_context_length(input_text: str) -> int:
    """Word-count proxy for context length. Swap for a real tokenizer count
    (e.g. tiktoken) if you want token-accurate figures -- word count is a
    reasonable, dependency-free approximation for relative comparisons."""
    return len(input_text.split())


def run_case_real(case: dict, endpoint: str, timeout: float = 30.0,
                   max_retries: int = 0) -> dict:
    """Runs one test case against a real agent endpoint and returns a raw
    result dict compatible with Trace's fields. `expected_tool` is left as
    an empty string since real audits have no ground truth -- correctness
    is judged from output_text by judging/judge_ensemble.py's
    score_trace_llm, not by comparing tool names."""
    start = time.time()
    retries_used = 0
    error = None
    output_text = ""
    tool_used = "unknown"

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(endpoint, json={"input": case["input_text"]}, timeout=timeout)
            resp.raise_for_status()
            body = resp.json()
            output_text = _extract_output_text(body)
            tool_used = str(body.get("tool_used", "unknown"))
            error = None
            break
        except Exception as e:
            error = str(e)
            retries_used = attempt
            if attempt < max_retries:
                continue

    return {
        "case_id": case["case_id"],
        "task_category": case["task_category"],
        "input_text": case["input_text"],
        "context_length": _estimate_context_length(case["input_text"]),
        "num_tools_available": DEFAULT_NUM_TOOLS_AVAILABLE,
        "retry_count": retries_used,
        "tool_used": tool_used,
        "expected_tool": "",  # no ground truth against a real agent
        "latency_s": round(time.time() - start, 3),
        "error": error,
        "output_text": output_text,
    }


def run_batch_real(cases: list[dict], endpoint: str, timeout: float = 30.0,
                    max_retries: int = 0) -> list[Trace]:
    traces = []
    for case in cases:
        result = run_case_real(case, endpoint=endpoint, timeout=timeout, max_retries=max_retries)
        traces.append(Trace(
            case_id=result["case_id"],
            task_category=result["task_category"],
            input_text=result["input_text"],
            context_length=result["context_length"],
            num_tools_available=result["num_tools_available"],
            retry_count=result["retry_count"],
            tool_used=result["tool_used"],
            expected_tool=result["expected_tool"],
            latency_s=result["latency_s"],
            error=result["error"],
            output_text=result["output_text"],
        ))
    return traces
