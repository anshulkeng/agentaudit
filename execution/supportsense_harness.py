"""
execution/supportsense_harness.py

Adapter for auditing SupportSense as a real target agent. SupportSense's API
doesn't match the generic single-call {"input": "..."} shape real_harness.py
assumes -- it's a two-step submit-then-poll API with a structured request
body (true_transcript, audio_quality, has_screenshot, etc). This module
bridges that gap: maps an AgentAudit-generated test case into SupportSense's
expected schema, POSTs it, GETs the result, and returns it in the same dict
shape real_harness.py's run_case_real() produces -- so the rest of the
pipeline (Trace, judge, causal engine) doesn't need to know or care which
target agent it's actually talking to.

AgentAudit's generated cases are plain text messages, so the audio/
screenshot fields are stubbed to their "not applicable" defaults -- this
only exercises SupportSense's text-triage path, not its multimodal
(audio/OCR) pipeline. That's a real, worth-noting scope limit if you write
this up: the audit covers text-in/text-out behavior, not the full
multimodal surface.
"""
import time
import requests
from execution.trace_schema import Trace

DEFAULT_NUM_TOOLS_AVAILABLE = 1  # SupportSense doesn't expose a toolset-size concept; treated as fixed


def _estimate_context_length(input_text: str) -> int:
    return len(input_text.split())


def run_case_supportsense(case: dict, endpoint: str, timeout: float = 30.0,
                           poll_attempts: int = 5, poll_delay: float = 0.5) -> dict:
    """Submits one test case to SupportSense and fetches its result via the
    two-step POST /case -> GET /case/{id} flow. Polls a few times since the
    POST response's status field could in principle mean "queued" rather
    than "done" even though current behavior looks synchronous -- cheap
    insurance against that assumption being wrong on a different run."""
    start = time.time()
    error = None
    output_text = ""
    tool_used = "unknown"
    category_confidence = None
    kb_confidence = None

    body = {
        "true_transcript": case["input_text"],
        "audio_quality": "clear",
        "has_screenshot": False,
        "true_screenshot_text": "",
        "screenshot_quality": "clear",
        "channel": "chat",
    }

    try:
        submit_resp = requests.post(f"{endpoint}/case", json=body, timeout=timeout)
        submit_resp.raise_for_status()
        case_id = submit_resp.json()["case_id"]

        result = None
        for _ in range(poll_attempts):
            get_resp = requests.get(f"{endpoint}/case/{case_id}", timeout=timeout)
            get_resp.raise_for_status()
            result = get_resp.json()
            if result.get("reply") is not None:
                break
            time.sleep(poll_delay)

        if result is None or result.get("reply") is None:
            error = "SupportSense never returned a reply after polling"
        else:
            output_text = result.get("reply", "")
            tool_used = result.get("category", "unknown")
            category_confidence = result.get("category_confidence")
            kb_match = result.get("kb_match") or {}
            kb_confidence = kb_match.get("confidence")

    except Exception as e:
        error = str(e)

    return {
        "case_id": case["case_id"],
        "task_category": case["task_category"],
        "input_text": case["input_text"],
        "context_length": _estimate_context_length(case["input_text"]),
        "num_tools_available": DEFAULT_NUM_TOOLS_AVAILABLE,
        "retry_count": 0,
        "tool_used": tool_used,
        "expected_tool": "",
        "latency_s": round(time.time() - start, 3),
        "error": error,
        "output_text": output_text,
        "category_confidence": category_confidence,
        "kb_confidence": kb_confidence,
    }


def run_batch_supportsense(cases: list[dict], endpoint: str, timeout: float = 30.0) -> list[Trace]:
    traces = []
    for case in cases:
        result = run_case_supportsense(case, endpoint=endpoint, timeout=timeout)
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
            category_confidence=result["category_confidence"],
            kb_confidence=result["kb_confidence"],
        ))
    return traces
