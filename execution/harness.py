"""
Runs a batch of test cases against the target agent and returns structured Trace
objects. In a real deployment, `_call_target()` makes an HTTP request to your
actual agent's API; here it calls the local synthetic target_agent for a fully
runnable demo.
"""
import time
from execution.trace_schema import Trace
from execution.target_agent import run_case as _call_target


def run_batch(cases: list[dict]) -> list[Trace]:
    traces = []
    for case in cases:
        start = time.time()
        try:
            result = _call_target(case)
            error = result.get("error")
        except Exception as e:
            result = {**case, "context_length": 0, "num_tools_available": 0,
                      "retry_count": 0, "tool_used": "NONE", "expected_tool": "NONE"}
            error = str(e)

        traces.append(Trace(
            case_id=result["case_id"],
            task_category=result["task_category"],
            input_text=result["input_text"],
            context_length=result["context_length"],
            num_tools_available=result["num_tools_available"],
            retry_count=result["retry_count"],
            tool_used=result["tool_used"],
            expected_tool=result["expected_tool"],
            latency_s=result.get("latency_s", time.time() - start),
            error=error,
            output_text=result.get("output_text"),
        ))
    return traces
