"""
This is the "agent under test." In a real deployment you'd replace `run_case()`
with an HTTP call to whatever agent you're auditing (see harness.py). For this
runnable MVP we simulate a toy support agent with a DELIBERATELY KNOWN causal
structure, so we can validate that the causal engine (causal/scm_analysis.py)
correctly recovers the true causal story instead of being fooled by a spurious
correlation. This "test against a known generating process before trusting it
on real unlabeled data" step is standard practice for validating causal methods.

Ground truth causal story we bake in:
  - task_category causally determines num_tools_available (refund requests are
    given a narrower toolset in this toy agent).
  - num_tools_available causally determines failure probability (fewer tools ->
    more wrong-tool errors).
  - task_category ALSO independently makes conversations longer (context_length),
    but context_length itself has NO causal effect on failure.
  - retry_count is pure noise (a placebo variable with no effect on anything).

So: context_length will be CORRELATED with failure (because task_category
confounds both), but is NOT a CAUSE of failure. A naive correlation analysis
gets this wrong. A causal analysis that adjusts for task_category gets it right.

Uses numpy's Generator (PCG64) instead of the stdlib `random` module for the
random draws below. Reason: stdlib `random`'s seeded output for random() is
stable, but higher-level helpers built on it (choice, gauss, randint) use an
internal implementation that has changed between Python versions for
performance reasons - so the same seed can produce a different dataset on
Python 3.12 vs 3.14, for example. NumPy's Generator is explicitly documented
as bit-for-bit reproducible across versions, which is what set_seed() below
actually needs to guarantee for eval/run_eval.py to be reproducible on
whatever Python version runs it.
"""
import numpy as np

CATEGORIES = ["how_to", "bug_report", "refund_request"]

EXPECTED_TOOL = {
    "how_to": "kb_lookup",
    "bug_report": "bug_tracker_tool",
    "refund_request": "refund_tool",
}

WRONG_TOOL_POOL = ["kb_lookup", "bug_tracker_tool", "refund_tool", "escalate_human"]

_rng = np.random.default_rng()  # unseeded by default; call set_seed() for reproducible runs


def set_seed(seed: int) -> None:
    """Call this before generating cases if you want a reproducible run
    (eval/run_eval.py does this). Reproducible across machines/Python
    versions since it's backed by numpy's Generator, not stdlib random."""
    global _rng
    _rng = np.random.default_rng(seed)


def _num_tools_available(task_category: str) -> int:
    # Refund requests are (by design, in this toy agent) usually given a
    # narrower toolset -- but not with perfect consistency (e.g. rollout
    # inconsistencies, per-session config drift), which is realistic and
    # also avoids perfect collinearity with task_category in the regression.
    base = 2 if task_category == "refund_request" else 4
    if _rng.random() < 0.15:
        base += int(_rng.choice([-1, 1]))
    return max(1, base)


def _context_length(task_category: str) -> int:
    base = 30
    category_bump = 25 if task_category == "refund_request" else 0
    noise = _rng.normal(0, 8)
    return max(5, int(base + category_bump + noise))


def run_case(case: dict) -> dict:
    """Runs one red-team case against the toy target agent and returns a raw result dict.
    Swap this function's body for a real HTTP call to your actual target agent."""
    task_category = case["task_category"]
    num_tools = _num_tools_available(task_category)
    expected_tool = EXPECTED_TOOL[task_category]

    # Fewer available tools -> higher chance of picking the wrong one. This is
    # the ONE true causal driver of failure in this toy world.
    if num_tools <= 2:
        correct_prob = 0.50
    elif num_tools == 3:
        correct_prob = 0.75
    else:
        correct_prob = 0.92
    if _rng.random() < correct_prob:
        tool_used = expected_tool
    else:
        wrong_options = [t for t in WRONG_TOOL_POOL if t != expected_tool]
        tool_used = str(_rng.choice(wrong_options))

    return {
        "case_id": case["case_id"],
        "task_category": task_category,
        "input_text": case["input_text"],
        "context_length": _context_length(task_category),
        "num_tools_available": num_tools,
        "retry_count": max(0, int(_rng.normal(1, 1))),  # placebo: no causal effect on anything
        "tool_used": tool_used,
        "expected_tool": expected_tool,
        "latency_s": round(float(_rng.uniform(0.2, 1.8)), 3),
        "error": None,
    }
