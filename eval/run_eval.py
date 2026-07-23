"""
Validates two things against the KNOWN ground truth baked into target_agent.py:
  1. The judge correctly detects failures (it should be ~100% here since it's
     rule-based against known expected_tool -- the real validation target once
     you swap in an LLM judge for unlabeled data is THIS number).
  2. The causal engine correctly identifies num_tools_available as the true
     causal driver and correctly identifies context_length as spurious
     (confounded, not causal).

PASS/FAIL below is based on statistical significance (p < 0.05) plus expected
sign, not an arbitrary coefficient-magnitude cutoff -- a coefficient can be
"real" and still smaller than some made-up threshold, especially with a
moderate sample size. If a run comes out borderline, that's the eval
correctly reporting that n_per_category is a bit low for this effect size,
not a bug -- bump n_per_category if you want more consistent significance.
"""
from agents.graph import run_audit
from execution.target_agent import set_seed

SIGNIFICANCE = 0.05


def _is_significant(p_value: float) -> bool:
    return p_value < SIGNIFICANCE


def run():
    set_seed(42)  # numpy-backed, reproducible across Python versions -
                  # see execution/target_agent.py's module docstring for why
                  # stdlib random.seed() couldn't guarantee that
    state = run_audit(n_per_category=400)
    causal = state["causal_results"]["factor_analysis"]

    print(f"Overall failure rate: {state['overall_failure_rate']:.1%}\n")

    ctx = causal["context_length"]
    tools = causal["num_tools_available"]
    retry = causal["retry_count"]

    print("=== Ground-truth check ===")
    print("context_length   (TRUE causal effect = NONE, confounded by task_category)")
    print(f"   naive correlation    = {ctx['naive_correlation']:+.3f}")
    print(f"   adjusted coefficient = {ctx['adjusted']['coefficient']:+.3f}  (p={ctx['adjusted']['p_value']})")
    print("   PASS: adjustment reveals a non-significant effect"
          if not _is_significant(ctx["adjusted"]["p_value"])
          else "   FAIL: adjustment did not reveal spuriousness")

    print("\nnum_tools_available (TRUE causal effect = REAL)")
    print(f"   naive correlation    = {tools['naive_correlation']:+.3f}")
    print(f"   adjusted coefficient = {tools['adjusted']['coefficient']:+.3f}  (p={tools['adjusted']['p_value']})")
    print("   PASS: effect persists and is significant"
          if _is_significant(tools["adjusted"]["p_value"]) and tools["adjusted"]["coefficient"] < 0
          else "   FAIL: expected a real, negative, significant effect here")

    print("\nretry_count (TRUE causal effect = NONE, pure placebo)")
    print(f"   naive correlation    = {retry['naive_correlation']:+.3f}")
    print(f"   adjusted coefficient = {retry['adjusted']['coefficient']:+.3f}  (p={retry['adjusted']['p_value']})")
    print("   PASS: no significant effect detected"
          if not _is_significant(retry["adjusted"]["p_value"])
          else "   FAIL: unexpected significant effect on a placebo variable")

    print("\n--- Full Report ---")
    print(state["report"])


if __name__ == "__main__":
    run()
