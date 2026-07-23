"""
Entry point for the CI reliability gate. Runs a fast audit ("ci" mode: templated cases, no HF text-generation download,
still hits the real target agent + real zero-shot judge) and fails the
build if the observed failure rate exceeds a configured threshold.

Environment variables:
  TARGET_AGENT_ENDPOINT   required -- URL of the agent under test
  MAX_FAILURE_RATE        optional, default 0.15 -- fail the build above this
  N_PER_CATEGORY          optional, default 15
"""
import os
import sys

from agents.graph import run_audit


def main() -> int:
    endpoint = os.environ.get("TARGET_AGENT_ENDPOINT")
    if not endpoint:
        print("ERROR: TARGET_AGENT_ENDPOINT is not set. The CI gate has nothing to audit.")
        return 2

    max_failure_rate = float(os.environ.get("MAX_FAILURE_RATE", "0.15"))
    n_per_category = int(os.environ.get("N_PER_CATEGORY", "15"))

    print(f"Auditing {endpoint} ({n_per_category} cases/category, "
          f"threshold={max_failure_rate:.0%})...")

    state = run_audit(n_per_category=n_per_category, mode="ci", target_endpoint=endpoint)
    failure_rate = state["overall_failure_rate"]

    print(f"\n{state['report']}\n")
    print(f"Overall failure rate: {failure_rate:.1%} (threshold: {max_failure_rate:.0%})")

    if failure_rate > max_failure_rate:
        print("FAIL: failure rate exceeds threshold -- blocking merge.")
        return 1

    print("PASS: failure rate within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
