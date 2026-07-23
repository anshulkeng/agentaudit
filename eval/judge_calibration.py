"""
eval/run_eval.py validates the CAUSAL engine against a synthetic ground
truth. This script validates the JUDGE itself (score_trace_llm) against
real, hand-labeled examples - the thing you actually need once you're
auditing a real agent where you don't already know the right answer.

How to use this:
  1. Run your real target agent against a handful of test cases where you
     ALREADY know what should happen - some genuinely correct, and some
     where you deliberately trigger a specific failure (force a wrong tool,
     omit context, etc).
  2. Fill in eval/injected_failures_real.jsonl with rows like:
       {"input_text": "...", "output_text": "...", "expected_failure_mode": "wrong_tool_used"}
     (use "no_failure" for the genuinely-correct cases)
  3. Run: python -m eval.judge_calibration
  4. Put the printed recall/precision numbers in the README - an honest
     number beats an assumption, and don't trust score_trace_llm on
     unlabeled real-agent output until you've done this.
"""
import json
import sys
from collections import defaultdict

from execution.trace_schema import Trace
from judging.judge_ensemble import score_trace_llm


def load_labeled_cases(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def calibrate(path: str = "eval/injected_failures_real.jsonl") -> dict:
    try:
        labeled = load_labeled_cases(path)
    except FileNotFoundError:
        print(f"No labeled cases found at {path}.")
        print("Create it first -- see the module docstring in eval/judge_calibration.py "
              "for the expected format. Skipping calibration.")
        return {}

    per_label_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    correct = 0

    for row in labeled:
        trace = Trace(
            case_id=row.get("case_id", "labeled"),
            task_category=row.get("task_category", "unknown"),
            input_text=row["input_text"],
            context_length=0,
            num_tools_available=0,
            retry_count=0,
            tool_used=row.get("tool_used", "unknown"),
            expected_tool="",
            latency_s=0.0,
            error=row.get("error"),
            output_text=row["output_text"],
        )
        predicted = score_trace_llm(trace)
        expected_mode = row["expected_failure_mode"]
        predicted_mode = predicted["failure_mode"]

        if predicted_mode == expected_mode:
            correct += 1
            per_label_counts[expected_mode]["tp"] += 1
        else:
            per_label_counts[expected_mode]["fn"] += 1
            per_label_counts[predicted_mode]["fp"] += 1

    overall_recall = correct / len(labeled) if labeled else 0.0

    print(f"Judge calibration against {len(labeled)} hand-labeled cases:")
    print(f"Overall detection recall: {overall_recall:.1%}\n")
    print(f"{'label':<35}{'precision':>10}{'recall':>10}")
    per_label_results = {}
    for label, counts in per_label_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        per_label_results[label] = {"precision": round(precision, 3), "recall": round(recall, 3)}
        print(f"{label:<35}{precision:>10.1%}{recall:>10.1%}")

    return {"overall_recall": round(overall_recall, 4), "per_label": per_label_results}


if __name__ == "__main__":
    result = calibrate()
    if not result:
        sys.exit(1)
