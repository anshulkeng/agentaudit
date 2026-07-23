"""
inspect_run.py

Runs a real audit and prints every individual case's judgment - avoids the
PowerShell quoting headaches of doing this as an inline `python -c` one-
liner. Run from the repo root:

    python inspect_run.py
    python inspect_run.py --n 30 --endpoint http://localhost:8899
    python inspect_run.py --mode supportsense --n 15 --endpoint http://localhost:8001
"""
import argparse

from agents.graph import run_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="cases per category")
    parser.add_argument("--endpoint", default="http://localhost:8899")
    parser.add_argument("--mode", default="real", choices=["real", "ci", "supportsense"])
    args = parser.parse_args()

    state = run_audit(n_per_category=args.n, mode=args.mode, target_endpoint=args.endpoint)

    print("\n=== Per-case judgments ===")
    for c in state["scored"]:
        mark = "OK " if c["correct"] else "FAIL"
        print(f"[{mark}] {c['failure_mode']:32} | {c['input_text'][:60]}")
        if not c["correct"]:
            print(f"        response: {c.get('output_text', '')[:80]}")

    print("\n=== Report ===")
    print(state["report"])


if __name__ == "__main__":
    main()
