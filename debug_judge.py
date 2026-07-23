"""
debug_judge.py

Directly scores a known-good and known-bad agent response and prints the
judge's raw reply plus the parsed verdict. Use this whenever the judge's
aggregate numbers look suspicious.

Run: python debug_judge.py
"""
from execution.trace_schema import Trace
from judging.judge_ensemble import score_trace_llm

CASES = [
    ("GOOD - answers the question",
     "How do I reset my account password?",
     "Click 'Forgot password' on the login screen and follow the emailed link."),
    ("BAD - generic non-answer (dummy agent's actual failure mode)",
     "How do I reset my account password?",
     "Thanks for reaching out! Have a great day."),
    ("BAD - wrong tool / off-topic",
     "I was charged twice this month, please refund me.",
     "Here are some articles about our product features you might enjoy."),
]

for label, user_input, response in CASES:
    trace = Trace(
        case_id="debug", task_category="how_to", input_text=user_input,
        context_length=5, num_tools_available=5, retry_count=0, tool_used="kb",
        expected_tool="", latency_s=0.1, error=None, output_text=response,
    )
    result = score_trace_llm(trace)
    print(f"\n=== {label} ===")
    print(f"  request:  {user_input}")
    print(f"  response: {response}")
    print(f"  verdict:  {result}")
