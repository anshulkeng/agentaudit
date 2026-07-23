"""
Defines the structured trace format that every run against the target agent
produces. Every downstream component (judge, causal analysis) reads this format.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Trace:
    case_id: str
    task_category: str          # e.g. "refund_request", "how_to", "bug_report"
    input_text: str
    context_length: int         # tokens/words in the conversation so far
    num_tools_available: int    # how many tools the agent could choose from
    retry_count: int            # how many times the agent retried before answering
    tool_used: str               # which tool the agent actually called
    expected_tool: str           # ground-truth correct tool (known because this is synthetic)
    latency_s: float
    error: Optional[str] = None
    output_text: Optional[str] = None  # the agent's final text response, if captured.
                                        # Required for the real LLM judge (judging/judge_ensemble.py
                                        # score_trace_llm), since a zero-shot classifier needs actual
                                        # text to judge. Not used by the synthetic ground-truth path.
    category_confidence: Optional[float] = None  # target-agent-reported confidence in its own
    kb_confidence: Optional[float] = None         # classification/retrieval, if it reports one.
                                                   # None for targets that don't expose this (synthetic
                                                   # agent, dummy_target_agent, generic real_harness) --
                                                   # the causal engine already handles all-None columns
                                                   # gracefully (reports "no variation", doesn't crash).

    @property
    def correct(self) -> bool:
        return self.error is None and self.tool_used == self.expected_tool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["correct"] = self.correct
        return d
