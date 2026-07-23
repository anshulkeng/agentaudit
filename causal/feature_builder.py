"""
Turns a list of scored trace dicts into a tabular pandas DataFrame ready for
causal analysis.
"""
import pandas as pd


def build_feature_table(scored_traces: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(scored_traces)
    keep_cols = ["case_id", "task_category", "context_length",
                 "num_tools_available", "retry_count", "correct", "failure_mode"]
    # category_confidence/kb_confidence only exist for targets that report
    # them (currently SupportSense) -- include if present, don't require it.
    for optional_col in ("category_confidence", "kb_confidence"):
        if optional_col in df.columns:
            keep_cols.append(optional_col)
    return df[keep_cols]
