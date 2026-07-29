import pandas as pd


def build_feature_table(scored_traces: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(scored_traces)
    keep_cols = ["case_id", "task_category", "context_length",
                 "num_tools_available", "retry_count", "correct", "failure_mode"]

    for optional_col in ("category_confidence", "kb_confidence"):
        if optional_col in df.columns and df[optional_col].notna().any():
            keep_cols.append(optional_col)
    return df[keep_cols]
