"""
Run with: streamlit run frontend/app_streamlit.py
(run from the project root so imports resolve)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from agents.graph import run_audit

st.set_page_config(page_title="AgentAudit", layout="wide")
st.title("AgentAudit — Causal Failure-Analysis Dashboard")
st.caption("Stress-tests a target agent and distinguishes causal failure drivers from spurious correlations.")

n_per_category = st.slider("Test cases per category", 20, 300, 150, step=10)

if st.button("Run Audit"):
    with st.spinner("Generating test cases, executing, judging, and running causal analysis..."):
        state = run_audit(n_per_category=n_per_category)

    st.metric("Overall Failure Rate", f"{state['overall_failure_rate']:.1%}")

    st.subheader("Naive Correlation vs. Causally-Adjusted Effect")
    rows = []
    for factor, d in state["causal_results"]["factor_analysis"].items():
        rows.append({
            "Factor": factor,
            "Naive Correlation": d["naive_correlation"],
            "Adjusted Coefficient": d["adjusted"]["coefficient"],
            "p-value": d["adjusted"]["p_value"],
            "Adjusted For": ", ".join(d["adjusted"]["adjustment_set"]) or "(nothing)",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Causal Graph (analyst-specified)")
    for a, b in state["causal_results"]["graph_edges"]:
        st.write(f"`{a}` → `{b}`")

    st.subheader("Root-Cause Report")
    st.text(state["report"])

    st.subheader("Sample Traces")
    df = pd.DataFrame(state["scored"])
    st.dataframe(df.head(30), use_container_width=True)
else:
    st.info("Click 'Run Audit' to generate test cases and run the full pipeline.")
