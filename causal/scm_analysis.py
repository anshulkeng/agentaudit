import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError


def build_causal_graph() -> nx.DiGraph:
    """Explicit, human-justified DAG -- not auto-discovered from data.
    Edges reflect the analyst's domain understanding of the target agent's
    architecture, not a pattern mined from the dataset itself."""
    g = nx.DiGraph()
    g.add_edges_from([
        ("task_category", "num_tools_available"),
        ("task_category", "context_length"),
        ("num_tools_available", "failure"),
        # SupportSense-specific factors (see execution/supportsense_harness.py).
        # Hypothesis: certain task categories are more likely to hit a
        # low-confidence classification or KB match, and low confidence in
        # turn causally drives failure (the agent picks a wrong/generic
        # response when it isn't sure). All-None for targets that don't
        # report these (synthetic agent, dummy agent) -- the zero-variance
        # guard in adjusted_effect() reports "no variation" for those
        # instead of erroring.
        ("task_category", "category_confidence"),
        ("category_confidence", "failure"),
        ("task_category", "kb_confidence"),
        ("kb_confidence", "failure"),
        # context_length and retry_count have NO edge into failure --
        # the graph asserts they are not causes. The analysis below tests
        # whether the data agrees.
    ])
    g.add_nodes_from(["context_length", "retry_count", "failure"])
    return g


def get_adjustment_set(graph: nx.DiGraph, factor: str) -> set:
    """Sufficient backdoor adjustment set: parents of `factor` that are not
    also descendants of `factor` (which would introduce bias, not remove it)."""
    if factor not in graph.nodes:
        return set()
    parents = set(graph.predecessors(factor))
    descendants = nx.descendants(graph, factor)
    return parents - descendants


def naive_correlation(df: pd.DataFrame, factor: str):
    """Returns None (not NaN) for a constant column instead of pandas' NaN,
    so downstream formatting doesn't have to special-case NaN vs. a real
    float -- constant columns happen in practice (e.g. a real target agent
    with a fixed toolset reports the same num_tools_available every time)."""
    if df[factor].nunique() <= 1:
        return None
    value = df[[factor, "failure_int"]].corr().iloc[0, 1]
    return None if pd.isna(value) else float(value)


def adjusted_effect(df: pd.DataFrame, factor: str, adjustment_set: set) -> dict:
    """Logistic regression of failure on `factor`, controlling for the
    adjustment set. Categorical adjustment variables are handled with C().
    Iterates over sorted(adjustment_set), not the raw set -- Python's set
    iteration order is randomized per-process (hash randomization), which
    would otherwise make the formula string, and depending on the solver,
    potentially the fit itself, non-reproducible between runs.

    Degrades gracefully instead of crashing when the regression can't be
    fit -- this happens for real, unremarkable reasons: a factor with zero
    variance in the current sample (e.g. a target agent with a fixed
    toolset size), or too few cases for the number of parameters causing
    (quasi-)perfect separation. Either one is a genuine "can't say" result,
    not a bug, and shouldn't take down the whole audit.
    """
    if df[factor].nunique() <= 1:
        return {
            "coefficient": None,
            "p_value": None,
            "adjustment_set": sorted(adjustment_set),
            "note": f"'{factor}' has no variation in this sample (constant value) -- "
                    f"can't estimate an effect for a factor that never changes. "
                    f"Common with a real target agent that has a fixed config "
                    f"(e.g. a fixed toolset size) rather than a bug.",
        }

    terms = [factor]
    for var in sorted(adjustment_set):
        if df[var].dtype == object:
            terms.append(f"C({var})")
        else:
            terms.append(var)
    formula = "failure_int ~ " + " + ".join(terms)

    try:
        model = smf.logit(formula, data=df).fit(disp=0)
        coefficient = model.params.get(factor)
        p_value = model.pvalues.get(factor)
        if coefficient is None or pd.isna(coefficient) or pd.isna(p_value):
            raise PerfectSeparationError("fit converged but produced a non-finite estimate")
        return {
            "coefficient": round(float(coefficient), 4),
            "p_value": round(float(p_value), 4),
            "adjustment_set": sorted(adjustment_set),
        }
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as e:
        return {
            "coefficient": None,
            "p_value": None,
            "adjustment_set": sorted(adjustment_set),
            "note": f"Could not fit a stable estimate ({type(e).__name__}) -- most likely "
                    f"too few cases for the number of parameters, causing (quasi-)perfect "
                    f"separation. Increase n_per_category and re-run before drawing any "
                    f"conclusion about '{factor}'.",
        }


def analyze(df: pd.DataFrame) -> dict:
    """Runs naive vs. causally-adjusted analysis for every candidate factor
    (every column except the outcome and identifiers) and flags where they diverge."""
    df = df.copy()
    df["failure_int"] = (~df["correct"]).astype(int)

    graph = build_causal_graph()
    # Base factors every target produces; category_confidence/kb_confidence
    # only exist for targets that report them (currently SupportSense) --
    # only analyze them when the column is actually present, so other
    # targets' reports aren't cluttered with a factor that's never there.
    candidate_factors = ["context_length", "num_tools_available", "retry_count"]
    for optional_factor in ("category_confidence", "kb_confidence"):
        if optional_factor in df.columns:
            candidate_factors.append(optional_factor)

    # Only report graph edges that are actually relevant to what got
    # analyzed this run -- otherwise a synthetic-mode report ends up
    # listing category_confidence/kb_confidence edges even though those
    # factors were never in candidate_factors for this run.
    relevant_nodes = {"task_category", "failure"} | set(candidate_factors)
    relevant_edges = [(a, b) for a, b in graph.edges if a in relevant_nodes and b in relevant_nodes]

    if df["failure_int"].nunique() <= 1:
        # Every case was judged the same way (all failures, or all correct).
        # There's nothing to explain statistically -- no factor can predict
        # an outcome that never varies, regardless of how much the factor
        # itself varies. This is a real, useful finding to surface (e.g. "the
        # judge is flagging 100% of cases as failures, go check the judge or
        # the target agent"), not something to silently paper over by
        # fitting a degenerate model anyway.
        outcome_value = "failure" if df["failure_int"].iloc[0] == 1 else "success"
        note = (
            f"Every case in this sample came back as '{outcome_value}' -- there's no "
            f"variation in the outcome to explain, so no factor's effect can be "
            f"estimated here. If this is unexpected, check judge calibration "
            f"(eval/judge_calibration.py) before trusting the underlying failure rate."
        )
        results = {
            factor: {
                "naive_correlation": None,
                "adjusted": {"coefficient": None, "p_value": None, "adjustment_set": [], "note": note},
            }
            for factor in candidate_factors
        }
        return {"graph_edges": relevant_edges, "factor_analysis": results}

    results = {}
    for factor in candidate_factors:
        naive = naive_correlation(df, factor)
        adj_set = get_adjustment_set(graph, factor)
        adjusted = adjusted_effect(df, factor, adj_set)
        results[factor] = {
            "naive_correlation": None if naive is None else round(naive, 4),
            "adjusted": adjusted,
        }

    return {
        "graph_edges": relevant_edges,
        "factor_analysis": results,
    }
