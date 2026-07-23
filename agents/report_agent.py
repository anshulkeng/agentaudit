"""
Turns the causal analysis output into a plain-English root-cause report. Uses
a template rather than an LLM summarization call, so it runs with zero
external dependencies. Swap in an HF summarization pipeline if you want more
natural prose, e.g.:

    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
"""


def _fmt(value, spec="+.3f"):
    """Formats a number, or a plain dash if the causal engine couldn't
    produce one (constant factor / perfect separation / too little data --
    see causal/scm_analysis.py's adjusted_effect for when this happens)."""
    return format(value, spec) if value is not None else "n/a"


def write_report(causal_results: dict, overall_failure_rate: float) -> str:
    lines = []
    lines.append("AgentAudit Root-Cause Report")
    lines.append(f"Overall failure rate: {overall_failure_rate:.1%}\n")

    for factor, data in causal_results["factor_analysis"].items():
        naive = data["naive_correlation"]
        adj = data["adjusted"]
        coefficient, p_value = adj["coefficient"], adj["p_value"]

        if coefficient is None or p_value is None:
            divergence_note = f" -> COULD NOT ESTIMATE: {adj.get('note', 'insufficient data for this factor.')}"
        elif naive is not None and abs(naive) > 0.05 and abs(coefficient) < 0.3 and p_value > 0.05:
            divergence_note = (
                " -> LIKELY SPURIOUS: naive correlation looked meaningful but the "
                "causally-adjusted estimate is small and not statistically significant "
                f"once {adj['adjustment_set']} is controlled for."
            )
        elif abs(coefficient) >= 0.3 and p_value <= 0.05:
            divergence_note = " -> Effect holds up after adjustment; likely a genuine driver."
        else:
            divergence_note = " -> No strong evidence of a causal effect, naive or adjusted."

        lines.append(
            f"- {factor}: naive correlation = {_fmt(naive)} | "
            f"adjusted coefficient = {_fmt(coefficient)} (p={_fmt(p_value, '.4f') if p_value is not None else 'n/a'}), "
            f"adjusted for {adj['adjustment_set'] or 'nothing (no confounders identified)'}"
            f"{divergence_note}"
        )

    lines.append(
        "\nCausal graph edges asserted by the analyst: "
        + ", ".join(f"{a}->{b}" for a, b in causal_results["graph_edges"])
    )
    return "\n".join(lines)
