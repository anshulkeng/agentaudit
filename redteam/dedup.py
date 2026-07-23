"""
Embedding-based near-duplicate removal for generated test cases. Doesn't
matter for the templated generator (those are deduped by construction) but
generate_cases_llm() produces noisy, sometimes near-identical samples that
need cleaning up before they burn executor budget.

Requires: pip install sentence-transformers
First call downloads the embedding model - run this on your own machine.
"""

_embedder = None  # lazy singleton, don't want an import to trigger a model load


def _get_embedder(model: str):
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "dedup_cases() needs sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        _embedder = SentenceTransformer(model)
    return _embedder


def dedup_cases(
    cases: list[dict],
    threshold: float = 0.88,
    model: str = "all-MiniLM-L6-v2",
    text_key: str = "input_text",
) -> list[dict]:
    """Keeps a case only if it isn't near-duplicate (cosine similarity >=
    threshold) with anything already kept. O(n^2), fine at a few hundred
    cases per category, would need something smarter past that.

    Dedup runs separately per task_category so a "how_to" case never gets
    dropped just for sharing phrasing with an unrelated "refund_request".
    """
    from sentence_transformers import util

    embedder = _get_embedder(model)
    kept: list[dict] = []

    by_category: dict[str, list[dict]] = {}
    for case in cases:
        by_category.setdefault(case.get("task_category", "_none"), []).append(case)

    for category, category_cases in by_category.items():
        texts = [c[text_key] for c in category_cases]
        embeddings = embedder.encode(texts, convert_to_tensor=True)
        kept_embeddings = []
        for i, emb in enumerate(embeddings):
            is_dup = any(util.cos_sim(emb, k).item() >= threshold for k in kept_embeddings)
            if not is_dup:
                kept.append(category_cases[i])
                kept_embeddings.append(emb)

    return kept


def dedup_report(original: list[dict], deduped: list[dict]) -> dict:
    """Quick before/after summary, handy for logging or dropping into the README."""
    return {
        "original_count": len(original),
        "deduped_count": len(deduped),
        "removed_count": len(original) - len(deduped),
        "removed_pct": round(100 * (1 - len(deduped) / len(original)), 1) if original else 0.0,
    }
