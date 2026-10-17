"""
Retrieval ranking metrics for positive (supported) scenarios.

Unsupported / policy / adversarial scenarios are summarized separately and
must not be scored as if gold chunk IDs existed.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], *, k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = {str(x) for x in relevant_ids}
    if not relevant:
        raise ValueError("relevant_ids must be non-empty for hit@k")
    top = [str(x) for x in ranked_ids[:k]]
    return 1.0 if any(item in relevant for item in top) else 0.0


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = {str(x) for x in relevant_ids}
    if not relevant:
        raise ValueError("relevant_ids must be non-empty for recall@k")
    top = {str(x) for x in ranked_ids[:k]}
    return len(top & relevant) / float(len(relevant))


def mean_reciprocal_rank(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    relevant = {str(x) for x in relevant_ids}
    if not relevant:
        raise ValueError("relevant_ids must be non-empty for MRR")
    for index, item in enumerate(ranked_ids, start=1):
        if str(item) in relevant:
            return 1.0 / float(index)
    return 0.0


def summarize_positive_retrieval(
    per_query: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    Aggregate Hit@1, Recall@3, MRR over supported queries.

    Each item must include hit_at_1, recall_at_3, mrr floats.
    """
    if not per_query:
        return {
            "n": 0.0,
            "hit_at_1": 0.0,
            "recall_at_3": 0.0,
            "mrr": 0.0,
        }
    n = float(len(per_query))
    return {
        "n": n,
        "hit_at_1": sum(float(x["hit_at_1"]) for x in per_query) / n,
        "recall_at_3": sum(float(x["recall_at_3"]) for x in per_query) / n,
        "mrr": sum(float(x["mrr"]) for x in per_query) / n,
    }


def summarize_negative_retrieval(
    per_query: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Report how often unsupported queries still retrieve any neighbor.

    Does not compute recall/MRR (no positive gold set).
    """
    if not per_query:
        return {
            "n": 0,
            "retrieved_any": 0,
            "retrieved_any_rate": 0.0,
            "mean_top1_similarity": None,
        }
    retrieved_any = sum(1 for x in per_query if int(x.get("n_retrieved", 0)) > 0)
    sims = [
        float(x["top1_similarity"])
        for x in per_query
        if x.get("top1_similarity") is not None
    ]
    stats: dict[str, Any] = {
        "n": len(per_query),
        "retrieved_any": retrieved_any,
        "retrieved_any_rate": retrieved_any / float(len(per_query)),
        "mean_top1_similarity": (sum(sims) / len(sims)) if sims else None,
        "min_top1_similarity": min(sims) if sims else None,
        "median_top1_similarity": _median(sims) if sims else None,
        "max_top1_similarity": max(sims) if sims else None,
    }
    return stats


def similarity_distribution(values: Sequence[float]) -> dict[str, float | None]:
    """Min / median / max / mean for a similarity sample (report-only)."""
    nums = [float(v) for v in values]
    if not nums:
        return {
            "n": 0.0,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": float(len(nums)),
        "min": min(nums),
        "median": _median(nums),
        "max": max(nums),
        "mean": sum(nums) / float(len(nums)),
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
