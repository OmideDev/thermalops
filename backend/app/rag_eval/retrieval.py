"""
Retrieval-only evaluation over the curated dataset.

Uses the existing InMemorySemanticIndex / search_index path. Does not call
Ollama or ask_with_evidence.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.rag.search import (
    EmbeddingBackend,
    InMemorySemanticIndex,
    search_index,
)
from app.rag_eval.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    similarity_distribution,
    summarize_negative_retrieval,
    summarize_positive_retrieval,
)


def evaluate_retrieval(
    scenarios: Sequence[dict[str, Any]],
    index: InMemorySemanticIndex,
    embedder: EmbeddingBackend,
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Score supported queries with Hit@1 / Recall@3 / MRR.

    Unsupported / policy / adversarial scenarios are reported separately and
    never enter positive recall/MRR aggregates.
    """
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    policy: list[dict[str, Any]] = []
    adversarial: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for scenario in scenarios:
        question = str(scenario["question"])
        hits = search_index(index, question, embedder, top_k=top_k)
        ranked_ids = [str(hit["chunk_id"]) for hit in hits]
        row: dict[str, Any] = {
            "id": scenario["id"],
            "category": scenario["category"],
            "split": scenario["split"],
            "n_retrieved": len(hits),
            "ranked_chunk_ids": ranked_ids,
            "ranked_pages": [hit.get("page_number") for hit in hits],
            "ranked_headings": [hit.get("section_heading") for hit in hits],
            "top1_similarity": (
                float(hits[0]["similarity"]) if hits else None
            ),
        }

        split = scenario["split"]
        if split == "retrieval_positive":
            gold = list(scenario["expected_chunk_ids"])
            row["hit_at_1"] = hit_at_k(ranked_ids, gold, k=1)
            row["recall_at_3"] = recall_at_k(ranked_ids, gold, k=min(3, top_k))
            row["mrr"] = mean_reciprocal_rank(ranked_ids, gold)
            positives.append(row)
            if row["hit_at_1"] < 1.0:
                failures.append(
                    {
                        "id": scenario["id"],
                        "question": question,
                        "gold_chunk_ids": gold,
                        "ranked_chunk_ids": ranked_ids,
                        "ranked_pages": row["ranked_pages"],
                        "ranked_headings": row["ranked_headings"],
                    }
                )
        elif split == "retrieval_negative":
            negatives.append(row)
        elif split == "policy":
            policy.append(row)
        elif split == "adversarial":
            adversarial.append(row)
        else:
            raise ValueError(f"unknown split: {split}")

    def _sims(rows: list[dict[str, Any]]) -> list[float]:
        return [
            float(row["top1_similarity"])
            for row in rows
            if row.get("top1_similarity") is not None
        ]

    return {
        "top_k": top_k,
        "positive": {
            "aggregate": summarize_positive_retrieval(positives),
            "top1_similarity": similarity_distribution(_sims(positives)),
            "per_query": positives,
            "failed_hit_at_1": failures,
        },
        "negative_unsupported": {
            "aggregate": summarize_negative_retrieval(negatives),
            "top1_similarity": similarity_distribution(_sims(negatives)),
            "per_query": negatives,
            "note": (
                "Nearest neighbors may still appear; this is not evidence "
                "support and is not scored as recall/MRR."
            ),
        },
        "policy": {
            "aggregate": summarize_negative_retrieval(policy),
            "top1_similarity": similarity_distribution(_sims(policy)),
            "per_query": policy,
            "note": (
                "Operational-policy abstention is an application gate; "
                "retrieval neighbors are descriptive only."
            ),
        },
        "adversarial": {
            "n": len(adversarial),
            "per_query": adversarial,
            "note": (
                "Adversarial items are not scored against the production "
                "index gold set. Use synthetic fixtures for generation tests."
            ),
        },
    }
