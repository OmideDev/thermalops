"""Reproducible RAG evaluation helpers (retrieval + answer; offline by default)."""

from app.rag_eval.answer_selection import load_answer_selection, selected_scenarios
from app.rag_eval.dataset import (
    CATEGORIES,
    EVAL_SPLITS,
    EXPECTED_BEHAVIORS,
    load_dataset,
    validate_dataset,
)
from app.rag_eval.metrics import (
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    summarize_negative_retrieval,
    summarize_positive_retrieval,
)

__all__ = [
    "CATEGORIES",
    "EVAL_SPLITS",
    "EXPECTED_BEHAVIORS",
    "load_dataset",
    "validate_dataset",
    "load_answer_selection",
    "selected_scenarios",
    "hit_at_k",
    "mean_reciprocal_rank",
    "recall_at_k",
    "summarize_negative_retrieval",
    "summarize_positive_retrieval",
]
