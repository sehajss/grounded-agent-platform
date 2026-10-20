"""Metrics for evaluating document retrieval against labeled examples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RetrievalExample:
    query: str
    expected_document_ids: frozenset[str]


@dataclass(frozen=True)
class ExampleMetrics:
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass(frozen=True)
class EvaluationSummary:
    example_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float


def score_retrieval(
    expected_document_ids: Iterable[str],
    retrieved_document_ids: Iterable[str],
    *,
    k: int = 5,
) -> ExampleMetrics:
    """Score one ranked retrieval result against its relevant document IDs."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    expected = set(expected_document_ids)
    # Retrieval returns ranked chunks, so one document can appear more than
    # once. Score unique documents only; repeated chunks are not extra hits.
    ranked = list(dict.fromkeys(retrieved_document_ids))[:k]

    if not expected:
        return ExampleMetrics(0.0, 0.0, 0.0)

    hits = [document_id in expected for document_id in ranked]
    recall = sum(hits) / len(expected)

    reciprocal_rank = 0.0
    for rank, is_relevant in enumerate(hits, start=1):
        if is_relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_relevant in enumerate(hits, start=1)
        if is_relevant
    )
    ideal_hits = min(len(expected), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return ExampleMetrics(
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k=ndcg,
    )


def summarize_metrics(
    results: Iterable[ExampleMetrics],
) -> EvaluationSummary:
    """Average per-example metrics into one evaluation summary."""
    rows = list(results)
    if not rows:
        return EvaluationSummary(0, 0.0, 0.0, 0.0)

    count = len(rows)
    return EvaluationSummary(
        example_count=count,
        recall_at_k=sum(row.recall_at_k for row in rows) / count,
        mean_reciprocal_rank=(
            sum(row.reciprocal_rank for row in rows) / count
        ),
        mean_ndcg_at_k=sum(row.ndcg_at_k for row in rows) / count,
    )
