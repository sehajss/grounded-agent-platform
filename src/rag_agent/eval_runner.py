"""Run retrieval evaluation over a labeled question set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from rag_agent.eval_dataset import EvalQuestion
from rag_agent.evaluation import (
    EvaluationSummary,
    score_retrieval,
    summarize_metrics,
)


@dataclass(frozen=True)
class QuestionEvaluation:
    question_id: str
    retrieved_sources: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass(frozen=True)
class EvaluationReport:
    k: int
    summary: EvaluationSummary
    questions: tuple[QuestionEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "summary": asdict(self.summary),
            "questions": [asdict(row) for row in self.questions],
        }


def run_retrieval_evaluation(
    questions: Iterable[EvalQuestion],
    retrieve: Callable[[str, int], Iterable[str]],
    *,
    k: int = 5,
) -> EvaluationReport:
    """Evaluate a retrieval callback against expected source paths or IDs."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    details: list[QuestionEvaluation] = []
    metric_rows = []

    for item in questions:
        ranked_sources = tuple(retrieve(item.question, k))[:k]
        metrics = score_retrieval(
            item.expected_sources,
            ranked_sources,
            k=k,
        )
        metric_rows.append(metrics)
        details.append(
            QuestionEvaluation(
                question_id=item.id,
                retrieved_sources=ranked_sources,
                recall_at_k=metrics.recall_at_k,
                reciprocal_rank=metrics.reciprocal_rank,
                ndcg_at_k=metrics.ndcg_at_k,
            )
        )

    return EvaluationReport(
        k=k,
        summary=summarize_metrics(metric_rows),
        questions=tuple(details),
    )
