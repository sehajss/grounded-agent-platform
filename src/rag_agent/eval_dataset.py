"""Load and validate the retrieval evaluation question set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_sources: tuple[str, ...]


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    """Read evaluation questions from a YAML file."""
    question_path = Path(path)
    with question_path.open("r", encoding="utf-8") as file:
        raw_questions = yaml.safe_load(file)

    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("The evaluation YAML must contain a non-empty list.")

    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise ValueError(f"Question at index {index} must be a mapping.")

        question_id = item.get("id")
        question = item.get("question")
        sources = item.get("expected_sources")

        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Question at index {index} needs a non-empty id.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate evaluation question id: {question_id}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question {question_id!r} needs question text.")
        if not isinstance(sources, list) or not sources:
            raise ValueError(
                f"Question {question_id!r} needs at least one expected source."
            )
        if any(not isinstance(source, str) or not source.strip() for source in sources):
            raise ValueError(
                f"Every expected source for {question_id!r} must be non-empty text."
            )

        seen_ids.add(question_id)
        questions.append(
            EvalQuestion(
                id=question_id,
                question=question.strip(),
                expected_sources=tuple(source.strip() for source in sources),
            )
        )

    return questions
