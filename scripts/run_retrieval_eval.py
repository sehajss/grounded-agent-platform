"""Evaluate the app's hybrid retriever against eval/questions.yaml."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from rag_agent.api import app
from rag_agent.database import get_db
from rag_agent.eval_dataset import load_eval_questions
from rag_agent.evaluation import score_retrieval, summarize_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval for a workspace."
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="UUID of the workspace containing the ingested documents.",
    )
    parser.add_argument(
        "--questions",
        default=str(PROJECT_ROOT / "eval" / "questions.yaml"),
        help="Path to the YAML evaluation questions.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "eval" / "results" / "retrieval.json"),
        help="Path where the JSON evaluation report will be written.",
    )
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def source_key(value: str) -> str:
    """Compare source paths by filename, as retrieval returns filenames."""
    return Path(value.replace("\\", "/")).name.casefold()


async def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.k < 1:
        raise ValueError("--k must be at least 1.")

    workspace_id = UUID(args.workspace_id)
    questions = load_eval_questions(args.questions)
    report_rows: list[dict[str, object]] = []
    metric_rows = []

    # Entering the app lifespan initializes its embedder and query services.
    async with app.router.lifespan_context(app):
        query_service = getattr(app.state, "query_service", None)
        retriever = getattr(query_service, "retriever", None)
        if retriever is None:
            raise RuntimeError(
                "The app's query service does not expose its retriever. "
                "Check the query service setup before running this evaluation."
            )

        db_generator = get_db()
        db = await anext(db_generator)
        try:
            for item in questions:
                chunks = await retriever.search(
                    db,
                    workspace_id=workspace_id,
                    query=item.question,
                    top_k=args.k,
                )

                expected = {source_key(path) for path in item.expected_sources}
                # The retriever ranks chunks, but this evaluation measures
                # document retrieval. Keep the first-ranked chunk per file.
                unique_chunks = []
                seen_sources = set()
                for chunk in chunks:
                    key = source_key(chunk.filename)
                    if key not in seen_sources:
                        seen_sources.add(key)
                        unique_chunks.append(chunk)

                retrieved = [source_key(chunk.filename) for chunk in unique_chunks]
                metrics = score_retrieval(expected, retrieved, k=args.k)
                metric_rows.append(metrics)

                report_rows.append(
                    {
                        "id": item.id,
                        "question": item.question,
                        "expected_sources": list(item.expected_sources),
                        "retrieved_sources": [
                            chunk.filename for chunk in unique_chunks
                        ],
                        "recall_at_k": metrics.recall_at_k,
                        "reciprocal_rank": metrics.reciprocal_rank,
                        "ndcg_at_k": metrics.ndcg_at_k,
                    }
                )
        finally:
            await db_generator.aclose()

    summary = summarize_metrics(metric_rows)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_id": str(workspace_id),
        "k": args.k,
        "summary": {
            "example_count": summary.example_count,
            "recall_at_k": summary.recall_at_k,
            "mean_reciprocal_rank": summary.mean_reciprocal_rank,
            "mean_ndcg_at_k": summary.mean_ndcg_at_k,
        },
        "questions": report_rows,
    }


async def main() -> None:
    args = parse_args()
    result = await evaluate(args)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result["summary"], indent=2))
    print(f"Saved evaluation report: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
