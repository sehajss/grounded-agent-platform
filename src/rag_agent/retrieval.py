from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.embeddings import Embedder
from rag_agent.metrics import RETRIEVAL_CALLS, RETRIEVAL_LATENCY, RETRIEVED_CHUNKS
from rag_agent.models import Document, DocumentChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    content: str
    heading: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    async def search(
        self,
        db: AsyncSession,
        *,
        workspace_id: UUID,
        query: str,
        top_k: int = 8,
        candidate_k: int | None = None,
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        outcome = "error"

        try:
            if not query.strip():
                outcome = "success"
                RETRIEVED_CHUNKS.observe(0)
                return []
            if top_k < 1:
                raise ValueError("top_k must be positive")

            candidate_k = candidate_k or max(top_k * 4, 30)
            vector = (await run_in_threadpool(self.embedder.encode, [query]))[0]

            distance = DocumentChunk.embedding.cosine_distance(vector)
            dense_statement = (
                select(
                    DocumentChunk,
                    Document.original_filename,
                    distance.label("distance"),
                )
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    DocumentChunk.workspace_id == workspace_id,
                    Document.status == "ready",
                    DocumentChunk.embedding.is_not(None),
                )
                .order_by(distance.asc())
                .limit(candidate_k)
            )
            dense_rows = (await db.execute(dense_statement)).all()

            search_vector = func.to_tsvector("english", DocumentChunk.content)
            search_query = func.plainto_tsquery("english", query)
            text_rank = func.ts_rank_cd(search_vector, search_query)
            lexical_statement = (
                select(
                    DocumentChunk,
                    Document.original_filename,
                    text_rank.label("rank"),
                )
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    DocumentChunk.workspace_id == workspace_id,
                    Document.status == "ready",
                    search_vector.op("@@")(search_query),
                )
                .order_by(text_rank.desc())
                .limit(candidate_k)
            )
            lexical_rows = (await db.execute(lexical_statement)).all()

            chunks: dict[UUID, tuple[DocumentChunk, str]] = {}
            dense_ranks: dict[UUID, int] = {}
            lexical_ranks: dict[UUID, int] = {}

            for rank, row in enumerate(dense_rows, start=1):
                chunk, filename, _distance = row
                chunks[chunk.id] = (chunk, filename)
                dense_ranks[chunk.id] = rank

            for rank, row in enumerate(lexical_rows, start=1):
                chunk, filename, _text_score = row
                chunks[chunk.id] = (chunk, filename)
                lexical_ranks[chunk.id] = rank

            fused_scores: dict[UUID, float] = {}
            for chunk_id in chunks:
                score = 0.0
                if chunk_id in dense_ranks:
                    score += self.dense_weight / (self.rrf_k + dense_ranks[chunk_id])
                if chunk_id in lexical_ranks:
                    score += self.lexical_weight / (self.rrf_k + lexical_ranks[chunk_id])
                fused_scores[chunk_id] = score

            ranked_ids = sorted(
                fused_scores,
                key=fused_scores.__getitem__,
                reverse=True,
            )[:top_k]

            results = []
            for chunk_id in ranked_ids:
                chunk, filename = chunks[chunk_id]
                heading = (chunk.attributes or {}).get("heading", "")
                results.append(
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        filename=filename,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        heading=str(heading),
                        score=fused_scores[chunk_id],
                        dense_rank=dense_ranks.get(chunk_id),
                        lexical_rank=lexical_ranks.get(chunk_id),
                    )
                )

            outcome = "success"
            RETRIEVED_CHUNKS.observe(len(results))
            return results
        finally:
            RETRIEVAL_CALLS.labels(outcome).inc()
            RETRIEVAL_LATENCY.observe(time.perf_counter() - started)