from __future__ import annotations

import hashlib
import mimetypes
import time
from pathlib import Path
from uuid import UUID

from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.chunking import chunk_markdown
from rag_agent.config import get_settings
from rag_agent.document_loader import extract_document_text
from rag_agent.embeddings import Embedder
from rag_agent.metrics import INGESTION_CHUNKS, INGESTION_DOCUMENTS, INGESTION_LATENCY
from rag_agent.models import Document, DocumentChunk


class IngestionService:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.settings = get_settings()

    async def ingest(
        self,
        db: AsyncSession,
        *,
        workspace_id: UUID,
        uploaded_by: UUID,
        filename: str,
        content: bytes,
    ) -> tuple[Document, int]:
        started = time.perf_counter()
        outcome = "error"
        document_id: UUID | None = None
        chunk_count = 0

        try:
            safe_filename = Path(filename.replace("\\", "/")).name
            if not safe_filename:
                raise ValueError("A filename is required")
            if len(safe_filename) > 255:
                raise ValueError("The filename must be 255 characters or fewer")

            document = Document(
                workspace_id=workspace_id,
                uploaded_by=uploaded_by,
                title=Path(safe_filename).stem[:300] or safe_filename,
                original_filename=safe_filename,
                content_type=mimetypes.guess_type(safe_filename)[0]
                or "application/octet-stream",
                sha256=hashlib.sha256(content).hexdigest(),
                status="processing",
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)
            document_id = document.id

            text = await run_in_threadpool(
                extract_document_text,
                safe_filename,
                content,
            )
            chunks = await run_in_threadpool(
                chunk_markdown,
                text,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            if not chunks:
                raise ValueError("The document produced no searchable text")

            for start in range(0, len(chunks), 64):
                batch = chunks[start : start + 64]
                vectors = await run_in_threadpool(
                    self.embedder.encode,
                    [chunk.content for chunk in batch],
                )
                db.add_all(
                    [
                        DocumentChunk(
                            workspace_id=workspace_id,
                            document_id=document_id,
                            chunk_index=chunk.index,
                            content=chunk.content,
                            token_count=chunk.token_count,
                            attributes={
                                **chunk.attributes,
                                "source_filename": safe_filename,
                            },
                            embedding=vector,
                        )
                        for chunk, vector in zip(batch, vectors, strict=True)
                    ]
                )
                await db.flush()

            document.status = "ready"
            document.error_message = None
            await db.commit()
            await db.refresh(document)

            chunk_count = len(chunks)
            outcome = "success"
            INGESTION_CHUNKS.inc(chunk_count)
            return document, chunk_count

        except Exception as exc:
            await db.rollback()
            if document_id is not None:
                try:
                    failed_document = await db.get(Document, document_id)
                    if failed_document is not None:
                        failed_document.status = "failed"
                        failed_document.error_message = (
                            f"Ingestion failed ({type(exc).__name__})"
                        )
                        await db.commit()
                except Exception:
                    await db.rollback()
            raise
        finally:
            INGESTION_DOCUMENTS.labels(outcome).inc()
            INGESTION_LATENCY.observe(time.perf_counter() - started)