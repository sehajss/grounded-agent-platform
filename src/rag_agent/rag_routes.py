from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.auth import get_current_user, require_workspace_role
from rag_agent.database import get_db
from rag_agent.document_loader import MAX_DOCUMENT_BYTES
from rag_agent.models import Document, User
from rag_agent.schemas import (
    DocumentRead,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)


router = APIRouter(prefix="/v1/workspaces/{workspace_id}", tags=["workspace data"])


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_document(
    workspace_id: UUID,
    request: Request,
    file: UploadFile = File(...),
    _membership=Depends(require_workspace_role("owner", "admin", "member")),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=422, detail="A filename is required")

    try:
        content = await file.read(MAX_DOCUMENT_BYTES + 1)
        document, chunk_count = await request.app.state.ingestion_service.ingest(
            db,
            workspace_id=workspace_id,
            uploaded_by=current_user.id,
            filename=file.filename,
            content=content,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    return DocumentUploadResponse(
        document=DocumentRead.model_validate(document),
        chunks_created=chunk_count,
    )


@router.get("/documents", response_model=list[DocumentRead])
async def list_documents(
    workspace_id: UUID,
    _membership=Depends(
        require_workspace_role("owner", "admin", "member", "viewer")
    ),
    db: AsyncSession = Depends(get_db),
) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    workspace_id: UUID,
    document_id: UUID,
    _membership=Depends(require_workspace_role("owner", "admin", "member")),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.workspace_id == workspace_id,
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.delete(document)
    await db.commit()


@router.post("/query", response_model=QueryResponse)
async def query_workspace(
    workspace_id: UUID,
    payload: QueryRequest,
    request: Request,
    _membership=Depends(
        require_workspace_role("owner", "admin", "member", "viewer")
    ),
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    return await request.app.state.query_service.answer(
        db,
        workspace_id=workspace_id,
        question=payload.question,
        top_k=payload.top_k,
        request_id=request.state.request_id,
    )