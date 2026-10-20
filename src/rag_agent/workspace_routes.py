from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.auth import get_current_user
from rag_agent.database import get_db
from rag_agent.models import (
    User,
    Workspace,
    WorkspaceMembership,
)
from rag_agent.schemas import WorkspaceCreate, WorkspaceRead


router = APIRouter(prefix="/v1", tags=["workspaces"])


@router.get("/workspaces", response_model=list[WorkspaceRead])
async def list_workspaces(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[Workspace]:
    result = await db.execute(
        select(Workspace)
        .join(
            WorkspaceMembership,
            WorkspaceMembership.workspace_id == Workspace.id,
        )
        .where(WorkspaceMembership.user_id == current_user.id)
        .order_by(Workspace.created_at)
    )
    return list(result.scalars().all())


@router.post(
    "/organizations/{organization_id}/workspaces",
    response_model=WorkspaceRead,
    status_code=201,
)
async def create_workspace(
    organization_id: UUID,
    payload: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Workspace:
    permission = await db.execute(
        select(WorkspaceMembership.role)
        .join(Workspace, Workspace.id == WorkspaceMembership.workspace_id)
        .where(
            Workspace.organization_id == organization_id,
            WorkspaceMembership.user_id == current_user.id,
            WorkspaceMembership.role.in_(("owner", "admin")),
        )
        .limit(1)
    )
    if permission.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=403,
            detail="Only an organization workspace owner or admin can create workspaces",
        )

    workspace = Workspace(
        organization_id=organization_id,
        name=payload.name.strip(),
        slug=payload.slug,
    )
    db.add(workspace)

    try:
        await db.flush()
        db.add(
            WorkspaceMembership(
                user_id=current_user.id,
                workspace_id=workspace.id,
                role="owner",
            )
        )
        await db.commit()
        await db.refresh(workspace)
        return workspace
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="That workspace slug is already in use in this organization",
        ) from exc