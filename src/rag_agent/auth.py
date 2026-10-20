from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.database import get_db
from rag_agent.models import User, WorkspaceMembership
from rag_agent.security import decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Bearer token is required")

    try:
        user_id = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User is not active")

    return user


def require_workspace_role(
    *allowed_roles: str,
) -> Callable:
    if not allowed_roles:
        raise ValueError("At least one workspace role must be allowed")

    async def check_membership(
        workspace_id: UUID,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> WorkspaceMembership:
        result = await db.execute(
            select(WorkspaceMembership).where(
                WorkspaceMembership.user_id == current_user.id,
                WorkspaceMembership.workspace_id == workspace_id,
            )
        )
        membership = result.scalar_one_or_none()

        if membership is None or membership.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this workspace",
            )

        return membership

    return check_membership