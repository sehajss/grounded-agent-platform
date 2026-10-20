from __future__ import annotations

import re
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.auth import get_current_user
from rag_agent.config import get_settings
from rag_agent.database import get_db
from rag_agent.models import (
    Organization,
    User,
    Workspace,
    WorkspaceMembership,
)
from rag_agent.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserRead,
)
from rag_agent.security import create_access_token, hash_password, verify_password


router = APIRouter(prefix="/v1/auth", tags=["authentication"])


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:88] or "workspace"


@router.post("/register", response_model=UserRead, status_code=201)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    settings = get_settings()
    if settings.app_env == "production":
        raise HTTPException(status_code=404, detail="Registration is disabled")

    password_hash = await run_in_threadpool(hash_password, payload.password)
    user = User(
        email=payload.email.strip().lower(),
        password_hash=password_hash,
    )
    organization = Organization(
        name=payload.organization_name.strip(),
        slug=f"{_slugify(payload.organization_name)}-{uuid4().hex[:8]}",
    )

    try:
        db.add_all([user, organization])
        await db.flush()

        workspace = Workspace(
            organization_id=organization.id,
            name=payload.workspace_name.strip(),
            slug="default",
        )
        db.add(workspace)
        await db.flush()

        db.add(
            WorkspaceMembership(
                user_id=user.id,
                workspace_id=workspace.id,
                role="owner",
            )
        )
        await db.commit()
        await db.refresh(user)
        return user
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="An account with that email may already exist",
        ) from exc


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(
        select(User).where(User.email == payload.email.strip().lower())
    )
    user = result.scalar_one_or_none()

    password_is_valid = False
    if user is not None and user.is_active:
        password_is_valid = await run_in_threadpool(
            verify_password,
            payload.password,
            user.password_hash,
        )

    if not password_is_valid or user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    settings = get_settings()
    return TokenResponse(
        access_token=create_access_token(user.id, settings),
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user