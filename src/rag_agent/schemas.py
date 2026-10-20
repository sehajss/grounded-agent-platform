from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    email: str = Field(
        min_length=5,
        max_length=320,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )
    password: str = Field(min_length=12, max_length=128)
    organization_name: str = Field(min_length=2, max_length=160)
    workspace_name: str = Field(min_length=2, max_length=160)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    is_active: bool
    created_at: datetime


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    slug: str = Field(
        min_length=2,
        max_length=100,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )


class WorkspaceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    organization_id: UUID
    name: str
    slug: str
    created_at: datetime


class Citation(BaseModel):
    document_id: UUID
    filename: str
    chunk_index: int
    excerpt: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    request_id: str


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    original_filename: str
    content_type: str
    status: Literal["processing", "ready", "failed"]
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    document: DocumentRead
    chunks_created: int