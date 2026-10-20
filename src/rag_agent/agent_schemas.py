"""API schemas for repository developer-agent runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=5, max_length=4_000)
    repository_path: str = Field(min_length=1, max_length=1_000)
    allow_writes: bool = False


class AgentToolTraceResponse(BaseModel):
    tool: str
    arguments: dict[str, object]
    reason: str
    result: str


class AgentRunResponse(BaseModel):
    task: str
    plan_summary: str
    trace: list[AgentToolTraceResponse]
    final_response: str