from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["local", "test", "production"] = "local"
    app_log_level: str = "INFO"

    database_url: str = (
        "postgresql+asyncpg://agent:agent@localhost:5432/grounded_agent"
    )

    jwt_secret_key: str = "local-development-only-change-before-deploying"
    jwt_access_token_expire_minutes: int = Field(default=60, ge=5, le=1440)

    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=900, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=5000)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)

    agent_allow_writes: bool = False
    agent_max_steps: int = Field(default=8, ge=1, le=30)
    agent_max_write_bytes: int = Field(default=100000, ge=1, le=1000000)
    agent_allowed_checks: str = "pytest,ruff,mypy"

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.app_env == "production" and len(self.jwt_secret_key) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")
        return self

    @property
    def allowed_checks(self) -> set[str]:
        return {
            command.strip()
            for command in self.agent_allowed_checks.split(",")
            if command.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()