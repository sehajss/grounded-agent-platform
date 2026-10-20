from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from rag_agent.config import get_settings


settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Base class for database models and migrations."""


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


async def close_database() -> None:
    await engine.dispose()