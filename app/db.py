from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


database_url: Optional[str] = settings.DATABASE_URL

# Handle potential protocol mismatch for asyncpg
if database_url:
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql://") and "+asyncpg" not in database_url:
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

if not database_url:
    # DATABASE_URL not configured – keep placeholders so imports don't crash.
    engine = None
    async_session: Optional[async_sessionmaker[AsyncSession]] = None
else:
    engine = create_async_engine(database_url, echo=False, future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides an async DB session."""
    if async_session is None:
        raise RuntimeError("DATABASE_URL is not configured; cannot create DB session.")
    async with async_session() as session:
        yield session


async def init_db() -> None:
    """Create all tables if they do not exist. Call on startup."""
    if engine is None:
        return
    from app import models  # noqa: F401 - ensure models are imported so metadata is populated

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)



