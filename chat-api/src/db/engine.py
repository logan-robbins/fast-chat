"""
Database engine configuration for async SQLite.

Provides async SQLAlchemy engine and session management for the Chat API.
Uses aiosqlite driver for async SQLite operations.

Configuration:
    DATABASE_URL: Connection string (default: sqlite+aiosqlite:///./chat.db)
    SQL_ECHO: Enable SQL statement logging (default: False)

Last Grunted: 02/03/2026 10:30:00 AM UTC
"""
import os
from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator

# Database URL (Async SQLite)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chat.db")

# Create Async Engine
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "False").lower() == "true",
    connect_args={"check_same_thread": False},  # Needed for SQLite
)


async def init_db() -> None:
    """
    Initialize database tables.
    
    Creates all SQLModel tables defined in models.py if they don't exist.
    Called during application startup via lifespan context manager.
    
    Returns:
        None
        
    Side Effects:
        Creates database tables in the configured database
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async database sessions.
    
    Provides an async SQLAlchemy session with automatic cleanup.
    Use with FastAPI's Depends() for request-scoped sessions.
    
    Yields:
        AsyncSession: Async database session
        
    Example:
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()
            
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
