"""
Database engine configuration for async PostgreSQL.

Provides async SQLAlchemy 2.0 engine and session management for the Chat API.
Uses asyncpg driver for high-performance async PostgreSQL operations.

Configuration Environment Variables:
    DATABASE_URL: PostgreSQL connection string
        Format: postgresql+asyncpg://user:password@host:port/database
        Default: postgresql+asyncpg://localhost/chatdb
    
    SQL_ECHO: Enable SQL statement logging for debugging
        Values: "true" or "false" (case-insensitive)
        Default: "false"
    
    DB_POOL_SIZE: Connection pool size (number of persistent connections)
        Default: 5
    
    DB_MAX_OVERFLOW: Maximum overflow connections above pool_size
        Default: 10
    
    DB_POOL_TIMEOUT: Seconds to wait for connection from pool
        Default: 30
    
    DB_POOL_RECYCLE: Seconds before connection is recycled (prevents stale connections)
        Default: 3600 (1 hour)

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://localhost/chatdb"
)

SQL_ECHO: bool = os.getenv("SQL_ECHO", "false").lower() == "true"

# Connection pool settings
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# ============================================================================
# Engine Configuration
# ============================================================================

def _create_engine() -> AsyncEngine:
    """
    Create the async SQLAlchemy engine with connection pooling.
    
    Returns:
        AsyncEngine: Configured async engine instance
        
    Note:
        Connection pooling is handled by SQLAlchemy's QueuePool (default).
        For serverless environments, consider using NullPool instead.
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return create_async_engine(
        DATABASE_URL,
        echo=SQL_ECHO,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_recycle=DB_POOL_RECYCLE,
        pool_pre_ping=True,  # Verify connections before use
    )


# Global engine instance - created once at module load
engine: AsyncEngine = _create_engine()

# ============================================================================
# Session Factory
# ============================================================================

# async_sessionmaker (SQLAlchemy 2.0+) replaces sessionmaker for async
# Created once and reused for all requests - more efficient than per-request creation
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent attribute expiry after commit
    autocommit=False,
    autoflush=False,
)


# ============================================================================
# Public API
# ============================================================================

async def init_db() -> None:
    """
    Initialize database tables.
    
    Creates all SQLModel tables defined in models.py if they don't exist.
    Called during application startup via lifespan context manager.
    
    This function is idempotent - safe to call multiple times.
    
    Returns:
        None
        
    Side Effects:
        Creates database tables in the configured database
        
    Raises:
        SQLAlchemyError: If database connection or table creation fails
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    logger.info("Creating database tables if not exists...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    logger.info("Database tables initialized")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async database sessions.
    
    Provides a request-scoped async SQLAlchemy session with automatic
    cleanup. Use with FastAPI's Depends() for dependency injection.
    
    The session is automatically closed when the request completes,
    whether successful or not.
    
    Yields:
        AsyncSession: Async database session for the request
        
    Example:
        @router.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(Item))
            return result.scalars().all()
            
    Note:
        - Sessions are request-scoped (one per request)
        - expire_on_commit=False prevents lazy load issues
        - Use selectinload() for eager loading relationships
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def close_db() -> None:
    """
    Close all database connections.
    
    Called during application shutdown to properly dispose of the
    connection pool and release all database connections.
    
    Returns:
        None
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    logger.info("Closing database connections...")
    await engine.dispose()
    logger.info("Database connections closed")


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of request context.
    
    Use this for background tasks, CLI commands, or other code that
    runs outside of FastAPI's dependency injection system.
    
    Yields:
        AsyncSession: Async database session
        
    Example:
        async with get_session_context() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()
            
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
