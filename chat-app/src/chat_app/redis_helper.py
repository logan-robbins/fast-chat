"""Redis helper for managing inflight request cancellation with async support.

Provides a key-value based cancellation mechanism using Redis with both
synchronous and asynchronous APIs. Uses redis-py's asyncio support for
proper async/await patterns.

The cancellation pattern works by:
1. Client sets a cancellation key in Redis when user wants to cancel
2. Service checks for cancellation key periodically during streaming
3. If key exists, service stops streaming and clears the key

Environment Variables:
    REDIS_HOST: Redis server hostname (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
    REDIS_USERNAME: Redis username (default: "default")
    REDIS_PASSWORD: Redis password (default: "")

Last Grunted: 02/04/2026 06:30:00 PM PST
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import redis
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Key prefix for all cancellation keys
CANCEL_KEY_PREFIX: str = "cancel:"

# Default TTL for cancellation keys (seconds)
DEFAULT_CANCEL_TTL: int = 60


class RedisCancelHelper:
    """Helper class for managing cancellation of inflight requests via Redis.

    Provides both synchronous and asynchronous APIs for cancellation management.
    The async API uses redis.asyncio for proper async/await patterns.

    Usage Pattern:
        1. Client calls set_cancel(thread_id) when user wants to cancel
        2. Service calls is_cancelled(thread_id) during streaming loop
        3. If cancelled, service stops streaming and calls clear_cancel(thread_id)

    Attributes:
        redis_host: Redis server hostname
        redis_port: Redis server port
        redis_username: Redis authentication username
        redis_password: Redis authentication password
        prefix: Key prefix for cancellation keys ("cancel:")
        
    Connection Management:
        - Sync client (_sync_client) is created lazily on first sync call
        - Async client (_async_client) is created lazily on first async call
        - Both clients use connection pooling automatically

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    
    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None
    ) -> None:
        """Initialize Redis cancel helper with connection parameters.

        Connections are created lazily - this constructor only stores config.

        Args:
            redis_host: Redis hostname. Defaults to REDIS_HOST env var or "localhost".
            redis_port: Redis port. Defaults to 6379.
            redis_password: Redis password. Defaults to REDIS_PASSWORD env var.

        Note:
            Connection failure is non-fatal - cancellation support is simply
            disabled and is_cancelled() always returns False.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        self.redis_host: str = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port: int = int(os.getenv("REDIS_PORT", str(redis_port)))
        self.redis_username: str = os.getenv("REDIS_USERNAME", "default")
        self.redis_password: str = redis_password or os.getenv("REDIS_PASSWORD", "")
        self.prefix: str = CANCEL_KEY_PREFIX

        # Lazy-initialized clients
        self._sync_client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        self._sync_connected: bool = False
        self._async_connected: bool = False

    def _get_sync_client(self) -> Optional[redis.Redis]:
        """Get or create the synchronous Redis client.
        
        Returns:
            redis.Redis | None: Connected client or None if connection failed.
            
        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        if self._sync_client is not None:
            return self._sync_client if self._sync_connected else None

        try:
            self._sync_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                username=self.redis_username,
                password=self.redis_password if self.redis_password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self._sync_client.ping()
            self._sync_connected = True
            logger.info(
                "Redis sync connection established",
                extra={"host": self.redis_host, "port": self.redis_port}
            )
            return self._sync_client
        except Exception as e:
            logger.warning(
                "Redis sync connection failed, cancellation support disabled",
                extra={"error": str(e), "host": self.redis_host}
            )
            self._sync_connected = False
            return None

    async def _get_async_client(self) -> Optional[aioredis.Redis]:
        """Get or create the asynchronous Redis client.
        
        Returns:
            aioredis.Redis | None: Connected client or None if connection failed.
            
        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        if self._async_client is not None:
            return self._async_client if self._async_connected else None

        try:
            self._async_client = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                username=self.redis_username,
                password=self.redis_password if self.redis_password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            await self._async_client.ping()
            self._async_connected = True
            logger.info(
                "Redis async connection established",
                extra={"host": self.redis_host, "port": self.redis_port}
            )
            return self._async_client
        except Exception as e:
            logger.warning(
                "Redis async connection failed, cancellation support disabled",
                extra={"error": str(e), "host": self.redis_host}
            )
            self._async_connected = False
            return None

    # -------------------------------------------------------------------------
    # Synchronous API
    # -------------------------------------------------------------------------
    
    def is_cancelled(self, cancel_key: str) -> bool:
        """Check if a cancellation key exists in Redis (sync).

        Args:
            cancel_key: The cancellation key to check (typically thread_id).

        Returns:
            bool: True if cancellation was requested, False otherwise.
                Always returns False if Redis is not connected.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        client = self._get_sync_client()
        if not client:
            return False
        
        try:
            result = client.exists(f"{self.prefix}{cancel_key}")
            if result:
                logger.info(
                    "Cancellation detected",
                    extra={"cancel_key": cancel_key}
                )
            return bool(result)
        except Exception as e:
            logger.error(
                "Error checking cancellation key",
                extra={"cancel_key": cancel_key, "error": str(e)}
            )
            return False
    
    def set_cancel(self, cancel_key: str, ttl: int = DEFAULT_CANCEL_TTL) -> bool:
        """Set a cancellation key in Redis with automatic expiration (sync).

        Args:
            cancel_key: The cancellation key to set (typically thread_id).
            ttl: Time-to-live in seconds before key auto-expires (default: 60).

        Returns:
            bool: True if key was set successfully, False on error.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        client = self._get_sync_client()
        if not client:
            return False
        
        try:
            client.setex(f"{self.prefix}{cancel_key}", ttl, "1")
            logger.info(
                "Cancellation key set",
                extra={"cancel_key": cancel_key, "ttl": ttl}
            )
            return True
        except Exception as e:
            logger.error(
                "Error setting cancellation key",
                extra={"cancel_key": cancel_key, "error": str(e)}
            )
            return False
    
    def clear_cancel(self, cancel_key: str) -> bool:
        """Clear a cancellation key from Redis (sync).

        Should be called after detecting cancellation to prevent stale keys.

        Args:
            cancel_key: The cancellation key to clear (typically thread_id).

        Returns:
            bool: True if key existed and was deleted, False otherwise.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        client = self._get_sync_client()
        if not client:
            return False
        
        try:
            result = client.delete(f"{self.prefix}{cancel_key}")
            if result:
                logger.info(
                    "Cancellation key cleared",
                    extra={"cancel_key": cancel_key}
                )
            return bool(result)
        except Exception as e:
            logger.error(
                "Error clearing cancellation key",
                extra={"cancel_key": cancel_key, "error": str(e)}
            )
            return False

    # -------------------------------------------------------------------------
    # Asynchronous API
    # -------------------------------------------------------------------------
    
    async def is_cancelled_async(self, cancel_key: str) -> bool:
        """Check if a cancellation key exists in Redis (async).

        Args:
            cancel_key: The cancellation key to check (typically thread_id).

        Returns:
            bool: True if cancellation was requested, False otherwise.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        client = await self._get_async_client()
        if not client:
            return False
        
        try:
            result = await client.exists(f"{self.prefix}{cancel_key}")
            if result:
                logger.info(
                    "Cancellation detected (async)",
                    extra={"cancel_key": cancel_key}
                )
            return bool(result)
        except Exception as e:
            logger.error(
                "Error checking cancellation key (async)",
                extra={"cancel_key": cancel_key, "error": str(e)}
            )
            return False
    
    async def set_cancel_async(self, cancel_key: str, ttl: int = DEFAULT_CANCEL_TTL) -> bool:
        """Set a cancellation key in Redis with automatic expiration (async).

        Args:
            cancel_key: The cancellation key to set (typically thread_id).
            ttl: Time-to-live in seconds before key auto-expires (default: 60).

        Returns:
            bool: True if key was set successfully, False on error.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        client = await self._get_async_client()
        if not client:
            return False
        
        try:
            await client.setex(f"{self.prefix}{cancel_key}", ttl, "1")
            logger.info(
                "Cancellation key set (async)",
                extra={"cancel_key": cancel_key, "ttl": ttl}
            )
            return True
        except Exception as e:
            logger.error(
                "Error setting cancellation key (async)",
                extra={"cancel_key": cancel_key, "error": str(e)}
            )
            return False
    
    async def clear_cancel_async(self, cancel_key: str) -> bool:
        """Clear a cancellation key from Redis (async).

        Args:
            cancel_key: The cancellation key to clear (typically thread_id).

        Returns:
            bool: True if key existed and was deleted, False otherwise.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        client = await self._get_async_client()
        if not client:
            return False
        
        try:
            result = await client.delete(f"{self.prefix}{cancel_key}")
            if result:
                logger.info(
                    "Cancellation key cleared (async)",
                    extra={"cancel_key": cancel_key}
                )
            return bool(result)
        except Exception as e:
            logger.error(
                "Error clearing cancellation key (async)",
                extra={"cancel_key": cancel_key, "error": str(e)}
            )
            return False

    async def close_async(self) -> None:
        """Close the async Redis connection properly.
        
        Should be called during application shutdown to prevent connection leaks.
        
        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
            self._async_connected = False
            logger.info("Redis async connection closed")


# Global instance - use this for cancellation operations
redis_cancel_helper = RedisCancelHelper()


__all__ = [
    "RedisCancelHelper",
    "redis_cancel_helper",
    "CANCEL_KEY_PREFIX",
    "DEFAULT_CANCEL_TTL",
]