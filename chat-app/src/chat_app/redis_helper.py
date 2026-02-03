"""Redis helper for managing inflight request cancellation.

Follows the same pattern as mistral and ollama services, providing a
key-value based cancellation mechanism using Redis.

The pattern works by:
1. Client sets a cancellation key in Redis when user wants to cancel
2. Service checks for cancellation key periodically during streaming
3. If key exists, service stops streaming and clears the key

Environment Variables:
    REDIS_HOST: Redis server hostname (default: localhost)
    REDIS_PORT: Redis server port (default: 6379)
    REDIS_USERNAME: Redis username (default: "default")
    REDIS_PASSWORD: Redis password (default: "")

Last Grunted: 02/03/2026 03:15:00 PM PST
"""

import os
import logging
import redis
from typing import Optional

logger = logging.getLogger(__name__)


class RedisCancelHelper:
    """Helper class for managing cancellation of inflight requests via Redis.

    Provides a simple key-value mechanism for cancelling long-running LLM
    requests. Clients set a cancellation key, and the streaming service
    periodically checks for it during processing.

    The pattern is:
    1. Client calls set_cancel(thread_id) when user wants to cancel
    2. Service calls is_cancelled(thread_id) during streaming loop
    3. If cancelled, service stops streaming and calls clear_cancel(thread_id)

    Attributes:
        redis_host (str): Redis server hostname
        redis_port (int): Redis server port
        redis_username (str): Redis authentication username
        redis_password (str): Redis authentication password
        redis_client (redis.Redis | None): Connected client or None if failed
        prefix (str): Key prefix for cancellation keys ("cancel:")

    Last Grunted: 02/03/2026 03:45:00 PM PST
    """
    
    def __init__(self, redis_host: str = None, redis_port: int = 6379, redis_password: str = None):
        """Initialize Redis connection for cancellation management.

        Args:
            redis_host (str, optional): Redis hostname. Defaults to REDIS_HOST
                env var or "localhost".
            redis_port (int): Redis port. Defaults to 6379.
            redis_password (str, optional): Redis password. Defaults to
                REDIS_PASSWORD env var or empty string.

        Note:
            Connection failure is non-fatal - cancellation support is simply
            disabled and is_cancelled() always returns False.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port
        self.redis_username = os.getenv("REDIS_USERNAME", "default")
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD", "")
        self.redis_client = None
        self.prefix = "cancel:"

        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                username=self.redis_username,
                password=self.redis_password if self.redis_password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"[CHAT-APP] Redis connection established for cancellation support")
        except Exception as e:
            logger.warning(f"[CHAT-APP] Redis connection failed: {e}. Cancellation support disabled.")
            self.redis_client = None
    
    def is_cancelled(self, cancel_key: str) -> bool:
        """Check if a cancellation key exists in Redis.

        Args:
            cancel_key (str): The cancellation key to check (typically thread_id).

        Returns:
            bool: True if cancellation was requested, False otherwise.
                Always returns False if Redis is not connected.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.exists(f"{self.prefix}{cancel_key}")
            if result:
                logger.info(f"[CHAT-APP] Cancellation detected for key: {cancel_key}")
            return bool(result)
        except Exception as e:
            logger.error(f"[CHAT-APP] Error checking cancellation key: {e}")
            return False
    
    def set_cancel(self, cancel_key: str, ttl: int = 60) -> bool:
        """Set a cancellation key in Redis with automatic expiration.

        Args:
            cancel_key (str): The cancellation key to set (typically thread_id).
            ttl (int): Time-to-live in seconds before key auto-expires.
                Defaults to 60 seconds.

        Returns:
            bool: True if key was set successfully, False on error or if
                Redis is not connected.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.setex(f"{self.prefix}{cancel_key}", ttl, "1")
            logger.info(f"[CHAT-APP] Cancellation key set: {cancel_key}")
            return True
        except Exception as e:
            logger.error(f"[CHAT-APP] Error setting cancellation key: {e}")
            return False
    
    def clear_cancel(self, cancel_key: str) -> bool:
        """Clear a cancellation key from Redis.

        Should be called after detecting cancellation to prevent stale keys
        from affecting future requests with the same thread_id.

        Args:
            cancel_key (str): The cancellation key to clear (typically thread_id).

        Returns:
            bool: True if key existed and was deleted, False if key didn't
                exist, on error, or if Redis is not connected.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.delete(f"{self.prefix}{cancel_key}")
            if result:
                logger.info(f"[CHAT-APP] Cancellation key cleared: {cancel_key}")
            return bool(result)
        except Exception as e:
            logger.error(f"[CHAT-APP] Error clearing cancellation key: {e}")
            return False


# Global instance
redis_cancel_helper = RedisCancelHelper()