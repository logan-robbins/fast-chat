"""
Shared HTTP client with connection pooling for backend communication.

Provides a long-lived httpx AsyncClient with proper connection pooling,
timeouts, and resource management. Using a shared client significantly
improves performance by reusing connections and avoiding TLS handshake
overhead on each request.

Configuration Environment Variables:
    CHAT_APP_URL: Backend chat-app service URL
        Default: http://localhost:8001/v1/chat/completions
    
    HTTP_MAX_CONNECTIONS: Maximum total connections in pool
        Default: 100
    
    HTTP_MAX_KEEPALIVE: Maximum keep-alive connections
        Default: 20
    
    HTTP_TIMEOUT_CONNECT: Connection timeout in seconds
        Default: 5.0
    
    HTTP_TIMEOUT_READ: Read timeout in seconds
        Default: 120.0
    
    HTTP_TIMEOUT_WRITE: Write timeout in seconds
        Default: 30.0
    
    HTTP_TIMEOUT_POOL: Pool timeout in seconds
        Default: 10.0

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import os
import structlog
from typing import Optional
from contextlib import asynccontextmanager

import httpx

logger = structlog.get_logger(__name__)

# ============================================================================
# Configuration
# ============================================================================

CHAT_APP_URL: str = os.getenv(
    "CHAT_APP_URL",
    "http://localhost:8001/v1/chat/completions"
)

# Connection pool settings
HTTP_MAX_CONNECTIONS: int = int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))
HTTP_MAX_KEEPALIVE: int = int(os.getenv("HTTP_MAX_KEEPALIVE", "20"))

# Timeout settings (seconds)
HTTP_TIMEOUT_CONNECT: float = float(os.getenv("HTTP_TIMEOUT_CONNECT", "5.0"))
HTTP_TIMEOUT_READ: float = float(os.getenv("HTTP_TIMEOUT_READ", "120.0"))
HTTP_TIMEOUT_WRITE: float = float(os.getenv("HTTP_TIMEOUT_WRITE", "30.0"))
HTTP_TIMEOUT_POOL: float = float(os.getenv("HTTP_TIMEOUT_POOL", "10.0"))


# ============================================================================
# Client Configuration
# ============================================================================

def _create_limits() -> httpx.Limits:
    """
    Create connection pool limits configuration.
    
    Returns:
        httpx.Limits: Configured connection limits
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return httpx.Limits(
        max_connections=HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=HTTP_MAX_KEEPALIVE,
        keepalive_expiry=5.0,  # Close idle connections after 5 seconds
    )


def _create_timeout() -> httpx.Timeout:
    """
    Create timeout configuration for HTTP requests.
    
    Returns:
        httpx.Timeout: Configured timeout settings
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return httpx.Timeout(
        connect=HTTP_TIMEOUT_CONNECT,
        read=HTTP_TIMEOUT_READ,
        write=HTTP_TIMEOUT_WRITE,
        pool=HTTP_TIMEOUT_POOL,
    )


# ============================================================================
# Client Singleton
# ============================================================================

# Global client instance - initialized lazily
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    """
    Get the shared HTTP client instance.
    
    Creates the client on first call (lazy initialization).
    The client is reused across all requests for connection pooling.
    
    Returns:
        httpx.AsyncClient: Shared client instance
        
    Raises:
        RuntimeError: If called after client has been closed
        
    Note:
        Call close_client() during application shutdown to properly
        release all connections.
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    global _client
    
    if _client is None:
        logger.info(
            "http_client.init",
            max_connections=HTTP_MAX_CONNECTIONS,
            max_keepalive=HTTP_MAX_KEEPALIVE,
        )
        _client = httpx.AsyncClient(
            limits=_create_limits(),
            timeout=_create_timeout(),
            http2=True,  # Enable HTTP/2 for better performance
            trust_env=False,  # Do not route internal service calls via proxy env vars
        )
    
    return _client


async def close_client() -> None:
    """
    Close the shared HTTP client and release all connections.
    
    Should be called during application shutdown to ensure clean
    resource cleanup.
    
    Returns:
        None
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    global _client
    
    if _client is not None:
        logger.info("Closing HTTP client")
        await _client.aclose()
        _client = None
        logger.info("HTTP client closed")


# ============================================================================
# Convenience Functions
# ============================================================================

async def post_to_backend(
    endpoint: str,
    json_data: dict,
    timeout: Optional[float] = None,
) -> httpx.Response:
    """
    Make a POST request to the backend chat-app service.
    
    Uses the shared client for connection pooling.
    
    Args:
        endpoint: API endpoint (appended to CHAT_APP_URL base)
        json_data: JSON payload to send
        timeout: Optional request-specific timeout override
        
    Returns:
        httpx.Response: The response from the backend
        
    Raises:
        httpx.HTTPError: If the request fails
        
    Example:
        response = await post_to_backend(
            "/v1/chat/completions",
            {"model": "gpt-4o", "messages": [...]}
        )
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    client = await get_client()
    
    # Build full URL
    base_url = CHAT_APP_URL.rstrip("/")
    url = f"{base_url.rsplit('/v1', 1)[0]}{endpoint}" if endpoint.startswith("/v1") else f"{base_url}{endpoint}"
    
    # Use default or override timeout
    request_timeout = httpx.Timeout(timeout) if timeout else None
    
    return await client.post(url, json=json_data, timeout=request_timeout)


@asynccontextmanager
async def stream_from_backend(
    endpoint: str,
    json_data: dict,
    timeout: Optional[float] = None,
):
    """
    Stream a POST request from the backend chat-app service.
    
    Uses the shared client for connection pooling with streaming support.
    
    Args:
        endpoint: API endpoint
        json_data: JSON payload to send
        timeout: Optional request-specific timeout override
        
    Yields:
        httpx.Response: Streaming response context
        
    Example:
        async with stream_from_backend("/v1/chat/completions", payload) as response:
            async for line in response.aiter_lines():
                print(line)
                
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    client = await get_client()
    
    # Build full URL (same logic as post_to_backend)
    base_url = CHAT_APP_URL.rstrip("/")
    url = f"{base_url.rsplit('/v1', 1)[0]}{endpoint}" if endpoint.startswith("/v1") else f"{base_url}{endpoint}"
    
    request_timeout = httpx.Timeout(timeout) if timeout else None
    
    async with client.stream("POST", url, json=json_data, timeout=request_timeout) as response:
        yield response
