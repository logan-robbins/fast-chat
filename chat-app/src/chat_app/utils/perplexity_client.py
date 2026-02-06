"""Perplexity API client utilities for web search functionality.

Provides the call_perplexity_api function for making async requests to
Perplexity's chat completions endpoint. Uses httpx for non-blocking HTTP
calls, consistent with the rest of the async codebase.

Supported Perplexity models (as of January 2026):
- sonar: Fast, lightweight search model with citations
- sonar-pro: Advanced model for complex queries, more citations
- sonar-reasoning: Enhanced reasoning with search (default)

Configuration via environment variables:
- PERPLEXITY_API_KEY: Required API key
- PERPLEXITY_API_ENDPOINT: API URL (default: https://api.perplexity.ai/chat/completions)
- PERPLEXITY_MODEL: Model name (default: sonar-reasoning)
- PERPLEXITY_MAX_TOKENS: Max response tokens (default: 2000)
- PERPLEXITY_TEMPERATURE: Response temperature 0.0-1.0 (default: 0.2)
- PERPLEXITY_TIMEOUT: Request timeout in seconds (default: 30)

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

import httpx

logger = logging.getLogger(__name__)

# Module-level async client (lazy initialization with connection pooling)
_client: Optional[httpx.AsyncClient] = None


class SearchResult(TypedDict):
    """Structure for a single search result from Perplexity API."""
    title: str
    url: str
    date: Optional[str]


class PerplexityResponse(TypedDict):
    """Structure for Perplexity API response with content and citations."""
    content: str
    search_results: List[SearchResult]


async def _get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client.

    Uses a module-level singleton with connection pooling for efficient
    reuse across multiple API calls.

    Returns:
        httpx.AsyncClient: Shared client instance.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    global _client
    if _client is None:
        timeout = float(os.getenv("PERPLEXITY_TIMEOUT", "30"))
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,
                read=timeout,
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=5,
            ),
        )
    return _client


async def close_client() -> None:
    """Close the shared HTTP client and release connections.

    Should be called during application shutdown to prevent connection leaks.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Perplexity HTTP client closed")


async def call_perplexity_api(messages: List[Dict[str, str]]) -> PerplexityResponse:
    """Call the Perplexity API chat completions endpoint asynchronously.

    Makes a non-blocking HTTP request to Perplexity's API with the provided
    messages. The API performs web search and returns an AI-synthesized
    response with source citations.

    Args:
        messages: List of message dictionaries with 'role' and 'content'.
            Typically a single user message with the search query.
            Example: [{"role": "user", "content": "What is Python?"}]

    Returns:
        PerplexityResponse: TypedDict with:
            - content (str): The AI-generated response text
            - search_results (List[SearchResult]): List of source citations
              with 'title', 'url', and optional 'date' fields

    Raises:
        ValueError: If PERPLEXITY_API_KEY environment variable is not set.

    Note:
        Errors are handled gracefully -- API errors, timeouts, and connection
        failures return a PerplexityResponse with an error message in content
        and an empty search_results list.

    Example:
        >>> response = await call_perplexity_api(
        ...     [{"role": "user", "content": "Python history"}]
        ... )
        >>> print(response['content'])
        "Python was created by Guido van Rossum..."

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY environment variable not set")
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Please add PERPLEXITY_API_KEY=pplx-your-key-here to your .env file"
        )

    url = os.getenv(
        "PERPLEXITY_API_ENDPOINT",
        "https://api.perplexity.ai/chat/completions",
    )

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": os.getenv("PERPLEXITY_MODEL", "sonar-reasoning"),
        "messages": messages,
        "max_tokens": int(os.getenv("PERPLEXITY_MAX_TOKENS", "2000")),
        "temperature": float(os.getenv("PERPLEXITY_TEMPERATURE", "0.2")),
    }

    logger.debug(
        "Preparing Perplexity API request",
        extra={
            "url": url,
            "model": payload["model"],
            "message_count": len(messages),
            "message_preview": messages[0].get("content", "")[:200] if messages else None,
        },
    )

    client = await _get_client()

    try:
        logger.debug("Sending async request to Perplexity API")
        response = await client.post(url, json=payload, headers=headers)

        logger.debug(
            "Received response from Perplexity API",
            extra={
                "status_code": response.status_code,
                "response_size": len(response.content) if response.content else 0,
            },
        )

        # Check for HTTP errors
        if response.status_code != 200:
            return _handle_http_error(response)

        result: Dict[str, Any] = response.json()
        content: str = result["choices"][0]["message"]["content"]
        search_results: List[SearchResult] = result.get("search_results", [])

        logger.debug(
            "Successfully parsed Perplexity API response",
            extra={
                "content_length": len(content),
                "search_results_count": len(search_results),
            },
        )

        return PerplexityResponse(
            content=content,
            search_results=search_results,
        )

    except httpx.TimeoutException as e:
        timeout_seconds = int(os.getenv("PERPLEXITY_TIMEOUT", "30"))
        logger.error(
            "Perplexity API request timed out",
            extra={"timeout": timeout_seconds, "error": str(e)},
        )
        return PerplexityResponse(
            content=f"Web search timed out after {timeout_seconds} seconds. Please try again.",
            search_results=[],
        )
    except httpx.ConnectError as e:
        logger.error(
            "Perplexity API connection error",
            extra={"error": str(e), "url": url},
        )
        return PerplexityResponse(
            content="Web search failed: Connection error. Please check your internet connection.",
            search_results=[],
        )
    except httpx.HTTPError as e:
        logger.error(
            "Perplexity API HTTP error",
            extra={
                "error_type": type(e).__name__,
                "error": str(e),
            },
            exc_info=True,
        )
        return PerplexityResponse(
            content=f"Web search failed: {type(e).__name__}: {e}",
            search_results=[],
        )
    except KeyError as e:
        logger.error(
            "Unexpected response format from Perplexity API",
            extra={"missing_key": str(e)},
            exc_info=True,
        )
        return PerplexityResponse(
            content=f"Unexpected response format from Perplexity API: Missing key {e}",
            search_results=[],
        )
    except Exception as e:
        logger.error(
            "Unexpected error calling Perplexity API",
            extra={"error_type": type(e).__name__, "error": str(e)},
            exc_info=True,
        )
        return PerplexityResponse(
            content=f"Error calling Perplexity API: {type(e).__name__}: {e}",
            search_results=[],
        )


def _handle_http_error(response: httpx.Response) -> PerplexityResponse:
    """Extract error details from a non-200 Perplexity API response.

    Args:
        response: The httpx response with a non-200 status code.

    Returns:
        PerplexityResponse with error message in content.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    try:
        error_data: Dict[str, Any] = response.json()
        error_message = error_data.get("message", f"HTTP {response.status_code}")
        error_type = error_data.get("type", "API Error")

        logger.error(
            "Perplexity API returned error",
            extra={
                "status_code": response.status_code,
                "error_type": error_type,
                "error_message": error_message,
            },
        )
        return PerplexityResponse(
            content=f"Perplexity API error ({error_type}): {error_message}",
            search_results=[],
        )
    except (ValueError, KeyError):
        logger.error(
            "Perplexity API returned non-JSON error response",
            extra={
                "status_code": response.status_code,
                "response_preview": response.text[:500] if response.text else None,
            },
        )
        return PerplexityResponse(
            content=f"Perplexity API error: HTTP {response.status_code}",
            search_results=[],
        )


__all__ = [
    "call_perplexity_api",
    "close_client",
    "PerplexityResponse",
    "SearchResult",
]
