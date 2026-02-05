"""
HTTP client for chat-api (BFF) communication.

Provides async HTTP client for calling chat-api endpoints, particularly
the vector search endpoint. Uses httpx with connection pooling.

Architecture:
    chat-app (orchestrator) calls chat-api (BFF/middleware) for:
    - Vector search (POST /v1/search)
    - Future: file operations, etc.
    
    This allows chat-api to own the vector store, making it easy to
    swap backends (ChromaDB -> Pinecone/Weaviate) without changing chat-app.

Configuration:
    CHAT_API_URL: Base URL for chat-api service
        Default: http://localhost:8000

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

CHAT_API_URL: str = os.getenv("CHAT_API_URL", "http://localhost:8000")

# Timeout settings
HTTP_TIMEOUT_CONNECT: float = 5.0
HTTP_TIMEOUT_READ: float = 30.0

# Module-level client (lazy initialization)
_client: Optional[httpx.AsyncClient] = None


# ============================================================================
# Client Management
# ============================================================================

async def get_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client.
    
    Returns:
        httpx.AsyncClient: Shared client with connection pooling
    """
    global _client
    
    if _client is None:
        logger.info(
            "Initializing BFF client",
            extra={"base_url": CHAT_API_URL}
        )
        _client = httpx.AsyncClient(
            base_url=CHAT_API_URL,
            timeout=httpx.Timeout(
                connect=HTTP_TIMEOUT_CONNECT,
                read=HTTP_TIMEOUT_READ,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=10,
            ),
        )
    
    return _client


async def close_client() -> None:
    """Close the HTTP client and release connections."""
    global _client
    
    if _client is not None:
        logger.info("Closing BFF client")
        await _client.aclose()
        _client = None


# ============================================================================
# Search API
# ============================================================================

class SearchResult:
    """Single search result from BFF."""
    
    def __init__(
        self,
        document_id: str,
        collection: str,
        filename: str,
        text: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        self.document_id = document_id
        self.collection = collection
        self.filename = filename
        self.text = text
        self.score = score
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            "id": self.document_id,
            "collection": self.collection,
            "filename": self.filename,
            "text": self.text,
            "similarity_score": self.score,
            "metadata": self.metadata,
        }


async def search_documents(
    query: str,
    collections: Optional[List[str]] = None,
    limit: int = 5,
) -> List[SearchResult]:
    """
    Search documents via chat-api's vector search endpoint.
    
    Args:
        query: Natural language search query
        collections: Collection names to search (default: ["documents"])
        limit: Maximum results to return (default: 5)
        
    Returns:
        List[SearchResult]: Search results sorted by relevance
        
    Raises:
        httpx.HTTPError: If the request fails
        ValueError: If the response is invalid
    """
    client = await get_client()
    
    payload = {
        "query": query,
        "collections": collections or ["documents"],
        "limit": limit,
    }
    
    logger.debug(
        "Searching documents via BFF",
        extra={
            "query_preview": query[:50],
            "collections": payload["collections"],
            "limit": limit,
        }
    )
    
    try:
        response = await client.post("/v1/search", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        for item in data.get("results", []):
            result = SearchResult(
                document_id=item.get("document_id", "unknown"),
                collection=item.get("collection", "unknown"),
                filename=item.get("filename", "unknown"),
                text=item.get("text", ""),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}),
            )
            results.append(result)
        
        logger.debug(
            "Search completed",
            extra={"results_count": len(results)}
        )
        
        return results
        
    except httpx.HTTPStatusError as e:
        logger.error(
            "Search request failed",
            extra={
                "status_code": e.response.status_code,
                "detail": e.response.text[:200] if e.response.text else None,
            }
        )
        raise
    except Exception as e:
        logger.error(
            "Search request error",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        raise
