"""
Vector search API router.

Provides semantic search over documents stored in ChromaDB. This endpoint
acts as the single gateway for all vector operations, abstracting the
storage backend from consumers (chat-app).

Endpoints:
    - POST /v1/search - Semantic search across document collections

Retrieval Features:
    - Score thresholding: Optional minimum similarity score to filter out
      low-relevance results (recommended: 0.3-0.7 depending on use case)
    - Metadata filtering: Optional ChromaDB where clause to narrow search
      scope by document metadata (filename, content_type, etc.)
    - Async embedding: Query embedding generation is offloaded to a thread
      to avoid blocking the event loop

Architecture:
    chat-api owns the vector store. chat-app and other consumers call this
    endpoint for RAG queries. When migrating to a managed vector DB (Pinecone,
    Weaviate, etc.), only this module needs to change.

Future Improvements (documented from 2026 research):
    - Hybrid search: Combine dense (embedding) + sparse (BM25) retrieval
      for 20-40% accuracy improvement (Anthropic contextual retrieval)
    - Reranking: Cross-encoder reranking step (Cohere Rerank, BGE Reranker)
      for ~33-40% accuracy improvement on top of initial retrieval

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
import asyncio
import os

import structlog
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.services.errors import internal_error

logger = structlog.get_logger(__name__)

router = APIRouter()


def _embed_query_compat(query: str, model: str) -> List[float]:
    """Compatibility embedding call for providers that reject `encoding_format`."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LITELLM_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    )
    response = client.embeddings.create(model=model, input=query)
    return response.data[0].embedding


# ============================================================================
# Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    """Request body for semantic search.
    
    Attributes:
        query: Natural language search query
        collections: List of collection names to search (default: ["documents"])
        limit: Maximum number of results (1-50, default 5)
        score_threshold: Optional minimum similarity score (0.0-1.0). Results
            below this threshold are excluded. Recommended: 0.3 for broad recall,
            0.5 for balanced precision/recall, 0.7 for high precision.
        where: Optional ChromaDB metadata filter dict. Narrows search to
            documents matching the filter criteria.
            Examples: {"filename": "report.pdf"}, {"content_type": "application/pdf"}
    """
    query: str = Field(..., min_length=1, description="Search query string")
    collections: List[str] = Field(
        default=["documents"],
        description="Collection names to search"
    )
    limit: int = Field(default=5, ge=1, le=50, description="Maximum results")
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0). None returns all results.",
    )
    where: Optional[Dict[str, Any]] = Field(
        default=None,
        description="ChromaDB metadata filter (e.g. {\"filename\": \"report.pdf\"})",
    )


class SearchResult(BaseModel):
    """Single search result with metadata.
    
    Attributes:
        document_id: Unique document/chunk ID
        collection: Source collection name
        filename: Original filename
        text: Full text content of the chunk
        score: Similarity score (0.0 to 1.0, higher is better)
        metadata: Additional metadata from the document
    """
    document_id: str
    collection: str
    filename: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response containing search results.
    
    Attributes:
        object: Always "list"
        results: List of search results sorted by relevance
        total: Total number of results returned
    """
    object: str = "list"
    results: List[SearchResult]
    total: int


# ============================================================================
# Helper Functions
# ============================================================================

def _get_docproc():
    """Import docproc lazily to avoid startup failures if not installed.
    
    Returns:
        Tuple of (get_embeddings_model, get_vector_store) functions
        
    Raises:
        RuntimeError: If docproc is not installed
    """
    try:
        from docproc import get_embeddings_model, get_vector_store
        return get_embeddings_model, get_vector_store
    except ImportError as e:
        logger.error("docproc library not installed")
        raise RuntimeError("docproc library required for search") from e


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/v1/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Semantic search across document collections.
    
    Performs vector similarity search using embeddings. Searches across
    specified collections and returns the most relevant document chunks.
    
    Args:
        request: Search parameters (query, collections, limit)
        
    Returns:
        SearchResponse: List of relevant document chunks with scores
        
    Example Request:
        POST /v1/search
        {
            "query": "What are the revenue projections?",
            "collections": ["thread_abc123"],
            "limit": 5
        }
        
    Example Response:
        {
            "object": "list",
            "results": [
                {
                    "document_id": "chunk_123",
                    "collection": "thread_abc123",
                    "filename": "financial_report.pdf",
                    "text": "Revenue is projected to grow 15%...",
                    "score": 0.89,
                    "metadata": {"page": 3}
                }
            ],
            "total": 1
        }
    """
    logger.info(
        "search.request",
        query_preview=request.query[:50],
        collections=request.collections,
        limit=request.limit,
    )
    
    try:
        get_embeddings_model, get_vector_store = _get_docproc()
    except RuntimeError as e:
        return internal_error(str(e))
    
    # Generate query embedding (offloaded to thread to avoid blocking event loop)
    try:
        embeddings = get_embeddings_model()
        query_embedding = await asyncio.to_thread(
            embeddings.embed_query, request.query
        )
    except Exception as e:
        error_text = str(e)
        logger.error(
            "search.embedding_failed",
            error=error_text,
            query_preview=request.query[:50],
        )

        # Compatibility fallback for OpenAI-compatible providers that do not
        # support the `encoding_format` parameter used by some SDK paths.
        if "Unknown parameter: 'encoding_format'" in error_text:
            try:
                embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
                query_embedding = await asyncio.to_thread(
                    _embed_query_compat,
                    request.query,
                    embedding_model,
                )
                logger.info(
                    "search.embedding_fallback_success",
                    model=embedding_model,
                    query_preview=request.query[:50],
                )
            except Exception as fallback_error:
                return internal_error(
                    f"Failed to generate query embedding: {fallback_error}"
                )
        else:
            return internal_error(f"Failed to generate query embedding: {e}")
    
    all_results: List[SearchResult] = []
    
    # Search each collection
    for collection_name in request.collections:
        try:
            store = get_vector_store(collection_name)
            await store.initialize()
            
            results = await store.search_similar(
                query_embedding,
                limit=request.limit,
                where=request.where,
                score_threshold=request.score_threshold,
            )
            
            for result in results:
                search_result = SearchResult(
                    document_id=result.get("id", "unknown"),
                    collection=collection_name,
                    filename=result.get("metadata", {}).get("filename", "unknown"),
                    text=result.get("text", ""),
                    score=result.get("similarity_score", 0.0),
                    metadata=result.get("metadata", {})
                )
                all_results.append(search_result)
                
        except Exception as e:
            logger.warning(
                "search.collection_failed",
                collection=collection_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Continue searching other collections
            continue
    
    # Sort by score (highest first) and limit
    all_results.sort(key=lambda x: x.score, reverse=True)
    all_results = all_results[:request.limit]
    
    logger.info(
        "search.completed",
        results_count=len(all_results),
        top_score=all_results[0].score if all_results else 0,
    )
    
    return SearchResponse(
        object="list",
        results=all_results,
        total=len(all_results)
    )
