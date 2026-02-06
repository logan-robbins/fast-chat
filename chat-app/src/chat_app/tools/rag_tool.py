"""RAG search tool using chat-api (BFF) for vector operations.

Provides the SearchDocumentContentTool which queries the vector store
for semantically similar document chunks. Supports thread-scoped collections
for multi-tenant isolation.

Architecture:
    The tool calls chat-api's /v1/search endpoint for vector operations.
    chat-api owns the vector store (ChromaDB), making it easy to swap
    backends without changing chat-app. Collections are passed via
    RunnableConfig["configurable"]["vector_collections"].

Multi-Tenancy:
    Collections are scoped per-thread to prevent cross-tenant information
    leakage. The tool extracts vector collections from:
    RunnableConfig["configurable"]["vector_collections"]

Status Streaming (ChatGPT/Claude 2025/2026 patterns):
    The async implementation emits user-friendly status events:
    - "Looking through your documents..." at start
    - "Searching collection 1/3: financial_docs" for progress
    - "Found 5 relevant documents" on completion
    
    Uses chat_app.status_streaming for consistent status format.

Dependencies:
- chat_app.services.bff_client: HTTP client for chat-api
- langchain_core>=0.3.0: For BaseTool, RunnableConfig
- pydantic>=2.5.0: For input schema validation
- chat_app.status_streaming: For status event emission

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default collection to search when none specified
DEFAULT_COLLECTION: str = "documents"

# Maximum results that can be returned
MAX_RESULTS: int = 20


class SearchDocumentContentInput(BaseModel):
    """Input schema for the RAG search tool."""

    query: str = Field(
        ..., 
        min_length=1,
        description="Query string to search document content."
    )
    limit: int = Field(
        default=5, 
        ge=1, 
        le=MAX_RESULTS, 
        description="Maximum results to return."
    )


class SearchDocumentContentTool(BaseTool):
    """Search uploaded document content using vector store semantic search.

    LangChain tool that performs semantic search over user-uploaded documents
    stored in the vector database. Respects thread-scoped collections for
    multi-tenant data isolation.

    The tool is designed to work with RunnableConfig to receive collection
    names at runtime, enabling per-request collection scoping.

    Attributes:
        name: Tool name exposed to LLM ("search_document_content")
        description: Human-readable description for LLM routing
        args_schema: Pydantic model for input validation

    Example:
        >>> tool = SearchDocumentContentTool()
        >>> config = {"configurable": {"vector_collections": ["thread_abc"]}}
        >>> result = tool.invoke({"query": "revenue projections", "limit": 5}, config=config)

    Multi-Tenancy:
        Vector collections are extracted from config["configurable"]["vector_collections"].
        If no collections are specified, falls back to DEFAULT_COLLECTION.

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """

    name: ClassVar[str] = "search_document_content"
    description: ClassVar[str] = (
        "Search uploaded document content for relevant information. "
        "Use this when you need to find specific information from user's documents."
    )
    args_schema: ClassVar[type[BaseModel]] = SearchDocumentContentInput

    def _run(
        self,
        query: str,
        limit: int = 5,
        *,
        config: Optional[RunnableConfig] = None,
        **_: object,
    ) -> str:
        """Execute synchronous document search (wraps async implementation).

        Handles the complexity of running async code from sync context,
        including when called from within an existing event loop.

        Args:
            query: Semantic search query string.
            limit: Maximum number of results to return (1-20, default 5).
            config: LangGraph config containing vector_collections.

        Returns:
            str: Formatted search results or error message.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        collections = self._extract_collections(config)
        
        logger.debug(
            "Starting sync document search",
            extra={
                "query_preview": query[:50],
                "limit": limit,
                "collection_count": len(collections)
            }
        )
        
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to use thread pool
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._run_sync, query, limit, collections
                    )
                    return future.result(timeout=30)
            except RuntimeError:
                # No running loop, we can create one
                return asyncio.run(self._search(query, limit, collections))
                
        except Exception as exc:
            logger.error(
                "Sync document search failed",
                extra={
                    "query_preview": query[:50],
                    "error": str(exc),
                    "error_type": type(exc).__name__
                },
                exc_info=True
            )
            return f"Search failed: {exc}"

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        *,
        config: Optional[RunnableConfig] = None,
        **_: object,
    ) -> str:
        """Execute asynchronous document search with progress streaming.

        Emits user-friendly status events via chat_app.status_streaming:
        - "Looking through your documents..." at start
        - "Searching collection N/M: collection_name" during search
        - "Found X relevant documents" on completion

        Args:
            query: Semantic search query string.
            limit: Maximum number of results to return (1-20, default 5).
            config: LangGraph config containing vector_collections.

        Returns:
            str: Formatted search results or error message.

        Last Grunted: 02/04/2026 07:30:00 PM PST
        """
        collections = self._extract_collections(config)
        
        logger.debug(
            "Starting async document search",
            extra={
                "query_preview": query[:50],
                "limit": limit,
                "collections": collections
            }
        )
        
        # Emit starting status (ChatGPT/Claude-style "Looking through your documents...")
        try:
            from chat_app.status_streaming import emit_tool_start
            emit_tool_start(
                tool_name="search_document_content",
                agent_name="knowledge_base",
                details={"query": query[:100], "collections": len(collections)}
            )
        except Exception:
            pass  # Status emission failures should not break tool execution
        
        result = await self._search(query, limit, collections)
        return result

    def _run_sync(self, query: str, limit: int, collections: List[str]) -> str:
        """Run search in a fresh event loop for sync contexts.

        Args:
            query: Search query string.
            limit: Maximum results to return.
            collections: Vector collection names to search.

        Returns:
            str: Formatted search results.

        Last Grunted: 02/04/2026 07:30:00 PM PST
        """
        # Emit starting status before running async search
        try:
            from chat_app.status_streaming import emit_tool_start
            emit_tool_start(
                tool_name="search_document_content",
                agent_name="knowledge_base",
                details={"query": query[:100], "collections": len(collections)}
            )
        except Exception:
            pass
        
        return asyncio.run(self._search(query, limit, collections))

    @staticmethod
    def _extract_collections(config: Optional[RunnableConfig]) -> List[str]:
        """Extract vector collection names from RunnableConfig.

        Tries multiple sources in priority order:
        1. Explicit vector_collections list in configurable
        2. Derived from thread_id (thread_{id} collection pattern)
        3. Empty list (falls back to DEFAULT_COLLECTION in _search)

        Args:
            config: LangGraph config dict with configurable options.

        Returns:
            List[str]: List of collection names.
        """
        if config is None:
            return []
        
        configurable = config.get("configurable") if isinstance(config, dict) else {}
        
        if not isinstance(configurable, dict):
            return []
            
        # Try explicit vector_collections first
        collections = configurable.get("vector_collections", [])
        result = [c for c in (collections or []) if c]
        
        # Fall back to deriving from thread_id
        if not result:
            thread_id = configurable.get("thread_id")
            if thread_id:
                result = [f"thread_{thread_id}"]
        
        return result

    async def _search(
        self,
        query: str,
        limit: int,
        collections: List[str],
    ) -> str:
        """Perform semantic search via chat-api's /v1/search endpoint.

        Calls the BFF (chat-api) which owns the vector store. This
        architecture allows easy backend swaps (ChromaDB -> Pinecone).

        Args:
            query: Natural language search query.
            limit: Maximum number of results (clamped to 1-MAX_RESULTS).
            collections: Collection names to search. Falls back to
                DEFAULT_COLLECTION if empty.

        Returns:
            str: Formatted results string or "No relevant documents found."

        Note:
            HTTP errors are caught and returned as error messages.

        Last Grunted: 02/05/2026 12:00:00 PM UTC
        """
        from chat_app.services.bff_client import search_documents
        
        # Clamp limit to valid range
        limit = max(1, min(limit, MAX_RESULTS))
        
        # Use default collection if none specified
        search_collections = collections if collections else [DEFAULT_COLLECTION]

        # Emit progress status (ChatGPT/Claude-style)
        try:
            from chat_app.status_streaming import emit_tool_progress
            emit_tool_progress(
                tool_name="search_document_content",
                message=f"Searching {len(search_collections)} collection(s)...",
                current=1,
                total=1,
                agent_name="knowledge_base",
                details={"collections": search_collections}
            )
        except Exception:
            pass  # Status emission failures should not break search
        
        try:
            results = await search_documents(
                query=query,
                collections=search_collections,
                limit=limit,
            )
        except Exception as e:
            logger.error(
                "Search request failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query_preview": query[:50],
                }
            )
            return f"Search failed: {e}"
        
        # Convert to internal format
        all_results: List[Dict[str, Any]] = []
        for result in results:
            raw_result: Dict[str, Any] = {
                "collection": result.collection,
                "filename": result.filename,
                "text": result.text,
                "text_preview": result.text[:500] if result.text else "",
                "score": result.score,
                "metadata": result.metadata,
                "document_id": result.document_id,
            }
            all_results.append(raw_result)

        # Emit completion status (ChatGPT/Claude-style "Found X relevant documents")
        try:
            from chat_app.status_streaming import emit_tool_complete
            emit_tool_complete(
                tool_name="search_document_content",
                agent_name="knowledge_base",
                details={"results_count": len(all_results)}
            )
        except Exception:
            pass

        if not all_results:
            logger.debug(
                "No documents found",
                extra={
                    "query_preview": query[:50],
                    "collections_searched": search_collections
                }
            )
            return "No relevant documents found."

        logger.debug(
            "Document search completed",
            extra={
                "results_count": len(all_results),
                "top_score": all_results[0]["score"] if all_results else 0
            }
        )

        return self._format_results(all_results)

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        """Format search results into human-readable string for LLM.

        Args:
            results: List of result dicts with keys:
                - filename (str): Source document name
                - score (float): Similarity score 0.0-1.0
                - text_preview (str): Matched content preview

        Returns:
            str: Numbered list with filename, relevance %, and content preview.

        Last Grunted: 02/04/2026 06:30:00 PM PST
        """
        lines: List[str] = ["Relevant documents found:\n"]
        
        for i, result in enumerate(results, 1):
            filename = result.get('filename', 'unknown')
            score = result.get('score', 0)
            preview = result.get('text_preview', result.get('text', '')[:500])
            
            lines.append(f"{i}. {filename}")
            lines.append(f"   Relevance: {score:.0%}")
            lines.append(f"   Content: {preview[:200]}...")
            lines.append("")
        
        return "\n".join(lines)


# Default instance for agent wiring
search_document_content = SearchDocumentContentTool()

__all__ = [
    "SearchDocumentContentTool",
    "search_document_content",
    "DEFAULT_COLLECTION",
    "MAX_RESULTS",
]
