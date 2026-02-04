"""RAG search tool using docproc library directly.

Provides the SearchDocumentContentTool which queries the vector store
for semantically similar document chunks. Supports thread-scoped collections
for multi-tenant isolation.

The tool extracts vector collections from RunnableConfig to search only
in the user's specific document collections, preventing cross-tenant
information leakage.

Architecture:
    The tool uses the docproc library which abstracts the vector store
    backend (ChromaDB). Collections are passed via
    RunnableConfig["configurable"]["vector_collections"].

Dependencies:
- docproc: Internal library for vector store operations
- langchain_core>=0.3.0: For BaseTool, RunnableConfig, BaseModel
- pydantic>=2.5.0: For input schema validation

Last Grunted: 02/03/2026 03:45:00 PM PST
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchDocumentContentInput(BaseModel):
    """Input schema for the RAG search tool."""

    query: str = Field(..., description="Query string to search document content.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum results to return.")


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

    Last Grunted: 02/03/2026 03:45:00 PM PST
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
        config: RunnableConfig = None,
        **_: object,
    ) -> str:
        """Execute synchronous document search (wraps async implementation).

        Args:
            query (str): Semantic search query string.
            limit (int): Maximum number of results to return (1-20, default 5).
            config (RunnableConfig, optional): LangGraph config containing
                vector_collections in configurable dict.

        Returns:
            str: Formatted search results or error message.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        collections = self._extract_collections(config)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._run_sync, query, limit, collections
                    )
                    return future.result(timeout=30)
            return loop.run_until_complete(self._search(query, limit, collections))
        except RuntimeError:
            return asyncio.run(self._search(query, limit, collections))
        except Exception as exc:
            logger.error(f"Search failed: {exc}", exc_info=True)
            return f"Search failed: {exc}"

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        *,
        config: RunnableConfig = None,
        **_: object,
    ) -> str:
        """Execute asynchronous document search.

        Args:
            query (str): Semantic search query string.
            limit (int): Maximum number of results to return (1-20, default 5).
            config (RunnableConfig, optional): LangGraph config containing
                vector_collections in configurable dict.

        Returns:
            str: Formatted search results or error message.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        collections = self._extract_collections(config)
        return await self._search(query, limit, collections)

    def _run_sync(self, query: str, limit: int, collections: list[str]) -> str:
        """Run search in a fresh event loop for sync contexts.

        Args:
            query (str): Search query string.
            limit (int): Maximum results to return.
            collections (list[str]): Vector collection names to search.

        Returns:
            str: Formatted search results.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        return asyncio.run(self._search(query, limit, collections))

    @staticmethod
    def _extract_collections(config: RunnableConfig | None) -> list[str]:
        """Extract vector collection names from RunnableConfig.

        Args:
            config (RunnableConfig | None): LangGraph config dict with optional
                configurable.vector_collections list.

        Returns:
            list[str]: List of collection names, empty if not configured.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        if config is None:
            return []
        configurable = config.get("configurable") if isinstance(config, dict) else {}
        collections = (
            configurable.get("vector_collections")
            if isinstance(configurable, dict)
            else []
        )
        return [c for c in collections or [] if c]

    async def _search(
        self,
        query: str,
        limit: int,
        collections: list[str],
    ) -> str:
        """Perform semantic search across vector collections using docproc.

        Args:
            query (str): Natural language search query.
            limit (int): Maximum number of results (clamped to 1-20).
            collections (list[str]): Collection names to search. Falls back
                to ["documents"] if empty.

        Returns:
            str: Formatted results string or "No relevant documents found."

        Raises:
            No exceptions raised - errors are caught and returned as strings.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        try:
            from docproc import get_vector_store, get_embeddings_model
        except ImportError:
            logger.error("docproc library not installed")
            return "Search unavailable: docproc library not installed"

        limit = max(1, min(limit, 20))
        all_results = []

        # Get embeddings model for query
        embeddings = get_embeddings_model()
        query_embedding = embeddings.embed_query(query)

        # Search in specified collections or default
        search_collections = collections if collections else ["documents"]

        for collection_name in search_collections:
            try:
                store = get_vector_store(collection_name)
                await store.initialize()
                
                results = await store.search_similar(query_embedding, limit=limit)
                
                # Store RAW results (not formatted) following LangGraph best practices
                for result in results:
                    raw_result = {
                        "collection": collection_name,
                        "filename": result.get("metadata", {}).get("filename", "unknown"),
                        "text": result.get("text", ""),  # Full text, not truncated
                        "text_preview": result.get("text", "")[:500],  # Preview for display
                        "score": result.get("similarity_score", 0),
                        "metadata": result.get("metadata", {}),
                        "document_id": result.get("id", "unknown"),
                    }
                    all_results.append(raw_result)
                    
            except Exception as e:
                logger.warning(f"Search failed for collection {collection_name}: {e}")
                continue

        if not all_results:
            return "No relevant documents found."

        # Sort by score and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:limit]

        return self._format_results(all_results)

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        """Format search results into human-readable string.

        Args:
            results (list[dict]): List of result dicts with keys:
                - filename (str): Source document name
                - score (float): Similarity score 0.0-1.0
                - text (str): Matched content snippet

        Returns:
            str: Numbered list with filename, relevance %, and content preview.

        Last Grunted: 02/03/2026 03:45:00 PM PST
        """
        lines = ["Relevant documents found:\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result['filename']}")
            lines.append(f"   Relevance: {result['score']:.0%}")
            # Use text_preview (formatted on-demand) instead of raw text
            preview = result.get('text_preview', result.get('text', '')[:500])
            lines.append(f"   Content: {preview[:200]}...")
            lines.append("")
        
        return "\n".join(lines)


# Default instance for agent wiring
search_document_content = SearchDocumentContentTool()

__all__ = [
    "SearchDocumentContentTool",
    "search_document_content",
]
