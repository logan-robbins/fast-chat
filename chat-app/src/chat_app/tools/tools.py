"""Tool implementations for Chat-App agents.

This module provides LangChain tools used by the specialized agents:
- perplexity_search: Async web search via Perplexity API with citations

The perplexity_search tool uses Perplexity's sonar model which provides
AI-synthesized answers with source citations.

Status Streaming (ChatGPT/Claude 2025/2026 patterns):
    Tools emit user-friendly status events via get_stream_writer():
    - "Searching the web for: [query]..." at start
    - "Web search complete" on success

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.tools import tool

# Import RAG tool for re-export
from .rag_tool import search_document_content

logger = logging.getLogger(__name__)


@tool
async def perplexity_search(query: str) -> str:
    """Search the web using Perplexity API and return content with citations.

    Performs an async web search using Perplexity's sonar model, which provides
    AI-synthesized answers with source citations. Results include both the answer
    content and a numbered SOURCES section for attribution.

    Args:
        query: The search query string. Must be non-empty and contain
            substantive search terms.

    Returns:
        str: Formatted response with two sections:
            1. Content: AI-synthesized answer from Perplexity
            2. SOURCES: Numbered list of source URLs with optional dates

    Raises:
        No exceptions raised -- errors are returned as formatted error messages.

    Environment Variables:
        PERPLEXITY_API_KEY: Required API key (raises ValueError if missing)
        PERPLEXITY_MODEL: Model name (default: "sonar")
        PERPLEXITY_TIMEOUT: Request timeout in seconds (default: 30)

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    from chat_app.utils.perplexity_client import call_perplexity_api

    # Validate query is not empty or just whitespace
    if not query or not query.strip():
        logger.warning(
            "Empty query provided to perplexity_search",
            extra={
                "query_length": len(query) if query else 0,
                "query_repr": repr(query) if query else None,
            },
        )
        return (
            "Error: Search query cannot be empty. Please provide a valid search query.\n\n"
            "---\nSOURCES:\n[1] Search error (invalid query)"
        )

    logger.info(
        "Calling Perplexity API for web search",
        extra={
            "query_length": len(query),
            "query_preview": query[:100],
        },
    )

    # Emit status event (ChatGPT/Claude-style "Searching the web...")
    try:
        from chat_app.status_streaming import emit_tool_start
        emit_tool_start(
            tool_name="perplexity_search",
            agent_name="websearch",
            details={"query": query[:100]},
        )
    except Exception:
        pass  # Status emission failures should not break tool execution

    try:
        # Call Perplexity with search request (async)
        messages: List[Dict[str, str]] = [{
            "role": "user",
            "content": f"Search the web comprehensively for: {query}. Return full context and all relevant information.",
        }]
        response = await call_perplexity_api(messages)

        logger.debug(
            "Perplexity API call completed",
            extra={
                "response_content_length": len(response.get("content", "")),
                "search_results_count": len(response.get("search_results", [])),
            },
        )

        # Build output with content and SOURCES section
        output_parts: List[str] = [response["content"]]
        output_parts.append("\n\n---\nSOURCES:")

        search_results: List[Dict[str, Any]] = response.get("search_results", [])
        if search_results:
            for i, sr in enumerate(search_results, start=1):
                url = sr.get("url", "")
                date = sr.get("date")
                date_str = f" ({date})" if date else ""
                output_parts.append(f"[{i}] {url}{date_str}")
        else:
            output_parts.append("[1] Web search via Perplexity (specific URL unavailable)")

        output = "\n".join(output_parts)
        logger.info(
            "Perplexity search completed successfully",
            extra={
                "output_length": len(output),
                "sources_count": len(search_results),
            },
        )

        # Emit completion status
        try:
            from chat_app.status_streaming import emit_tool_complete
            emit_tool_complete(
                tool_name="perplexity_search",
                agent_name="websearch",
                details={"sources_count": len(search_results)},
            )
        except Exception:
            pass

        return output

    except Exception as e:
        logger.error(
            "Exception in perplexity_search",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_preview": query[:100] if query else None,
            },
            exc_info=True,
        )
        return (
            f"Search failed: {type(e).__name__}: {e}\n\n"
            f"---\nSOURCES:\n[1] Search error (no sources available)"
        )
