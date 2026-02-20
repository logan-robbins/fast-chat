"""Tool implementations for Chat-App agents.

This module provides LangChain tools used by the specialized agents:
- perplexity_search: Async web search via configured provider (Perplexity or MCP)

The perplexity_search tool supports:
- WEB_SEARCH_PROVIDER=perplexity (default): Perplexity sonar model
- WEB_SEARCH_PROVIDER=mcp: MCP `web_search` capability via MCP_SERVERS_JSON

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

from chat_app.services.runtime_config import get_web_search_runtime_config

# Import RAG tool for re-export
from .rag_tool import search_document_content

logger = logging.getLogger(__name__)


@tool
async def web_search(query: str) -> str:
    """Search the web using the configured MCP provider (e.g., Bright Data) and return cited content.

    Performs an async web search via the MCP `web_search` capability.

    Args:
        query: The search query string. Must be non-empty and contain
            substantive search terms.

    Returns:
        str: The search results provided by the MCP server.

    Environment Variables:
        MCP_SERVERS_JSON: Required configuration for MCP servers
        MCP_WEB_SEARCH_TOOL: MCP tool name (default: "search_engine")
        MCP_WEB_SEARCH_ENGINE: Search engine argument (default: "google")

    Last Grunted: 02/16/2026 05:05:00 PM PST
    """
    # Validate query is not empty or just whitespace
    if not query or not query.strip():
        logger.warning(
            "Empty query provided to web_search",
            extra={
                "query_length": len(query) if query else 0,
                "query_repr": repr(query) if query else None,
            },
        )
        return "Error: Search query cannot be empty. Please provide a valid search query."

    try:
        runtime_config = get_web_search_runtime_config()
    except RuntimeError as exc:
        return f"Search configuration failed: {exc}"

    logger.info(
        "Calling MCP web search provider",
        extra={
            "query_length": len(query),
            "query_preview": query[:100],
        },
    )

    # Emit status event (ChatGPT/Claude-style "Searching the web...")
    try:
        from chat_app.status_streaming import emit_tool_start
        emit_tool_start(
            tool_name="web_search",
            agent_name="websearch",
            details={"query": query[:100]},
        )
    except Exception:
            pass  # Status emission failures should not break tool execution

    try:
        from chat_app.services.mcp_client import search_web_via_mcp

        output = await search_web_via_mcp(query)
        
        try:
            from chat_app.status_streaming import emit_tool_complete
            emit_tool_complete(
                tool_name="web_search",
                agent_name="websearch",
                details={"provider": "mcp"},
            )
        except Exception:
            pass
            
        return output

    except Exception as e:
        logger.error(
            "Exception in web_search",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "query_preview": query[:100] if query else None,
            },
            exc_info=True,
        )
        return f"Search failed: {type(e).__name__}: {e}"
