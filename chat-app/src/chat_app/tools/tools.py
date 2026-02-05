"""Tool implementations for Chat-App agents.

This module provides LangChain tools used by the specialized agents:
- perplexity_search: Web search via Perplexity API with citations

The perplexity_search tool uses Perplexity's sonar model (2025) which provides
AI-synthesized answers with source citations at ~1200 tokens/second.

Status Streaming (ChatGPT/Claude 2025/2026 patterns):
    Tools emit user-friendly status events via get_stream_writer():
    - "Searching the web for: [query]..." at start
    - "Web search complete" on success

Last Grunted: 02/04/2026 07:30:00 PM PST
"""
from langchain_core.tools import tool
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG tool
from .rag_tool import search_document_content

logger = logging.getLogger(__name__)


@tool
def perplexity_search(query: str) -> str:
    """Search the web using Perplexity API and return content with citations.

    Performs a web search using Perplexity's sonar model (2025 version built
    on Llama 3.3 70B), which provides AI-synthesized answers with source
    citations. Results include both the answer content and a numbered
    SOURCES section for attribution.

    Args:
        query (str): The search query string. Must be non-empty and contain
            substantive search terms. Whitespace-only queries are rejected.

    Returns:
        str: Formatted response with two sections:
            1. Content: AI-synthesized answer from Perplexity
            2. SOURCES: Numbered list of source URLs with optional dates
               Format: "[n] https://url.com (YYYY-MM-DD)"

    Example:
        >>> result = perplexity_search("Python release date")
        >>> "---\\nSOURCES:" in result
        True

    Raises:
        No exceptions raised - errors are returned as formatted error messages
        with appropriate SOURCES section indicating the error.

    Environment Variables:
        PERPLEXITY_API_KEY: Required API key (raises ValueError if missing)
        PERPLEXITY_MODEL: Model name (default: "sonar")
        PERPLEXITY_TIMEOUT: Request timeout in seconds (default: 30)

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    from chat_app.utils.perplexity_client import call_perplexity_api

    # Validate query is not empty or just whitespace
    if not query or not query.strip():
        logger.warning(
            "Empty query provided to perplexity_search",
            extra={
                "query_length": len(query) if query else 0,
                "query_repr": repr(query) if query else None
            }
        )
        return (
            "Error: Search query cannot be empty. Please provide a valid search query.\n\n"
            "---\nSOURCES:\n[1] Search error (invalid query)"
        )

    logger.info(
        "Calling Perplexity API for web search",
        extra={
            "query_length": len(query),
            "query_preview": query[:100]
        }
    )
    
    # Emit status event (ChatGPT/Claude-style "Searching the web...")
    try:
        from chat_app.status_streaming import emit_tool_start
        emit_tool_start(
            tool_name="perplexity_search",
            agent_name="websearch",
            details={"query": query[:100]}
        )
    except Exception:
        pass  # Status emission failures should not break tool execution

    try:
        # Call Perplexity with search request
        messages = [{
            "role": "user",
            "content": f"Search the web comprehensively for: {query}. Return full context and all relevant information."
        }]
        response = call_perplexity_api(messages)
        
        logger.debug(
            "Perplexity API call completed",
            extra={
                "response_content_length": len(response.get('content', '')),
                "search_results_count": len(response.get('search_results', [])),
                "response_preview": response.get('content', '')[:200] if response.get('content') else None
            }
        )

        # Build output with content and SOURCES section
        output_parts = [response['content']]
        output_parts.append("\n\n---\nSOURCES:")

        search_results = response.get('search_results', [])
        if search_results:
            for i, sr in enumerate(search_results, start=1):
                url = sr.get('url', '')
                date = sr.get('date')
                date_str = f" ({date})" if date else ""
                output_parts.append(f"[{i}] {url}{date_str}")
        else:
            # Fallback when no sources available from API
            output_parts.append("[1] Web search via Perplexity (specific URL unavailable)")

        output = "\n".join(output_parts)
        logger.info(
            "Perplexity search completed successfully",
            extra={
                "output_length": len(output),
                "sources_count": len(search_results) if search_results else 0
            }
        )
        
        # Emit completion status
        try:
            from chat_app.status_streaming import emit_tool_complete
            emit_tool_complete(
                tool_name="perplexity_search",
                agent_name="websearch",
                details={"sources_count": len(search_results) if search_results else 0}
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
                "query_preview": query[:100] if query else None
            },
            exc_info=True
        )
        return f"Search failed: {type(e).__name__}: {str(e)}\n\n---\nSOURCES:\n[1] Search error (no sources available)"
