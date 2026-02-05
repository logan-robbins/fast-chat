"""Status streaming utilities for ChatGPT/Claude-style user status updates.

This module provides utilities for emitting user-friendly status updates during
LangGraph execution, following the patterns established by ChatGPT and Claude
in 2025/2026:

- "Thinking..." indicators during reasoning
- "Searching the web..." during tool execution
- "Looking through your documents..." during RAG
- "Running Python code..." during code interpreter

Implementation uses LangGraph's get_stream_writer() for custom event emission
which flows through stream_mode="custom" to the API layer.

OpenAI Responses API Pattern Reference:
- response.web_search_call: in_progress → searching → completed
- response.file_search_call: in_progress → searching → completed  
- response.code_interpreter_call: in_progress → interpreting → completed

Dependencies:
- langgraph>=0.2.0: For get_stream_writer
- chat_app.state: For StatusEvent TypedDict

Last Grunted: 02/04/2026 07:30:00 PM PST
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chat_app.state import (
    StatusEvent,
    STATUS_THINKING,
    STATUS_TOOL_START,
    STATUS_TOOL_PROGRESS,
    STATUS_TOOL_COMPLETE,
    STATUS_AGENT_HANDOFF,
    STATUS_ERROR,
    STATUS_MESSAGES,
)

logger = logging.getLogger(__name__)


def _get_writer():
    """Get the stream writer, returning None if not available.
    
    Wraps the import and call to handle cases where get_stream_writer()
    fails (e.g., called outside streaming context or Python < 3.11 async).
    
    Returns:
        StreamWriter | None: The stream writer if available, else None.
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:
        return None


def _emit_status(event: StatusEvent) -> None:
    """Emit a status event via the stream writer.
    
    Core emission function that handles the actual streaming.
    Silently skips if no writer is available.
    
    Args:
        event: The StatusEvent to emit.
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    writer = _get_writer()
    if writer:
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Wrap in status envelope for API layer
        writer({"status": event})
        
        logger.debug(
            "Emitted status event",
            extra={
                "status_type": event.get("type"),
                "message": event.get("message", "")[:50]
            }
        )


def emit_thinking(message: Optional[str] = None) -> None:
    """Emit a "thinking" status event.
    
    Use when the supervisor is reasoning about routing or synthesizing.
    
    Args:
        message: Custom message, defaults to "Thinking..."
        
    Example:
        >>> emit_thinking()  # "Thinking..."
        >>> emit_thinking("Analyzing your request...")
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    _emit_status({
        "type": STATUS_THINKING,
        "message": message or STATUS_MESSAGES["supervisor_thinking"],
    })


def emit_agent_handoff(
    agent_name: str,
    task_description: Optional[str] = None
) -> None:
    """Emit an agent handoff status event.
    
    Use when the supervisor hands off to a specialized agent.
    
    Args:
        agent_name: Name of the agent (websearch, knowledge_base, code_interpreter)
        task_description: Optional task description for context
        
    Example:
        >>> emit_agent_handoff("websearch")
        >>> # Emits: "Searching the web..."
        >>> emit_agent_handoff("knowledge_base")
        >>> # Emits: "Looking through your documents..."
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    # Get user-friendly message for this agent
    message_key = f"{agent_name}_handoff"
    message = STATUS_MESSAGES.get(message_key, f"Consulting {agent_name}...")
    
    event: StatusEvent = {
        "type": STATUS_AGENT_HANDOFF,
        "message": message,
        "agent": agent_name,
    }
    
    if task_description:
        event["details"] = {"task_preview": task_description[:100]}
    
    _emit_status(event)


def emit_tool_start(
    tool_name: str,
    message: Optional[str] = None,
    agent_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Emit a tool start status event.
    
    Use when a tool begins execution.
    
    Args:
        tool_name: Name of the tool (perplexity_search, search_document_content, etc.)
        message: Custom message, or auto-generated from tool_name
        agent_name: Parent agent name for context
        details: Optional structured data (query, parameters, etc.)
        
    Example:
        >>> emit_tool_start("perplexity_search", details={"query": "Python 3.12"})
        >>> # Emits: "Searching the web for: Python 3.12"
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    # Auto-generate message if not provided
    if not message:
        if tool_name == "perplexity_search":
            query = details.get("query", "") if details else ""
            if query:
                message = STATUS_MESSAGES["websearch_start"].format(query=query[:50])
            else:
                message = STATUS_MESSAGES["websearch_handoff"]
        elif tool_name == "search_document_content":
            query = details.get("query", "") if details else ""
            if query:
                message = STATUS_MESSAGES["rag_start"].format(query=query[:50])
            else:
                message = STATUS_MESSAGES["knowledge_base_handoff"]
        elif tool_name == "python_repl":
            message = STATUS_MESSAGES["code_start"]
        else:
            message = f"Running {tool_name}..."
    
    event: StatusEvent = {
        "type": STATUS_TOOL_START,
        "message": message,
        "tool": tool_name,
    }
    
    if agent_name:
        event["agent"] = agent_name
    if details:
        event["details"] = details
    
    _emit_status(event)


def emit_tool_progress(
    tool_name: str,
    message: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    agent_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Emit a tool progress status event.
    
    Use for tools with multiple steps or long-running operations.
    
    Args:
        tool_name: Name of the tool
        message: Progress message
        current: Current step number (for progress indicators)
        total: Total steps (for progress indicators)
        agent_name: Parent agent name
        details: Optional structured data
        
    Example:
        >>> emit_tool_progress(
        ...     "search_document_content",
        ...     "Searching collection 2/3: financial_docs",
        ...     current=2, total=3
        ... )
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    event: StatusEvent = {
        "type": STATUS_TOOL_PROGRESS,
        "message": message,
        "tool": tool_name,
    }
    
    if agent_name:
        event["agent"] = agent_name
    
    # Build details with progress info
    progress_details: Dict[str, Any] = details.copy() if details else {}
    if current is not None:
        progress_details["current"] = current
    if total is not None:
        progress_details["total"] = total
    
    if progress_details:
        event["details"] = progress_details
    
    _emit_status(event)


def emit_tool_complete(
    tool_name: str,
    message: Optional[str] = None,
    agent_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Emit a tool completion status event.
    
    Use when a tool finishes successfully.
    
    Args:
        tool_name: Name of the tool
        message: Custom completion message
        agent_name: Parent agent name
        details: Optional structured data (results count, etc.)
        
    Example:
        >>> emit_tool_complete(
        ...     "search_document_content",
        ...     details={"results_count": 5}
        ... )
        >>> # Emits: "Found 5 relevant documents"
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    # Auto-generate message if not provided
    if not message:
        if tool_name == "perplexity_search":
            message = STATUS_MESSAGES["websearch_complete"]
        elif tool_name == "search_document_content":
            count = details.get("results_count", 0) if details else 0
            message = STATUS_MESSAGES["rag_complete"].format(count=count)
        elif tool_name == "python_repl":
            message = STATUS_MESSAGES["code_complete"]
        else:
            message = f"{tool_name} complete"
    
    event: StatusEvent = {
        "type": STATUS_TOOL_COMPLETE,
        "message": message,
        "tool": tool_name,
    }
    
    if agent_name:
        event["agent"] = agent_name
    if details:
        event["details"] = details
    
    _emit_status(event)


def emit_error(
    error_message: str,
    tool_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Emit an error status event.
    
    Use when an error occurs during tool or agent execution.
    Note: This is for user-facing status updates, not logging.
    
    Args:
        error_message: User-friendly error description
        tool_name: Tool that encountered the error
        agent_name: Agent that encountered the error
        details: Optional error details
        
    Example:
        >>> emit_error("Search service temporarily unavailable", tool_name="perplexity_search")
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    event: StatusEvent = {
        "type": STATUS_ERROR,
        "message": STATUS_MESSAGES["error"].format(error=error_message),
    }
    
    if tool_name:
        event["tool"] = tool_name
    if agent_name:
        event["agent"] = agent_name
    if details:
        event["details"] = details
    
    _emit_status(event)


# RAG-specific helpers for backward compatibility with existing rag_tool.py patterns

def emit_rag_progress(
    message: str,
    current: int,
    total: int,
    collection: Optional[str] = None,
    results_count: Optional[int] = None
) -> None:
    """Emit RAG-specific progress (for backward compatibility).
    
    Wraps emit_tool_progress with RAG-specific formatting.
    
    Args:
        message: Progress message
        current: Current collection index
        total: Total collections to search
        collection: Current collection name
        results_count: Number of results found so far
        
    Last Grunted: 02/04/2026 07:30:00 PM PST
    """
    details: Dict[str, Any] = {"collection": collection} if collection else {}
    if results_count is not None:
        details["results_count"] = results_count
    
    emit_tool_progress(
        tool_name="search_document_content",
        message=message,
        current=current,
        total=total,
        agent_name="knowledge_base",
        details=details
    )


__all__ = [
    # Core emission functions
    "emit_thinking",
    "emit_agent_handoff",
    "emit_tool_start",
    "emit_tool_progress",
    "emit_tool_complete",
    "emit_error",
    # RAG-specific helper
    "emit_rag_progress",
]
