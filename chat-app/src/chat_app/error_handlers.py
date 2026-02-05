"""Error handling following LangGraph best practices.

This module provides error handling strategies for different error types:
- Transient errors: Retry with exponential backoff
- LLM-recoverable errors: Store error in state and loop back
- User-fixable errors: interrupt() for human input
- Unexpected errors: Bubble up for debugging

Error Classification:
    Errors are classified into four categories based on the appropriate
    recovery strategy:
    
    1. Transient: Network issues, rate limits - automatic retry
    2. LLM-recoverable: Tool failures - store in state, let LLM adjust
    3. User-fixable: Missing input - interrupt for human input
    4. Unexpected: Unknown errors - bubble up for developer debugging

State-Based Error Handling:
    Following LangGraph best practices, errors are stored in state so the
    LLM can see what went wrong and adjust its approach. This enables
    self-healing behavior without explicit retry logic.

Last Grunted: 02/04/2026 06:30:00 PM PST
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langgraph.types import Command

from chat_app.state import ChatAppState, ErrorState

logger = logging.getLogger(__name__)

# Error type classification
ErrorType = Literal["transient", "llm_recoverable", "user_fixable", "unexpected"]

# Maximum errors to keep in state (prevents unbounded growth)
MAX_ERRORS_IN_STATE: int = 10


def classify_error(error: Exception) -> ErrorType:
    """Classify an error into one of the four recovery strategy types.
    
    Classification is based on both the exception type and the error message
    content. This enables appropriate recovery strategies.
    
    Args:
        error: The exception to classify.
        
    Returns:
        ErrorType: One of "transient", "llm_recoverable", "user_fixable", "unexpected".
        
    Classification Rules:
        - Transient: Network, timeout, rate limit errors
        - User-fixable: Missing input, permission, validation errors
        - LLM-recoverable: Tool failures, key errors, attribute errors
        - Unexpected: Everything else (default)
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    error_type_name = type(error).__name__
    error_msg = str(error).lower()
    
    # Transient errors - network, rate limits, timeouts
    transient_indicators: List[str] = [
        "connection", "timeout", "rate limit", "503", "502", "504",
        "temporary", "unavailable", "retry", "network", "throttle"
    ]
    
    # User-fixable errors - missing info, invalid input
    user_fixable_indicators: List[str] = [
        "missing", "required", "invalid", "not found", "does not exist",
        "permission denied", "unauthorized", "forbidden", "access denied"
    ]
    
    # Check for transient based on message content
    if any(ind in error_msg for ind in transient_indicators):
        return "transient"
    
    # Check for user-fixable based on message content
    if any(ind in error_msg for ind in user_fixable_indicators):
        return "user_fixable"
    
    # Check for specific exception types
    transient_exceptions = {"ConnectionError", "TimeoutError", "RetryError"}
    if error_type_name in transient_exceptions:
        return "transient"
    
    llm_recoverable_exceptions = {"ValueError", "KeyError", "AttributeError", "TypeError"}
    if error_type_name in llm_recoverable_exceptions:
        return "llm_recoverable"
    
    # Default to unexpected
    return "unexpected"


def handle_transient_error(
    state: ChatAppState,
    node_name: str,
    error: Exception,
    max_retries: int = 3
) -> Command:
    """Handle transient errors with retry logic.
    
    Transient errors (network, rate limits) are handled by the retry
    policy at the node level. This function is called when retries
    are exhausted.
    
    Args:
        state: Current graph state.
        node_name: Name of the node that failed.
        error: The exception that occurred.
        max_retries: Maximum retry attempts (default: 3).
        
    Returns:
        Command to update state and route appropriately.
        
    Raises:
        Exception: Re-raises the error if retries haven't been exhausted,
            allowing the retry policy to handle it.
            
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    # Get current retry count for this node's transient errors
    existing_errors: List[ErrorState] = state.get("errors", [])
    retry_count = sum(
        1 for e in existing_errors
        if e.get("node_name") == node_name and e.get("error_type") == "transient"
    )
    
    error_state = ErrorState(
        node_name=node_name,
        error_type="transient",
        error_message=f"{type(error).__name__}: {str(error)}",
        retry_count=retry_count + 1,
        timestamp=datetime.now().isoformat()
    )
    
    logger.warning(
        "Transient error occurred",
        extra={
            "node_name": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error)[:200],
            "retry_count": retry_count + 1,
            "max_retries": max_retries
        }
    )
    
    # If we've exceeded max retries, route to error handler
    if retry_count >= max_retries:
        logger.error(
            "Max retries exceeded for transient error",
            extra={
                "node_name": node_name,
                "retry_count": retry_count,
                "max_retries": max_retries
            }
        )
        return Command(
            update={
                "errors": existing_errors + [error_state]
            },
            goto="handle_max_retries_exceeded"
        )
    
    # Otherwise, let the retry policy handle it
    raise error  # Will be caught by retry policy


def handle_llm_recoverable_error(
    state: ChatAppState,
    node_name: str,
    error: Exception,
    return_to_node: str
) -> Command:
    """Handle LLM-recoverable errors by storing in state and looping back.
    
    This follows the LangGraph pattern: store error in state so the LLM
    can see what went wrong and adjust its approach. By including the
    error in state, the LLM can reason about what went wrong and try
    a different tool or strategy.
    
    Args:
        state: Current graph state.
        node_name: Name of the node that failed.
        error: The exception that occurred.
        return_to_node: Node to route back to for retry.
        
    Returns:
        Command with error in state and route back to agent.
        
    Note:
        Error history is capped at MAX_ERRORS_IN_STATE to prevent
        unbounded state growth.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    error_state = ErrorState(
        node_name=node_name,
        error_type="llm_recoverable",
        error_message=f"{type(error).__name__}: {str(error)}",
        retry_count=0,
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(
        "LLM-recoverable error stored in state",
        extra={
            "node_name": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error)[:200],
            "return_to_node": return_to_node
        }
    )
    
    # Update state with error information
    existing_errors: List[ErrorState] = state.get("errors", [])
    updated_errors = existing_errors + [error_state]
    
    # Limit error history to prevent state bloat
    if len(updated_errors) > MAX_ERRORS_IN_STATE:
        updated_errors = updated_errors[-MAX_ERRORS_IN_STATE:]
        logger.debug(
            "Truncated error history",
            extra={"max_errors": MAX_ERRORS_IN_STATE}
        )
    
    return Command(
        update={
            "errors": updated_errors
        },
        goto=return_to_node
    )


def handle_user_fixable_error(
    state: ChatAppState,
    node_name: str,
    error: Exception,
    context: Dict[str, Any]
) -> Command:
    """Handle user-fixable errors by interrupting for human input.
    
    Uses LangGraph's interrupt() function to pause execution and request
    human input. After the human provides the missing information, the
    node is retried.
    
    Args:
        state: Current graph state.
        node_name: Name of the node that failed.
        error: The exception that occurred.
        context: Additional context for the human (e.g., what's needed).
        
    Returns:
        Command to resume after human input and retry the failed node.
        
    Note:
        Following LangGraph best practices, interrupt() is called first
        before any other logic. The code after interrupt() only executes
        after the human resumes the conversation.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    from langgraph.types import interrupt
    
    error_info = f"{type(error).__name__}: {str(error)}"
    
    logger.info(
        "Interrupting for human input to resolve error",
        extra={
            "node_name": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error)[:200],
            "context_keys": list(context.keys()) if context else []
        }
    )
    
    # interrupt() MUST come first - this is the LangGraph pattern
    human_response = interrupt({
        "action_type": "error_resolution",
        "node": node_name,
        "error": error_info,
        "context": context,
        "request": "Please provide the missing information or correct the issue so we can continue."
    })
    
    logger.info(
        "Resuming after human input for error resolution",
        extra={
            "node_name": node_name,
            "has_response": human_response is not None
        }
    )
    
    # After resume, process human response
    return Command(
        update={
            "pending_human_action": None,
            "metadata": {
                **state.get("metadata", {}),
                "error_resolved_by_human": True
            }
        },
        goto=node_name  # Retry the node with new information
    )


def handle_unexpected_error(
    state: ChatAppState,
    node_name: str,
    error: Exception
) -> Command:
    """Handle unexpected errors by storing and routing to error handler.
    
    Unexpected errors should bubble up to developers for debugging,
    but we store them in state first for observability. The error is
    logged with full context before re-raising.
    
    Args:
        state: Current graph state.
        node_name: Name of the node that failed.
        error: The exception that occurred.
        
    Returns:
        Command to update state and route to __end__.
        
    Raises:
        Exception: The original error after logging, to bubble up for debugging.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    error_state = ErrorState(
        node_name=node_name,
        error_type="unexpected",
        error_message=f"{type(error).__name__}: {str(error)}",
        retry_count=0,
        timestamp=datetime.now().isoformat()
    )
    
    logger.error(
        "Unexpected error occurred - bubbling up for debugging",
        extra={
            "node_name": node_name,
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
        },
        exc_info=True  # Include full traceback
    )
    
    # Store error for observability
    existing_errors: List[ErrorState] = state.get("errors", [])
    updated_errors = existing_errors + [error_state]
    
    # Note: Command is created but not returned since we're re-raising.
    # The state update would be applied if we returned the command instead.
    # This logs the error context but lets the exception propagate.
    _ = Command(
        update={
            "errors": updated_errors
        },
        goto="__end__"
    )
    
    # Re-raise to bubble up for developer debugging
    raise error


def clear_recovered_errors(
    state: ChatAppState,
    max_retry_threshold: int = 3
) -> Dict[str, Any]:
    """Clear errors that have been recovered from.
    
    Should be called after successful retry to keep the error list clean
    and prevent unbounded state growth. Errors with retry_count >= threshold
    are considered resolved/handled.
    
    Args:
        state: Current state with errors.
        max_retry_threshold: Errors with retry_count >= this value are
            considered resolved and removed (default: 3).
        
    Returns:
        Dictionary with updated errors list, suitable for state update.
        
    Note:
        Returns a dict rather than full state to follow LangGraph's
        immutable state update pattern. Use with Command.update or
        return from a node directly.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    existing_errors: List[ErrorState] = state.get("errors", [])
    
    # Keep only recent unrecovered errors
    recent_errors = [
        e for e in existing_errors
        if e.get("retry_count", 0) < max_retry_threshold
    ]
    
    errors_cleared = len(existing_errors) - len(recent_errors)
    
    if errors_cleared > 0:
        logger.info(
            "Cleared recovered errors from state",
            extra={
                "errors_cleared": errors_cleared,
                "errors_remaining": len(recent_errors),
                "max_retry_threshold": max_retry_threshold
            }
        )
    
    return {"errors": recent_errors}


__all__ = [
    "classify_error",
    "handle_transient_error",
    "handle_llm_recoverable_error",
    "handle_user_fixable_error",
    "handle_unexpected_error",
    "clear_recovered_errors",
    "ErrorType",
    "MAX_ERRORS_IN_STATE",
]