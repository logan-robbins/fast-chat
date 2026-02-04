"""Error handling following LangGraph best practices.

This module provides error handling strategies for different error types:
- Transient errors: Retry with exponential backoff
- LLM-recoverable errors: Store error in state and loop back
- User-fixable errors: interrupt() for human input
- Unexpected errors: Bubble up for debugging

Last Grunted: 02/04/2026 03:30:00 PM PST
"""
from datetime import datetime
from typing import Literal, Dict, Any, Optional
from langgraph.types import Command

from chat_app.state import ChatAppState, ErrorState


def classify_error(error: Exception) -> Literal["transient", "llm_recoverable", "user_fixable", "unexpected"]:
    """Classify an error into one of the four types.
    
    Args:
        error: The exception to classify
        
    Returns:
        Error type classification
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Transient errors - network, rate limits, timeouts
    transient_indicators = [
        "connection", "timeout", "rate limit", "503", "502", "504",
        "temporary", "unavailable", "retry", "network"
    ]
    
    # User-fixable errors - missing info, invalid input
    user_fixable_indicators = [
        "missing", "required", "invalid", "not found", "does not exist",
        "permission denied", "unauthorized"
    ]
    
    # Check for transient
    if any(ind in error_msg for ind in transient_indicators):
        return "transient"
    
    # Check for user-fixable
    if any(ind in error_msg for ind in user_fixable_indicators):
        return "user_fixable"
    
    # Check for specific exception types
    if error_type in ["ConnectionError", "TimeoutError", "RetryError"]:
        return "transient"
    
    if error_type in ["ValueError", "KeyError", "AttributeError"]:
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
        state: Current graph state
        node_name: Name of the node that failed
        error: The exception that occurred
        max_retries: Maximum retry attempts
        
    Returns:
        Command to update state and route appropriately
    """
    # Get current retry count
    retry_count = sum(
        1 for e in state.get("errors", [])
        if e.get("node_name") == node_name and e.get("error_type") == "transient"
    )
    
    error_state = ErrorState(
        node_name=node_name,
        error_type="transient",
        error_message=f"{type(error).__name__}: {str(error)}",
        retry_count=retry_count + 1,
        timestamp=datetime.now().isoformat()
    )
    
    # If we've exceeded max retries, route to error handler
    if retry_count >= max_retries:
        return Command(
            update={
                "errors": state.get("errors", []) + [error_state]
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
    can see what went wrong and adjust its approach.
    
    Args:
        state: Current graph state
        node_name: Name of the node that failed
        error: The exception that occurred
        return_to_node: Node to route back to for retry
        
    Returns:
        Command with error in state and route back to agent
    """
    error_state = ErrorState(
        node_name=node_name,
        error_type="llm_recoverable",
        error_message=f"{type(error).__name__}: {str(error)}",
        retry_count=0,
        timestamp=datetime.now().isoformat()
    )
    
    # Update state with error information
    updated_errors = state.get("errors", []) + [error_state]
    
    # Limit error history to prevent state bloat
    if len(updated_errors) > 10:
        updated_errors = updated_errors[-10:]
    
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
    
    Args:
        state: Current graph state
        node_name: Name of the node that failed
        error: The exception that occurred
        context: Additional context for the human
        
    Returns:
        Command to interrupt for human input
    """
    from langgraph.types import interrupt
    
    # interrupt() MUST come first - this is the LangGraph pattern
    human_response = interrupt({
        "action_type": "error_resolution",
        "node": node_name,
        "error": f"{type(error).__name__}: {str(error)}",
        "context": context,
        "request": "Please provide the missing information or correct the issue so we can continue."
    })
    
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
    but we store them in state first for observability.
    
    Args:
        state: Current graph state
        node_name: Name of the node that failed
        error: The exception that occurred
        
    Returns:
        Command to update state and route to error handler
        
    Raises:
        The original error after logging, to bubble up for debugging
    """
    error_state = ErrorState(
        node_name=node_name,
        error_type="unexpected",
        error_message=f"{type(error).__name__}: {str(error)}",
        retry_count=0,
        timestamp=datetime.now().isoformat()
    )
    
    # Store error for observability
    updated_errors = state.get("errors", []) + [error_state]
    
    # Update state
    Command(
        update={
            "errors": updated_errors
        },
        goto="__end__"
    )
    
    # Re-raise to bubble up for developer debugging
    raise error


def clear_recovered_errors(state: ChatAppState) -> ChatAppState:
    """Clear errors that have been recovered from.
    
    Should be called after successful retry to keep error list clean.
    
    Args:
        state: Current state with errors
        
    Returns:
        Updated state with recovered errors cleared
    """
    # Keep only recent unrecovered errors
    # This is a simple implementation - could be more sophisticated
    recent_errors = [
        e for e in state.get("errors", [])
        if e.get("retry_count", 0) < 3  # Keep errors that haven't been resolved
    ]
    
    return {
        **state,
        "errors": recent_errors
    }


__all__ = [
    "classify_error",
    "handle_transient_error",
    "handle_llm_recoverable_error",
    "handle_user_fixable_error",
    "handle_unexpected_error",
    "clear_recovered_errors"
]