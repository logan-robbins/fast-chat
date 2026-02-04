"""Retry policies for external service calls following LangGraph best practices.

This module defines retry policies for different types of operations:
- Transient errors: Network issues, rate limits - automatic retry with exponential backoff
- LLM-recoverable errors: Tool failures - store error in state and loop back
- User-fixable errors: Missing info - interrupt() for human input
- Unexpected errors: Bubble up for debugging

Last Grunted: 02/04/2026 03:30:00 PM PST
"""
from langgraph.types import RetryPolicy


# Retry policy for external API calls (Perplexity, OpenAI)
# Handles transient errors like network issues, rate limits
def get_external_api_retry_policy() -> RetryPolicy:
    """Get retry policy for external API calls.
    
    Uses exponential backoff for transient failures:
    - Initial interval: 1 second
    - Max attempts: 3
    - Multiplier: 2.0 (doubles each retry)
    
    Returns:
        RetryPolicy: Configured for transient error handling
    """
    return RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        max_interval=10.0,
        multiplier=2.0,
        retry_on=(
            ConnectionError,
            TimeoutError,
            # Common HTTP errors that might be transient
            Exception  # Will be filtered by error type checking in node
        )
    )


# Retry policy for vector store operations
def get_vector_store_retry_policy() -> RetryPolicy:
    """Get retry policy for vector store operations.
    
    ChromaDB or other vector stores may have transient failures.
    
    Returns:
        RetryPolicy: Configured for database operations
    """
    return RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        max_interval=5.0,
        multiplier=1.5
    )


# Retry policy for code execution (more conservative)
def get_code_execution_retry_policy() -> RetryPolicy:
    """Get retry policy for code execution.
    
    Code execution failures are often not transient, so we use
    fewer retries and shorter intervals.
    
    Returns:
        RetryPolicy: Conservative policy for code operations
    """
    return RetryPolicy(
        max_attempts=2,
        initial_interval=0.5,
        max_interval=2.0,
        multiplier=1.5
    )


# Retry policy for agent handoff calls
def get_agent_handoff_retry_policy() -> RetryPolicy:
    """Get retry policy for agent handoff operations.
    
    Agent handoffs may fail due to transient issues in the
    subagent's external dependencies.
    
    Returns:
        RetryPolicy: Configured for agent delegation
    """
    return RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        max_interval=8.0,
        multiplier=2.0
    )


__all__ = [
    "get_external_api_retry_policy",
    "get_vector_store_retry_policy", 
    "get_code_execution_retry_policy",
    "get_agent_handoff_retry_policy"
]