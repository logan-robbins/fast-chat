"""Retry policies for external service calls following LangGraph best practices.

This module defines retry policies for different types of operations:
- External API calls: Network issues, rate limits, timeouts
- Vector store operations: Database connectivity issues
- Code execution: Conservative retries for non-transient failures
- Agent handoffs: Transient issues in subagent dependencies

LangGraph RetryPolicy applies automatic retry with exponential backoff
at the node level. The retry_on parameter should target SPECIFIC
recoverable exceptions -- never broad Exception.

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from __future__ import annotations

import httpx
from langgraph.types import RetryPolicy


def get_external_api_retry_policy() -> RetryPolicy:
    """Get retry policy for external API calls (Perplexity, OpenAI).

    Targets specific transient exceptions: network errors, timeouts,
    and rate limits. Does NOT retry on ValueError, KeyError, etc.
    which indicate code bugs rather than transient failures.

    Returns:
        RetryPolicy: Configured for transient HTTP error handling with
            exponential backoff (1s, 2s, 4s) up to 3 attempts.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    return RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        max_interval=10.0,
        multiplier=2.0,
        retry_on=(
            ConnectionError,
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ),
    )


def get_vector_store_retry_policy() -> RetryPolicy:
    """Get retry policy for vector store operations.

    Vector store backends (ChromaDB, pgvector) may have transient
    connectivity issues. Retries on connection and timeout errors.

    Returns:
        RetryPolicy: Configured for database operation retries with
            shorter intervals (0.5s, 0.75s, 1.125s).

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    return RetryPolicy(
        max_attempts=3,
        initial_interval=0.5,
        max_interval=5.0,
        multiplier=1.5,
        retry_on=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )


def get_code_execution_retry_policy() -> RetryPolicy:
    """Get retry policy for code execution.

    Code execution failures are typically NOT transient (syntax errors,
    runtime errors). Only retries on system-level issues.

    Returns:
        RetryPolicy: Conservative policy -- 2 attempts max, short intervals.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    return RetryPolicy(
        max_attempts=2,
        initial_interval=0.5,
        max_interval=2.0,
        multiplier=1.5,
        retry_on=(
            ConnectionError,
            TimeoutError,
        ),
    )


def get_agent_handoff_retry_policy() -> RetryPolicy:
    """Get retry policy for agent handoff operations.

    Agent handoffs may fail due to transient issues in the subagent's
    external dependencies (API timeouts, rate limits).

    Returns:
        RetryPolicy: 3 attempts with exponential backoff (1s, 2s, 4s).

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    return RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        max_interval=8.0,
        multiplier=2.0,
        retry_on=(
            ConnectionError,
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ),
    )


__all__ = [
    "get_external_api_retry_policy",
    "get_vector_store_retry_policy",
    "get_code_execution_retry_policy",
    "get_agent_handoff_retry_policy",
]
