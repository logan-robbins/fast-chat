"""Chat-App: Multi-agent chat backend following LangGraph best practices.

This package provides a LangGraph-based multi-agent system with:
- Explicit TypedDict state schema (ChatAppState)
- Structured routing decisions
- Retry policies for external services
- Human-in-the-loop support with interrupt()
- Error handling following LangGraph patterns
- State validation

Last Grunted: 02/04/2026 03:30:00 PM PST
"""

from chat_app.state import (
    ChatAppState,
    UserIntent,
    AgentResult,
    ErrorState,
    PendingHumanAction,
    RoutingDecision,
    HumanResponse,
    create_default_state
)

from chat_app.retry_policies import (
    get_external_api_retry_policy,
    get_vector_store_retry_policy,
    get_code_execution_retry_policy,
    get_agent_handoff_retry_policy
)

from chat_app.error_handlers import (
    classify_error,
    handle_transient_error,
    handle_llm_recoverable_error,
    handle_user_fixable_error,
    handle_unexpected_error,
    clear_recovered_errors
)

from chat_app.state_validation import (
    StateValidationError,
    validate_state_schema,
    validate_state_transition,
    sanitize_state_for_logging
)

__version__ = "2.0.0"

__all__ = [
    # State
    "ChatAppState",
    "UserIntent",
    "AgentResult",
    "ErrorState",
    "PendingHumanAction",
    "RoutingDecision",
    "HumanResponse",
    "create_default_state",
    # Retry policies
    "get_external_api_retry_policy",
    "get_vector_store_retry_policy",
    "get_code_execution_retry_policy",
    "get_agent_handoff_retry_policy",
    # Error handlers
    "classify_error",
    "handle_transient_error",
    "handle_llm_recoverable_error",
    "handle_user_fixable_error",
    "handle_unexpected_error",
    "clear_recovered_errors",
    # Validation
    "StateValidationError",
    "validate_state_schema",
    "validate_state_transition",
    "sanitize_state_for_logging",
]
