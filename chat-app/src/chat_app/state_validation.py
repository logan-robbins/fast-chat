"""State validation for ChatAppState following LangGraph best practices.

This module provides validation functions to ensure state consistency
across node transitions and catches errors early.

Graph Topology (create_supervisor):
    The supervisor graph created by langgraph_supervisor.create_supervisor
    produces the following node topology:

    __start__ → supervisor → {websearch, knowledge_base, code_interpreter, __end__}
    websearch → supervisor
    knowledge_base → supervisor
    code_interpreter → supervisor
    supervisor → __end__

    The supervisor node contains the LLM call loop and tool execution.
    Agent nodes are subgraphs invoked via handoff tools.

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from typing import List, Dict, Any, Optional, Set
from langchain_core.messages import BaseMessage

from chat_app.state import ChatAppState, UserIntent, AgentResult, ErrorState


class StateValidationError(Exception):
    """Raised when state validation fails."""
    pass


# Valid state transitions matching actual create_supervisor graph topology.
# The supervisor uses tool-calling to route to agent subgraph nodes.
# Agents always return to the supervisor, which then decides to delegate
# again or respond to the user (__end__).
VALID_AGENT_TRANSITIONS: Dict[str, Set[str]] = {
    "__start__": {"supervisor"},
    "supervisor": {"websearch", "knowledge_base", "code_interpreter", "__end__"},
    "websearch": {"supervisor"},
    "knowledge_base": {"supervisor"},
    "code_interpreter": {"supervisor"},
}


def validate_state_schema(state: Any) -> ChatAppState:
    """Validate that state conforms to ChatAppState schema.
    
    Args:
        state: The state to validate
        
    Returns:
        Validated ChatAppState
        
    Raises:
        StateValidationError: If state is invalid
    """
    if not isinstance(state, dict):
        raise StateValidationError(f"State must be a dict, got {type(state)}")
    
    # Check required fields
    required_fields = ["messages"]
    for field in required_fields:
        if field not in state:
            raise StateValidationError(f"Missing required field: {field}")
    
    # Validate messages
    if not isinstance(state["messages"], list):
        raise StateValidationError(f"messages must be a list, got {type(state['messages'])}")
    
    # Validate optional fields if present
    if "user_intent" in state and state["user_intent"] is not None:
        validate_user_intent(state["user_intent"])
    
    if "agent_results" in state:
        if not isinstance(state["agent_results"], list):
            raise StateValidationError("agent_results must be a list")
        for result in state["agent_results"]:
            validate_agent_result(result)
    
    if "errors" in state:
        if not isinstance(state["errors"], list):
            raise StateValidationError("errors must be a list")
        for error in state["errors"]:
            validate_error_state(error)
    
    return state


def validate_user_intent(intent: Any) -> None:
    """Validate UserIntent structure.
    
    Args:
        intent: The intent to validate
        
    Raises:
        StateValidationError: If intent is invalid
    """
    if not isinstance(intent, dict):
        raise StateValidationError(f"UserIntent must be a dict, got {type(intent)}")
    
    valid_intents = {"websearch", "knowledge_base", "code_interpreter", "general", "ambiguous"}
    if intent.get("primary_intent") not in valid_intents:
        raise StateValidationError(
            f"Invalid primary_intent: {intent.get('primary_intent')}. "
            f"Must be one of: {valid_intents}"
        )
    
    valid_confidences = {"high", "medium", "low"}
    if intent.get("confidence") not in valid_confidences:
        raise StateValidationError(
            f"Invalid confidence: {intent.get('confidence')}. "
            f"Must be one of: {valid_confidences}"
        )


def validate_agent_result(result: Any) -> None:
    """Validate AgentResult structure.
    
    Args:
        result: The result to validate
        
    Raises:
        StateValidationError: If result is invalid
    """
    if not isinstance(result, dict):
        raise StateValidationError(f"AgentResult must be a dict, got {type(result)}")
    
    required_fields = ["agent_name", "task_description", "raw_response"]
    for field in required_fields:
        if field not in result:
            raise StateValidationError(f"AgentResult missing required field: {field}")


def validate_error_state(error: Any) -> None:
    """Validate ErrorState structure.
    
    Args:
        error: The error to validate
        
    Raises:
        StateValidationError: If error is invalid
    """
    if not isinstance(error, dict):
        raise StateValidationError(f"ErrorState must be a dict, got {type(error)}")
    
    valid_types = {"transient", "llm_recoverable", "user_fixable", "unexpected"}
    if error.get("error_type") not in valid_types:
        raise StateValidationError(
            f"Invalid error_type: {error.get('error_type')}. "
            f"Must be one of: {valid_types}"
        )


def validate_state_transition(
    current_node: str,
    next_node: str,
    valid_transitions: Optional[Dict[str, Set[str]]] = None
) -> None:
    """Validate that a state transition is allowed.
    
    Checks that the transition from current_node to next_node is valid
    according to the graph topology created by create_supervisor.
    
    Args:
        current_node: Current node name (e.g., "supervisor", "websearch").
        next_node: Target node name.
        valid_transitions: Dict of valid transitions. Defaults to
            VALID_AGENT_TRANSITIONS which matches the create_supervisor topology.
        
    Raises:
        StateValidationError: If the transition is not in the allowed set.
        
    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    if valid_transitions is None:
        valid_transitions = VALID_AGENT_TRANSITIONS
    
    valid_next = valid_transitions.get(current_node, set())
    if next_node not in valid_next:
        raise StateValidationError(
            f"Invalid transition from '{current_node}' to '{next_node}'. "
            f"Valid transitions: {valid_next}"
        )


def sanitize_state_for_logging(state: ChatAppState) -> Dict[str, Any]:
    """Create a sanitized version of state for logging.
    
    Removes sensitive information like full message content.
    
    Args:
        state: The state to sanitize
        
    Returns:
        Sanitized state dict
    """
    return {
        "message_count": len(state.get("messages", [])),
        "has_user_intent": state.get("user_intent") is not None,
        "agent_results_count": len(state.get("agent_results", [])),
        "errors_count": len(state.get("errors", [])),
        "has_pending_human_action": state.get("pending_human_action") is not None,
    }


__all__ = [
    "StateValidationError",
    "validate_state_schema",
    "validate_user_intent",
    "validate_agent_result",
    "validate_error_state",
    "validate_state_transition",
    "sanitize_state_for_logging",
    "VALID_AGENT_TRANSITIONS"
]