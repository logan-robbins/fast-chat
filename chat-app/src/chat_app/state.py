"""State schema for Chat-App following LangGraph best practices.

This module defines the TypedDict state schema for type-safe state management.
Following the LangGraph documentation pattern of storing raw data, not formatted text.

State Design Principles:
- Store raw data that cannot be reconstructed
- Store classification/intent results needed by multiple nodes
- Store expensive-to-fetch data (search results, external API responses)
- Store execution metadata for debugging and recovery
- Format prompts on-demand inside nodes, not in state

Last Grunted: 02/04/2026 03:30:00 PM PST
"""
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from operator import add
from langchain_core.messages import BaseMessage


class UserIntent(TypedDict):
    """Structured user intent classification.
    
    Used for routing decisions and stored as raw structured data,
    not formatted text.
    """
    primary_intent: Literal["websearch", "knowledge_base", "code_interpreter", "general", "ambiguous"]
    confidence: Literal["high", "medium", "low"]
    requires_confirmation: bool
    description: str


class AgentResult(TypedDict):
    """Raw result from a delegated agent.
    
    Stores the agent's response in a structured format for potential
    re-formatting by the supervisor.
    """
    agent_name: str
    task_description: str
    raw_response: str
    citations: List[Dict[str, str]]  # List of {source: str, url_or_doc: str}
    timestamp: str


class ErrorState(TypedDict):
    """Error information for LLM-recoverable errors.
    
    Following the pattern of storing errors in state so the LLM
    can see what went wrong and adjust its approach.
    """
    node_name: str
    error_type: Literal["transient", "llm_recoverable", "user_fixable", "unexpected"]
    error_message: str
    retry_count: int
    timestamp: str


class PendingHumanAction(TypedDict):
    """Information about pending human-in-the-loop action.
    
    Used when interrupt() is called to pause execution for human input.
    """
    action_type: Literal["code_confirmation", "intent_clarification", "general_review"]
    context: Dict[str, Any]  # Context needed for the human to make decision
    requested_at: str
    resume_node: str  # Node to resume execution at


class ChatAppState(TypedDict):
    """Main state schema for the chat application.
    
    Following LangGraph best practices of keeping state raw and
    formatting prompts on-demand.
    
    Attributes:
        messages: Conversation history (LangChain message objects)
        user_intent: Structured classification of user intent
        agent_results: Raw results from delegated agents
        errors: List of recoverable errors for LLM to see
        pending_human_action: Pending human-in-the-loop request
        metadata: Execution metadata for debugging and recovery
    """
    # Conversation history - stored as raw message objects
    messages: Annotated[List[BaseMessage], add]
    
    # User intent classification - raw structured data
    user_intent: Optional[UserIntent]
    
    # Agent delegation results - raw responses
    agent_results: Annotated[List[AgentResult], add]
    
    # Error tracking for LLM-recoverable errors
    errors: Annotated[List[ErrorState], add]
    
    # Human-in-the-loop state
    pending_human_action: Optional[PendingHumanAction]
    
    # Execution metadata
    metadata: Dict[str, Any]


class RoutingDecision(TypedDict):
    """Structured output for supervisor routing decisions.
    
    Used when the supervisor needs to classify intent and route
    to appropriate agents.
    """
    selected_agent: Literal["websearch", "knowledge_base", "code_interpreter", "supervisor_direct"]
    reasoning: str
    task_description: str
    requires_human_confirmation: bool


class HumanResponse(TypedDict):
    """Structured response from human-in-the-loop.
    
    Used when resuming after interrupt() with human input.
    """
    approved: bool
    edited_content: Optional[str]
    additional_context: Optional[str]
    action_taken: Literal["approved", "rejected", "modified", "cancelled"]


# Default empty state factory
def create_default_state() -> ChatAppState:
    """Create a default empty state for new conversations.
    
    Returns:
        ChatAppState: Empty state with proper defaults
    """
    return {
        "messages": [],
        "user_intent": None,
        "agent_results": [],
        "errors": [],
        "pending_human_action": None,
        "metadata": {
            "conversation_start": None,
            "last_agent_called": None,
            "total_agent_calls": 0
        }
    }


__all__ = [
    "ChatAppState",
    "UserIntent", 
    "AgentResult",
    "ErrorState",
    "PendingHumanAction",
    "RoutingDecision",
    "HumanResponse",
    "create_default_state"
]