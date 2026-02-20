"""State schema for Chat-App following LangGraph best practices.

This module defines the TypedDict state schema for type-safe state management.
Following the LangGraph documentation pattern of storing raw data, not formatted text.

State Design Principles:
- Store raw data that cannot be reconstructed
- Store classification/intent results needed by multiple nodes
- Store expensive-to-fetch data (search results, external API responses)
- Store execution metadata for debugging and recovery
- Format prompts on-demand inside nodes, not in state

Status Streaming (ChatGPT/Claude 2025/2026 patterns):
- StatusEvent provides user-friendly status updates during streaming
- Follows OpenAI Responses API patterns: in_progress → searching/thinking → completed
- Hides internal implementation details from users

Last Grunted: 02/04/2026 07:30:00 PM PST
"""
from typing import TypedDict, Literal, Optional, List, Dict, Any, Annotated
from operator import add
from langchain_core.messages import BaseMessage


# -----------------------------------------------------------------------------
# Status Streaming Types (ChatGPT/Claude-style status updates)
# -----------------------------------------------------------------------------

class StatusEvent(TypedDict, total=False):
    """User-friendly status event for streaming UI updates.
    
    Follows ChatGPT/Claude 2025/2026 patterns:
    - "Thinking..." for reasoning/routing
    - "Searching the web..." for web search
    - "Looking through your documents..." for RAG
    - "Running Python code..." for code interpreter
    
    OpenAI Responses API Pattern Reference:
    - response.web_search_call: in_progress → searching → completed
    - response.file_search_call: in_progress → searching → completed
    - response.code_interpreter_call: in_progress → interpreting → completed
    
    Attributes:
        type: Status event type (thinking, tool_start, tool_progress, tool_complete, agent_handoff)
        message: User-friendly message to display (required)
        agent: Agent name if applicable (websearch, knowledge_base, code_interpreter)
        tool: Tool name if applicable (perplexity_search, search_document_content, python_repl)
        details: Optional structured data for rich UI rendering
        timestamp: ISO timestamp of the event
    
    Example:
        >>> status: StatusEvent = {
        ...     "type": "tool_start",
        ...     "message": "Searching the web for Python 3.12 features...",
        ...     "agent": "websearch",
        ...     "tool": "perplexity_search"
        ... }
    """
    type: Literal["thinking", "tool_start", "tool_progress", "tool_complete", "agent_handoff", "error"]
    message: str  # Required - user-friendly message
    agent: Optional[str]  # Agent name (websearch, knowledge_base, code_interpreter)
    tool: Optional[str]  # Tool name
    details: Optional[Dict[str, Any]]  # Optional structured data
    timestamp: Optional[str]  # ISO timestamp


# Status event type constants for type safety
STATUS_THINKING = "thinking"
STATUS_TOOL_START = "tool_start"
STATUS_TOOL_PROGRESS = "tool_progress"
STATUS_TOOL_COMPLETE = "tool_complete"
STATUS_AGENT_HANDOFF = "agent_handoff"
STATUS_ERROR = "error"


# User-friendly message templates
STATUS_MESSAGES = {
    # Supervisor/Thinking
    "supervisor_thinking": "Thinking...",
    "supervisor_routing": "Analyzing your request...",
    
    # WebSearch Agent
    "websearch_handoff": "Searching the web...",
    "websearch_start": "Searching the web for: {query}",
    "websearch_complete": "Web search complete",
    
    # Knowledge Base / RAG Agent  
    "knowledge_base_handoff": "Looking through your documents...",
    "rag_start": "Searching documents for: {query}",
    "rag_searching": "Searching collection {current}/{total}: {collection}",
    "rag_complete": "Found {count} relevant documents",
    
    # Code Interpreter Agent
    "code_interpreter_handoff": "Running Python code...",
    "code_start": "Executing Python code...",
    "code_complete": "Code execution complete",
    
    # Generic
    "agent_complete": "Analysis complete",
    "error": "An error occurred: {error}",
}


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


class TokenUsage(TypedDict):
    """Token usage statistics for context window tracking.
    
    Enables real-time context window visualization in the UI.
    Updated during streaming to provide running totals.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_window_limit: int  # Model's max context window
    context_utilization_pct: float  # Percentage of context used


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
        conversation_summary: Running summary of older messages
        token_usage: Current token usage for context tracking
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
    
    # Running summary of conversation (for long conversations)
    conversation_summary: Optional[str]
    
    # Token usage for context window tracking
    token_usage: Optional[TokenUsage]
    
    # Execution metadata
    metadata: Dict[str, Any]
    
    # LangGraph 1.0+ requires remaining_steps for agent recursion tracking
    remaining_steps: int

    # Ephemeral prompt context for dynamic system instructions
    llm_input_messages: Optional[List[BaseMessage]]


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
        "conversation_summary": None,
        "token_usage": None,
        "metadata": {
            "conversation_start": None,
            "last_agent_called": None,
            "total_agent_calls": 0
        }
    }


__all__ = [
    # State schemas
    "ChatAppState",
    "UserIntent", 
    "AgentResult",
    "ErrorState",
    "PendingHumanAction",
    "RoutingDecision",
    "HumanResponse",
    "TokenUsage",
    "create_default_state",
    # Status streaming (ChatGPT/Claude-style)
    "StatusEvent",
    "STATUS_THINKING",
    "STATUS_TOOL_START",
    "STATUS_TOOL_PROGRESS",
    "STATUS_TOOL_COMPLETE",
    "STATUS_AGENT_HANDOFF",
    "STATUS_ERROR",
    "STATUS_MESSAGES",
]