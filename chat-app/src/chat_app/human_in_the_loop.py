"""Human-in-the-loop implementation using LangGraph interrupt().

This module provides functions for pausing execution to collect human input:
- Code execution confirmation
- Intent clarification for ambiguous requests
- General review workflows

Following LangGraph best practices (2026 docs):
- interrupt() must come first in the node (any code before it will re-run on resume)
- Do NOT wrap interrupt() in try/except blocks
- Side effects before interrupt() must be idempotent
- interrupt() values must be JSON-serializable
- Command(goto=...) must reference valid nodes in the supervisor graph topology

Graph Topology Reference (create_supervisor):
    Valid nodes: supervisor, websearch, knowledge_base, code_interpreter, __end__
    After HITL resolution, route to:
    - "supervisor": re-process with new context (default for most cases)
    - "__end__": terminate the conversation

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt

from chat_app.state import ChatAppState, HumanResponse

logger = logging.getLogger(__name__)


def request_code_execution_confirmation(
    state: ChatAppState,
    code_content: str,
    agent_name: str = "code_interpreter"
) -> Command:
    """Request human confirmation before executing code.

    Pauses graph execution via interrupt() and waits for human approval.
    On approval, routes back to the supervisor which can re-delegate to
    code_interpreter. On rejection, ends the conversation with a cancellation
    message.

    Args:
        state: Current graph state.
        code_content: The code that will be executed.
        agent_name: Name of the agent requesting execution.

    Returns:
        Command routing to "supervisor" (approved) or "__end__" (rejected).

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    # interrupt() MUST come first -- code before it re-runs on resume
    human_response: Dict[str, Any] = interrupt({
        "action_type": "code_confirmation",
        "message": "Code execution requested",
        "agent": agent_name,
        "code_to_execute": code_content,
        "request": "Please review the code above and confirm execution (approve/reject/edit)",
    })

    # Process human response after resume
    approved = human_response.get("approved", False)
    edited_content = human_response.get("edited_content")

    if approved:
        logger.info(
            "Code execution approved by human",
            extra={
                "agent_name": agent_name,
                "was_edited": edited_content is not None,
            }
        )
        # Route back to supervisor which can re-delegate to code_interpreter
        return Command(
            update={
                "pending_human_action": None,
                "metadata": {
                    **state.get("metadata", {}),
                    "code_execution_confirmed": True,
                    "code_was_edited": edited_content is not None,
                },
            },
            goto="supervisor",
        )
    else:
        logger.info("Code execution rejected by human")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Code execution was cancelled by user. How else can I help you?"
                    )
                ],
                "pending_human_action": None,
            },
            goto="__end__",
        )


def request_intent_clarification(state: ChatAppState) -> Command:
    """Request clarification when user intent is ambiguous.

    Pauses graph execution and presents clarification options to the user.
    After the user responds, routes back to the supervisor with updated
    intent classification so it can route appropriately.

    Args:
        state: Current graph state.

    Returns:
        Command routing to "supervisor" with updated user_intent.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    last_user_msg = ""
    messages = state.get("messages", [])
    if messages:
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_msg = msg.content
                break

    # interrupt() MUST come first
    human_response: Dict[str, Any] = interrupt({
        "action_type": "intent_clarification",
        "message": "I need some clarification",
        "context": {
            "last_user_message": last_user_msg,
            "conversation_history_length": len(messages),
        },
        "request": (
            "Your request could be handled in multiple ways. Please clarify: "
            "Are you looking for (1) web search results, "
            "(2) information from uploaded documents, or "
            "(3) code execution/analysis?"
        ),
    })

    # Process clarification
    clarification = human_response.get("clarification", "")
    selected_option = human_response.get("selected_option", "")

    # Map clarification to intent
    intent_mapping: Dict[str, str] = {
        "1": "websearch",
        "2": "knowledge_base",
        "3": "code_interpreter",
        "web": "websearch",
        "search": "websearch",
        "document": "knowledge_base",
        "documents": "knowledge_base",
        "file": "knowledge_base",
        "files": "knowledge_base",
        "code": "code_interpreter",
        "python": "code_interpreter",
        "calculate": "code_interpreter",
    }

    detected_intent = intent_mapping.get(selected_option.lower(), "general")

    logger.info(
        "Intent clarified by human",
        extra={
            "detected_intent": detected_intent,
            "selected_option": selected_option,
        }
    )

    # Route back to supervisor with updated intent -- supervisor will route
    # to the appropriate agent based on the clarified intent
    return Command(
        update={
            "user_intent": {
                "primary_intent": detected_intent,
                "confidence": "high",  # High confidence after clarification
                "requires_confirmation": False,
                "description": f"User clarified: {clarification}",
            },
            "pending_human_action": None,
        },
        goto="supervisor",
    )


def request_general_review(
    state: ChatAppState,
    content_to_review: str,
    review_context: str = ""
) -> Command:
    """Request human review of generated content.

    Pauses graph execution for human review of high-stakes or sensitive content.
    On approval, delivers the content to the user. On rejection, routes back
    to the supervisor to try a different approach.

    Args:
        state: Current graph state.
        content_to_review: The content requiring review.
        review_context: Additional context about what is being reviewed.

    Returns:
        Command routing to "__end__" (approved) or "supervisor" (rejected).

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    # interrupt() MUST come first
    human_response: Dict[str, Any] = interrupt({
        "action_type": "general_review",
        "message": "Please review the following content",
        "content": content_to_review,
        "context": review_context,
        "request": "Please approve, reject, or edit the content above",
    })

    approved = human_response.get("approved", False)
    edited_content = human_response.get("edited_content")

    if approved:
        final_content = edited_content or content_to_review
        logger.info("Content approved by human")
        return Command(
            update={
                "messages": [
                    AIMessage(content=final_content)
                ],
                "pending_human_action": None,
            },
            goto="__end__",
        )
    else:
        logger.info("Content rejected by human, routing back to supervisor")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Content was not approved. Let me try a different approach."
                    )
                ],
                "pending_human_action": None,
            },
            goto="supervisor",
        )


__all__ = [
    "request_code_execution_confirmation",
    "request_intent_clarification",
    "request_general_review",
]
