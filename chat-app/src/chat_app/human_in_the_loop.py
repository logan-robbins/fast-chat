"""Human-in-the-loop implementation using LangGraph interrupt().

This module provides functions for pausing execution to collect human input:
- Code execution confirmation
- Intent clarification for ambiguous requests
- General review workflows

Following LangGraph best practices, interrupt() must come first in the node
(any code before it will re-run on resume).

Last Grunted: 02/04/2026 03:30:00 PM PST
"""
from datetime import datetime
from typing import Literal, Dict, Any, Optional
from langgraph.types import interrupt, Command

from chat_app.state import ChatAppState, HumanResponse, PendingHumanAction


def request_code_execution_confirmation(
    state: ChatAppState,
    code_content: str,
    agent_name: str = "code_interpreter"
) -> Command:
    """Request human confirmation before executing code.
    
    interrupt() must come first - any code before it will re-run on resume.
    
    Args:
        state: Current graph state
        code_content: The code that will be executed
        agent_name: Name of the agent requesting execution
        
    Returns:
        Command with updated state and next node
    """
    # interrupt() MUST come first - this is the LangGraph pattern
    human_response = interrupt({
        "action_type": "code_confirmation",
        "message": "Code execution requested",
        "agent": agent_name,
        "code_to_execute": code_content,
        "context": {
            "conversation_length": len(state.get("messages", [])),
            "previous_agent_calls": state.get("metadata", {}).get("total_agent_calls", 0)
        },
        "request": "Please review the code above and confirm execution (approve/reject/edit)"
    })
    
    # Process human response after resume
    response = HumanResponse(
        approved=human_response.get("approved", False),
        edited_content=human_response.get("edited_content"),
        additional_context=human_response.get("additional_context"),
        action_taken="approved" if human_response.get("approved") else "rejected"
    )
    
    if response["approved"]:
        # Use edited content if provided, otherwise use original
        final_code = response["edited_content"] or code_content
        return Command(
            update={
                "metadata": {
                    **state.get("metadata", {}),
                    "code_execution_confirmed": True,
                    "code_was_edited": response["edited_content"] is not None
                }
            },
            goto="execute_code"
        )
    else:
        # User rejected - return error message
        return Command(
            update={
                "messages": state.get("messages", []) + [
                    {
                        "role": "assistant",
                        "content": "Code execution was cancelled by user. How else can I help you?"
                    }
                ],
                "pending_human_action": None
            },
            goto="__end__"
        )


def request_intent_clarification(state: ChatAppState) -> Command:
    """Request clarification when user intent is ambiguous.
    
    Used when confidence in routing decision is low.
    
    Args:
        state: Current graph state
        
    Returns:
        Command with clarification response
    """
    # interrupt() MUST come first
    human_response = interrupt({
        "action_type": "intent_clarification",
        "message": "I need some clarification",
        "context": {
            "last_user_message": state["messages"][-1].content if state.get("messages") else "",
            "conversation_history_length": len(state.get("messages", []))
        },
        "request": "Your request could be handled in multiple ways. Please clarify: Are you looking for (1) web search results, (2) information from uploaded documents, or (3) code execution/analysis?"
    })
    
    # Process clarification
    clarification = human_response.get("clarification", "")
    selected_option = human_response.get("selected_option", "")
    
    # Map clarification to intent
    intent_mapping = {
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
        "calculate": "code_interpreter"
    }
    
    detected_intent = intent_mapping.get(selected_option.lower(), "general")
    
    return Command(
        update={
            "user_intent": {
                "primary_intent": detected_intent,
                "confidence": "high",  # High confidence after clarification
                "requires_confirmation": False,
                "description": f"User clarified: {clarification}"
            },
            "pending_human_action": None
        },
        goto=f"delegate_to_{detected_intent}"
    )


def request_general_review(
    state: ChatAppState,
    content_to_review: str,
    review_context: str = ""
) -> Command:
    """Request human review of generated content.
    
    Used for high-stakes or sensitive content that needs approval.
    
    Args:
        state: Current graph state
        content_to_review: The content requiring review
        review_context: Additional context about what is being reviewed
        
    Returns:
        Command with review response
    """
    # interrupt() MUST come first
    human_response = interrupt({
        "action_type": "general_review",
        "message": "Please review the following content",
        "content": content_to_review,
        "context": review_context,
        "request": "Please approve, reject, or edit the content above"
    })
    
    response = HumanResponse(
        approved=human_response.get("approved", False),
        edited_content=human_response.get("edited_content"),
        additional_context=human_response.get("additional_context"),
        action_taken="approved" if human_response.get("approved") else "rejected"
    )
    
    if response["approved"]:
        final_content = response["edited_content"] or content_to_review
        return Command(
            update={
                "messages": state.get("messages", []) + [
                    {
                        "role": "assistant", 
                        "content": final_content
                    }
                ]
            },
            goto="__end__"
        )
    else:
        return Command(
            update={
                "messages": state.get("messages", []) + [
                    {
                        "role": "assistant",
                        "content": "Content was not approved. Let me try a different approach."
                    }
                ]
            },
            goto="retry_with_feedback"
        )


__all__ = [
    "request_code_execution_confirmation",
    "request_intent_clarification", 
    "request_general_review"
]