"""Message summarization and token counting for long conversations.

This module provides functionality to summarize conversation messages
when they exceed context window limits. Uses gpt-4o-mini for cost-effective
summarization and tiktoken for accurate token counting.

Token Counting:
    - Uses tiktoken with cl100k_base encoding (GPT-4o tokenizer)
    - Falls back to character-based estimation if tiktoken unavailable
    - Includes message overhead (~4 tokens per message)

Summarization Strategy:
    - Maintains a running summary of older messages
    - New messages are summarized and appended to existing summary
    - Summary is kept concise (<500 words) to minimize token usage

Last Grunted: 02/04/2026 06:30:00 PM PST
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import trim_messages as lc_trim_messages

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Use cl100k_base encoding (GPT-4, GPT-4o, GPT-3.5-turbo)
    _encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _encoding = None
    logger.warning(
        "tiktoken not installed, using approximate token counting. "
        "Install with: pip install tiktoken"
    )


# Message overhead per message (role, formatting, etc.)
MESSAGE_TOKEN_OVERHEAD: int = 4


async def summarize_messages(
    messages: List[BaseMessage],
    existing_summary: str = ""
) -> str:
    """Summarize conversation messages using gpt-4o-mini.
    
    Uses a cheap model (gpt-4o-mini) to create or extend a conversation summary.
    If an existing summary is provided, it extends it with new messages.
    
    Args:
        messages: List of messages to summarize.
        existing_summary: Optional existing summary to extend.
        
    Returns:
        str: New or extended summary.
        
    Example:
        >>> summary = await summarize_messages(old_messages)
        >>> extended = await summarize_messages(new_messages, summary)
        
    Note:
        On error, returns existing_summary or truncated messages as fallback.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    if not messages:
        return existing_summary
    
    model = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
    
    formatted = format_messages_for_summary(messages)
    
    if existing_summary:
        prompt = f"""You are maintaining a running summary of a conversation. 
Extend the existing summary with the new messages provided.

Keep the summary concise but include all important facts, decisions, and context.
Focus on information that would be useful for continuing the conversation.

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES TO ADD:
{formatted}

UPDATED SUMMARY:"""
    else:
        prompt = f"""Summarize the following conversation concisely.
Include key facts, decisions, user preferences, and important context.
Keep the summary under 500 words.
Focus on information that would be useful for continuing the conversation.

CONVERSATION:
{formatted}

SUMMARY:"""
    
    try:
        response = await model.ainvoke([SystemMessage(content=prompt)])
        summary = response.content.strip()
        
        logger.debug(
            "Generated conversation summary",
            extra={
                "message_count": len(messages),
                "summary_length": len(summary),
                "had_existing_summary": bool(existing_summary)
            }
        )
        
        return summary
        
    except Exception as e:
        logger.error(
            "Summarization failed, using fallback",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        # Fallback: return existing summary or truncated messages
        if existing_summary:
            return existing_summary
        return formatted[:500] + "..." if len(formatted) > 500 else formatted


def format_messages_for_summary(messages: List[BaseMessage]) -> str:
    """Format messages for summarization prompt.
    
    Truncates very long individual messages to prevent context overflow
    during summarization.
    
    Args:
        messages: List of messages to format.
        
    Returns:
        str: Formatted messages as newline-separated string.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    formatted: List[str] = []
    
    for msg in messages:
        role = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', '')
        
        # Truncate very long messages for summarization
        if len(content) > 500:
            content = content[:500] + "..."
        
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken or approximate estimation.
    
    Uses tiktoken with cl100k_base encoding when available for accurate
    counting. Falls back to character-based estimation (~4 chars/token)
    if tiktoken is not installed.
    
    Args:
        text: Text to count tokens for.
        
    Returns:
        int: Token count (exact with tiktoken, estimated otherwise).
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE and _encoding:
        return len(_encoding.encode(text))
    
    # Fallback: ~4 characters per token
    return len(text) // 4


def count_tokens_approximately(text: str) -> int:
    """Estimate token count for text (for LangChain compatibility).
    
    This function exists for compatibility with LangChain's trim_messages
    which expects a token_counter callable. Uses tiktoken when available.
    
    Args:
        text: Text to count tokens for.
        
    Returns:
        int: Token count.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    return count_tokens(text)


def count_message_tokens(messages: List[BaseMessage]) -> int:
    """Count total tokens in a list of messages.
    
    Includes per-message overhead for role and formatting tokens.
    Uses tiktoken for accurate counting when available.
    
    Args:
        messages: List of messages to count.
        
    Returns:
        int: Total token count including overhead.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    total = 0
    
    for msg in messages:
        # Add per-message overhead
        total += MESSAGE_TOKEN_OVERHEAD
        
        content = getattr(msg, 'content', '')
        total += count_tokens(content)
    
    return total


async def maybe_summarize(
    messages: List[BaseMessage],
    existing_summary: str = "",
    max_tokens: int = 100000,
    summarize_threshold: int = 120000
) -> Tuple[List[BaseMessage], str]:
    """Trim messages and summarize if they exceed threshold.
    
    Returns trimmed messages and updated summary. Messages are trimmed
    to keep the most recent ones within max_tokens, and removed messages
    are summarized to preserve context.
    
    Args:
        messages: Full conversation history.
        existing_summary: Current summary of older messages.
        max_tokens: Target max tokens after trimming (default: 100000).
        summarize_threshold: Threshold to trigger trimming (default: 120000).
        
    Returns:
        Tuple[List[BaseMessage], str]: (trimmed_messages, new_or_existing_summary)
        
    Strategy:
        1. Count total tokens in messages
        2. If under max_tokens, return unchanged
        3. If over threshold, trim to max_tokens using "last" strategy
        4. Summarize removed messages and combine with existing summary
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    total_tokens = count_message_tokens(messages)
    
    if total_tokens < max_tokens:
        # No trimming needed
        return messages, existing_summary
    
    logger.info(
        "Trimming messages and summarizing",
        extra={
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "message_count": len(messages)
        }
    )
    
    # Trim messages to fit within max_tokens
    # Keep most recent messages, starting from human message
    trimmed = lc_trim_messages(
        messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=max_tokens,
        start_on="human",
        end_on=("human", "tool"),
    )
    
    # If we trimmed, summarize the removed messages
    if len(trimmed) < len(messages):
        removed_count = len(messages) - len(trimmed)
        removed_messages = messages[:removed_count]
        
        logger.debug(
            "Summarizing removed messages",
            extra={
                "removed_count": removed_count,
                "kept_count": len(trimmed)
            }
        )
        
        new_summary = await summarize_messages(removed_messages, existing_summary)
        return list(trimmed), new_summary
    
    return list(messages), existing_summary


__all__ = [
    "summarize_messages",
    "format_messages_for_summary",
    "count_tokens",
    "count_tokens_approximately",
    "count_message_tokens",
    "maybe_summarize",
    "TIKTOKEN_AVAILABLE",
    "MESSAGE_TOKEN_OVERHEAD",
]
