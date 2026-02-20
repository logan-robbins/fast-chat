"""Auto-generate conversation titles from first user message.

This module provides functionality to automatically generate concise,
descriptive titles for conversation threads based on the first user message.
It uses the chat-app service to generate titles using an LLM.

Uses the shared HTTP client for efficient connection pooling.

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import structlog
from typing import Any, Optional

import httpx

from src.services.http_client import get_client, CHAT_APP_URL

logger = structlog.get_logger(__name__)

# Title generation prompt template
TITLE_PROMPT_TEMPLATE = """Generate a short, descriptive title (max 5 words) for a conversation that starts with this message:

Message: "{message}"

Title:"""

# Maximum length for message preview in prompt
MAX_MESSAGE_PREVIEW_LENGTH: int = 200

# Maximum length for generated title
MAX_TITLE_LENGTH: int = 50

# Default model for title generation (cheap and fast)
DEFAULT_TITLE_MODEL: str = "gpt-4o-mini"


def _create_fallback_title(message: str) -> str:
    """
    Create a fallback title from the first few words of a message.
    
    Args:
        message: The user message
        
    Returns:
        str: Fallback title (first 5 words, max 50 chars)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    if not message:
        return "New Chat"
    
    words = message.split()[:5]
    fallback = " ".join(words)[:MAX_TITLE_LENGTH]
    return fallback if fallback.strip() else "New Chat"


def _clean_title(title: str) -> str:
    """
    Clean and normalize a generated title.
    
    Removes quotes, excessive whitespace, and truncates to max length.
    
    Args:
        title: Raw title from LLM
        
    Returns:
        str: Cleaned title
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Strip whitespace
    title = title.strip()
    
    # Remove surrounding quotes
    if (title.startswith('"') and title.endswith('"')) or \
       (title.startswith("'") and title.endswith("'")):
        title = title[1:-1]
    
    # Also strip any remaining quote characters from edges
    title = title.strip('"\'')
    
    # Truncate to max length
    return title[:MAX_TITLE_LENGTH]


def _extract_text_content(value: Any) -> str:
    """Normalize model message content payloads to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                if item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
        return "".join(parts).strip() if parts else str(value)

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(value)


async def generate_title(
    first_message: str,
    model: str = DEFAULT_TITLE_MODEL,
    timeout: float = 10.0
) -> Optional[str]:
    """
    Generate a concise title from the first user message.
    
    Uses the chat-app service to generate a short, descriptive title
    (max 5 words) for a conversation based on the first user message.
    
    Uses the shared HTTP client for connection pooling efficiency.
    
    Args:
        first_message: The first message from the user
        model: The model to use for title generation
        timeout: Request timeout in seconds
        
    Returns:
        Optional[str]: Generated title or None if generation fails
        
    Example:
        >>> title = await generate_title("How do I implement a binary search tree in Python?")
        >>> print(title)
        "Binary Search Tree Implementation"
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    if not first_message or not first_message.strip():
        return None
    
    # Truncate message for prompt
    message_preview = first_message[:MAX_MESSAGE_PREVIEW_LENGTH]
    prompt = TITLE_PROMPT_TEMPLATE.format(message=message_preview)
    
    try:
        client = await get_client()
        
        response = await client.post(
            CHAT_APP_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.3,
                "stream": False
            },
            timeout=httpx.Timeout(timeout)
        )
        
        if response.status_code != 200:
            logger.warning(
                "title_generation.backend_error",
                status_code=response.status_code,
                model=model
            )
            return None
        
        data = response.json()
        
        if "choices" not in data or not data["choices"]:
            logger.warning("title_generation.empty_response", model=model)
            return None
        
        raw_title = _extract_text_content(data["choices"][0].get("message", {}).get("content", ""))
        title = _clean_title(raw_title)
        
        if not title:
            return None
        
        logger.debug(
            "title_generation.success",
            model=model,
            title=title
        )
        return title
        
    except httpx.TimeoutException:
        logger.warning(
            "title_generation.timeout",
            timeout=timeout,
            model=model
        )
        return None
        
    except Exception as e:
        logger.warning(
            "title_generation.error",
            error=str(e),
            model=model
        )
        return None


async def generate_title_with_fallback(
    first_message: str,
    model: str = DEFAULT_TITLE_MODEL
) -> str:
    """
    Generate a title with fallback to first words.
    
    This version always returns a title, falling back to the first
    few words of the message if LLM generation fails.
    
    Args:
        first_message: The first message from the user
        model: The model to use for title generation
        
    Returns:
        str: Generated or fallback title (never empty)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    title = await generate_title(first_message, model)
    
    if title:
        return title
    
    return _create_fallback_title(first_message)
