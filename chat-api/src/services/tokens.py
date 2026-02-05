"""
Token counting utilities using tiktoken.

Provides accurate token counting for OpenAI models using the official
tiktoken library. Falls back to character-based estimation if tiktoken
is unavailable or for unsupported models.

Supported encodings:
    - o200k_base: gpt-4o, gpt-4o-mini, o1, o3 series
    - cl100k_base: gpt-4-turbo, gpt-4, gpt-3.5-turbo

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import logging
from functools import lru_cache
from typing import List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Try to import tiktoken, fall back to estimation if unavailable
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning(
        "tiktoken not installed - using character-based token estimation. "
        "Install with: pip install tiktoken"
    )
    TIKTOKEN_AVAILABLE = False


# ============================================================================
# Encoding Cache
# ============================================================================

@lru_cache(maxsize=10)
def _get_encoding(model: str):
    """
    Get the tiktoken encoding for a model (cached).
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4-turbo")
        
    Returns:
        tiktoken.Encoding or None if unavailable
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Model not found, try to infer encoding from model name
        if any(m in model.lower() for m in ["gpt-4o", "o1", "o3"]):
            return tiktoken.get_encoding("o200k_base")
        elif any(m in model.lower() for m in ["gpt-4", "gpt-3.5"]):
            return tiktoken.get_encoding("cl100k_base")
        else:
            # Default to latest encoding
            logger.debug(f"Unknown model '{model}', using o200k_base encoding")
            return tiktoken.get_encoding("o200k_base")


# ============================================================================
# Token Counting Functions
# ============================================================================

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in a text string.
    
    Uses tiktoken for accurate counting, falls back to character-based
    estimation (~4 chars/token) if tiktoken is unavailable.
    
    Args:
        text: The text to count tokens for
        model: Model name for encoding selection
        
    Returns:
        int: Token count (minimum 0)
        
    Example:
        >>> count_tokens("Hello, world!", "gpt-4o")
        4
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    if not text:
        return 0
    
    encoding = _get_encoding(model)
    
    if encoding is not None:
        return len(encoding.encode(text))
    
    # Fallback: ~4 characters per token for English text
    return max(1, len(text) // 4)


def estimate_tokens(text: str) -> int:
    """
    Quick token estimation without model-specific encoding.
    
    Uses ~4 characters per token heuristic. Useful for rough estimates
    where accuracy is less critical.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        int: Estimated token count (minimum 0 for empty, minimum 1 for non-empty)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_message_tokens(
    messages: List[dict],
    model: str = "gpt-4o"
) -> int:
    """
    Count tokens in a list of chat messages.
    
    Includes overhead for message structure per OpenAI's token counting:
    - ~4 tokens overhead per message for role/structure
    - Content tokens based on actual encoding
    - ~3 tokens for priming
    
    Args:
        messages: List of message dicts with "role" and "content"
        model: Model name for encoding selection
        
    Returns:
        int: Total token count for the messages
        
    Reference:
        https://platform.openai.com/docs/guides/text-generation/managing-tokens
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    total = 0
    
    for message in messages:
        # Message overhead (~4 tokens for role, name, etc.)
        total += 4
        
        # Count content tokens
        content = message.get("content", "")
        if content:
            if isinstance(content, str):
                total += count_tokens(content, model)
            elif isinstance(content, list):
                # Handle array content (multimodal)
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            total += count_tokens(text, model)
        
        # Count name if present
        name = message.get("name", "")
        if name:
            total += count_tokens(name, model)
    
    # Add priming tokens
    total += 3
    
    return total


def count_chat_message_tokens(
    messages: List[Any],
    model: str = "gpt-4o"
) -> int:
    """
    Count tokens for ChatMessage Pydantic objects.
    
    Similar to count_message_tokens but works with Pydantic models
    that have role, content, and name attributes.
    
    Args:
        messages: List of ChatMessage objects
        model: Model name for encoding selection
        
    Returns:
        int: Total token count
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    total = 0
    
    for msg in messages:
        total += 4  # Message overhead
        
        if hasattr(msg, "content") and msg.content:
            total += count_tokens(msg.content, model)
        
        if hasattr(msg, "name") and msg.name:
            total += count_tokens(msg.name, model)
    
    total += 3  # Priming tokens
    
    return total


def count_input_tokens(
    input_data: Optional[Union[str, List[Any]]],
    instructions: Optional[str] = None,
    tools: Optional[List[dict]] = None,
    model: str = "gpt-4o"
) -> int:
    """
    Count estimated input tokens for a Responses API request.
    
    Includes tokens from:
    - Input text/messages
    - System instructions
    - Tool definitions
    - Message structure overhead
    
    Args:
        input_data: String or list of input items
        instructions: System/developer message
        tools: List of tool definitions
        model: Model name for encoding
        
    Returns:
        int: Estimated total input tokens
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    total = 0
    
    # Count instructions
    if instructions:
        total += count_tokens(instructions, model) + 4  # +4 for message overhead
    
    # Count input
    if isinstance(input_data, str):
        total += count_tokens(input_data, model) + 4
    elif isinstance(input_data, list):
        for item in input_data:
            total += 4  # Message overhead
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    total += count_tokens(content, model)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            text = c.get("text", "")
                            if text:
                                total += count_tokens(text, model)
            elif hasattr(item, "content"):
                content = item.content
                if isinstance(content, str):
                    total += count_tokens(content, model)
    
    # Count tools (rough estimate: ~50 base + description)
    if tools:
        for tool in tools:
            total += 50  # Base tool overhead
            if isinstance(tool, dict):
                desc = tool.get("description", "")
                if desc:
                    total += count_tokens(desc, model)
    
    # Add base overhead
    total += 3  # Priming tokens
    
    return total


# ============================================================================
# Model Context Limits
# ============================================================================

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # GPT-4o series
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    
    # GPT-4 series
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    
    # GPT-3.5 series
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    
    # O-series (reasoning models)
    "o1": 200000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
}


def get_context_limit(model: str) -> int:
    """
    Get the context window limit for a model.
    
    Args:
        model: Model name
        
    Returns:
        int: Maximum context tokens (defaults to 128000 for unknown models)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Direct match
    if model in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model]
    
    # Try base model name match
    for known_model, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(known_model):
            return limit
    
    # Default to common limit
    return 128000


def check_context_overflow(
    token_count: int,
    model: str,
    max_output_tokens: Optional[int] = None
) -> tuple[bool, int]:
    """
    Check if token count exceeds model context limit.
    
    Args:
        token_count: Number of input tokens
        model: Model name
        max_output_tokens: Reserved tokens for output
        
    Returns:
        tuple[bool, int]: (is_overflow, tokens_over_limit)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    context_limit = get_context_limit(model)
    output_reserve = max_output_tokens or (context_limit // 4)  # Default: 25% for output
    
    available = context_limit - output_reserve
    overflow = token_count - available
    
    return overflow > 0, max(0, overflow)
