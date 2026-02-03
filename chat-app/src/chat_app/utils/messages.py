"""Utilities for converting messages between dict and LangChain formats.

Provides functions to transform message dictionaries into LangChain message
types (HumanMessage, AIMessage, etc.) for use with LangGraph supervisor and
agent graphs.

Last Grunted: 02/03/2026 03:45:00 PM PST
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def _to_langchain_message(msg):
    """Convert a message dict to a LangChain message object.

    Maps role strings to appropriate LangChain message types.

    Args:
        msg: A dict with "role" and "content" keys. Handles None values
            gracefully by defaulting to empty strings.

    Returns:
        One of:
            - HumanMessage: For roles "user", "human", or unknown
            - AIMessage: For roles "assistant", "ai"
            - SystemMessage: For role "system"

    Example:
        >>> _to_langchain_message({"role": "user", "content": "Hello"})
        HumanMessage(content="Hello")
        >>> _to_langchain_message({"role": "assistant", "content": "Hi!"})
        AIMessage(content="Hi!")

    Last Grunted: 02/03/2026 03:45:00 PM PST
    """
    if isinstance(msg, dict):
        role = (msg.get("role", "") or "").lower()
        content = msg.get("content", "") or ""
    else:
        role = (getattr(msg, "role", "") or "").lower()
        content = getattr(msg, "content", "") or ""

    if role in ("user", "human"):
        return HumanMessage(content=content)
    if role in ("assistant", "ai"):
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    # Fallback to human
    return HumanMessage(content=content)


def build_input_messages(request):
    """Build LangChain message list from a request object.

    Converts all messages from the request (memory + latest messages) into
    LangChain format for the supervisor graph.

    Args:
        request: Request object with optional attributes:
            - memory: List of previous conversation messages
            - messages: List of latest user messages

    Returns:
        list: List of LangChain message objects (HumanMessage, AIMessage, etc.).
            Falls back to [HumanMessage(content="Hello")] if no messages provided.

    Note:
        Memory messages are processed first, then latest messages, preserving
        chronological order. Conversion errors are silently ignored to ensure
        the graph always receives some input.

    Last Grunted: 02/03/2026 03:45:00 PM PST
    """
    input_messages = []
    
    # Convert memory messages to LangChain format
    if getattr(request, "memory", None):
        try:
            mem_msgs = [_to_langchain_message(m) for m in request.memory]
            input_messages.extend(mem_msgs)
        except Exception:
            pass
    
    # Convert latest messages to LangChain format
    if getattr(request, "messages", None):
        try:
            latest_msgs = [_to_langchain_message(m) for m in request.messages]
            input_messages.extend(latest_msgs)
        except Exception:
            pass
    
    # Fallback if no messages provided
    if not input_messages:
        input_messages = [HumanMessage(content="Hello")]
    
    return input_messages
