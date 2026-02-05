"""Long-term memory management for user data via semantic search.

This module provides persistent memory storage for user preferences,
facts, and other data that should persist across conversations.
Uses the langgraph-store-postgres package for vector-based semantic search.

Architecture:
    - PostgresStore: Production storage with pgvector semantic search
    - In-memory dict: Development fallback when database unavailable
    
Memory Types:
    - "preference": User preferences (language, tone, etc.)
    - "fact": Facts about the user (name, role, etc.)
    - "general": General memories and context

Namespace Structure:
    Memories are stored in namespaces: ("memories", user_id)
    This provides per-user isolation for multi-tenant applications.

Last Grunted: 02/04/2026 06:30:00 PM PST
"""
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from chat_app.config import get_settings

logger = logging.getLogger(__name__)

# Type alias for memory types
MemoryType = Literal["preference", "fact", "general"]

# Try to import langgraph-store-postgres
try:
    from langchain.embeddings import init_embeddings
    from langgraph.store.postgres import PostgresStore
    LANGGRAPH_STORE_AVAILABLE = True
except ImportError:
    LANGGRAPH_STORE_AVAILABLE = False
    logger.warning(
        "langgraph-store-postgres not installed. "
        "Install with: pip install langgraph-store-postgres"
    )


# Global store instance (lazy loaded)
_memory_store: Optional[Union["PostgresStore", Dict]] = None


def get_memory_store() -> Union["PostgresStore", Dict]:
    """Get or initialize the long-term memory store singleton.
    
    Returns a PostgresStore instance configured for semantic search
    using OpenAI text-embedding-3-small (1536 dimensions). Falls back
    to an in-memory dict if the database is unavailable.
    
    The store is lazily initialized on first access and cached globally.
    
    Returns:
        PostgresStore | dict: Configured memory store for semantic search.
            - PostgresStore: Full semantic search with pgvector
            - dict: In-memory fallback (substring matching only)
            
    Note:
        Call this function to get the store; do not access _memory_store directly.
        The store is thread-safe for concurrent access.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    global _memory_store
    
    if _memory_store is not None:
        return _memory_store
    
    if not LANGGRAPH_STORE_AVAILABLE:
        logger.info("Using in-memory storage (langgraph-store-postgres not available)")
        _memory_store = {}
        return _memory_store
    
    settings = get_settings()
    
    if not settings.database_url:
        logger.info("Using in-memory storage (DATABASE_URL not configured)")
        _memory_store = {}
        return _memory_store
    
    try:
        embeddings = init_embeddings("openai:text-embedding-3-small")
        
        store = PostgresStore(
            settings.database_url,
            index={
                "embed": embeddings,
                "dims": 1536,
            }
        )
        _memory_store = store
        logger.info("PostgresStore initialized for semantic memory search")
        return store
    except Exception as e:
        logger.error(
            "Failed to initialize PostgresStore, falling back to in-memory",
            extra={"error": str(e), "error_type": type(e).__name__}
        )
        _memory_store = {}
        return _memory_store


async def store_user_memory(
    user_id: str, 
    content: str, 
    memory_type: MemoryType = "general",
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Store a memory for a user with semantic search indexing.
    
    Creates a new memory entry in the user's namespace. The memory is
    automatically indexed for semantic search (when using PostgresStore).
    
    Args:
        user_id: The user's unique identifier (used as namespace).
        content: The memory content to store (indexed for semantic search).
        memory_type: Category of memory for filtering:
            - "preference": User preferences (language, style, etc.)
            - "fact": Facts about the user (name, role, etc.)
            - "general": General memories and context
        metadata: Additional metadata to store with the memory.
        
    Returns:
        str | None: UUID of the stored memory if successful, None on error.
        
    Example:
        >>> memory_id = await store_user_memory(
        ...     "user-123", 
        ...     "User prefers Python over JavaScript",
        ...     "preference"
        ... )
        >>> print(memory_id)
        "550e8400-e29b-41d4-a716-446655440000"
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    store = get_memory_store()
    
    memory_id = str(uuid.uuid4())
    namespace = ("memories", user_id)
    
    data: Dict[str, Any] = {
        "text": content,
        "type": memory_type,
        "metadata": metadata or {}
    }
    
    if isinstance(store, dict):
        # In-memory fallback
        if namespace not in store:
            store[namespace] = {}
        store[namespace][memory_id] = data
        logger.debug(
            "Stored memory (in-memory)",
            extra={"user_id": user_id, "memory_id": memory_id, "type": memory_type}
        )
        return memory_id
    
    # PostgresStore
    try:
        store.put(namespace, memory_id, data)
        logger.debug(
            "Stored memory",
            extra={"user_id": user_id, "memory_id": memory_id, "type": memory_type}
        )
        return memory_id
    except Exception as e:
        logger.error(
            "Failed to store memory",
            extra={
                "user_id": user_id,
                "memory_type": memory_type,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        return None


async def search_user_memories(
    user_id: str, 
    query: str, 
    limit: int = 5,
    memory_type: Optional[MemoryType] = None
) -> List[Dict[str, Any]]:
    """Search user memories by semantic similarity.
    
    Uses vector similarity search (PostgresStore) or substring matching
    (in-memory fallback) to find relevant memories based on the query.
    
    Args:
        user_id: The user's unique identifier (namespace key).
        query: Natural language query for semantic matching.
        limit: Maximum number of results to return (default: 5).
        memory_type: Optional filter by memory type ("preference", "fact", "general").
        
    Returns:
        List[Dict]: List of matching memories, each containing:
            - id: Memory UUID
            - text: Memory content
            - type: Memory type ("preference", "fact", "general")
            - metadata: Additional stored metadata
            - score: Similarity score (PostgresStore only, None for in-memory)
            
    Example:
        >>> memories = await search_user_memories(
        ...     "user-123", 
        ...     "What programming languages does the user prefer?"
        ... )
        >>> for mem in memories:
        ...     print(f"{mem['type']}: {mem['text']} (score: {mem.get('score')})")
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    store = get_memory_store()
    namespace = ("memories", user_id)
    
    if isinstance(store, dict):
        # In-memory fallback - simple substring search
        if namespace not in store:
            return []
        
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()
        
        for memory_id, data in store[namespace].items():
            if memory_type and data.get("type") != memory_type:
                continue
            # Simple substring matching for fallback
            if query_lower in data.get("text", "").lower():
                results.append({
                    "id": memory_id,
                    "text": data["text"],
                    "type": data.get("type", "general"),
                    "metadata": data.get("metadata", {}),
                    "score": None  # No score for in-memory search
                })
        return results[:limit]
    
    # PostgresStore with semantic search
    try:
        results = store.search(namespace, query=query, limit=limit)
        
        filtered_results: List[Dict[str, Any]] = []
        for item in results:
            item_type = item.value.get("type", "general")
            if memory_type and item_type != memory_type:
                continue
            filtered_results.append({
                "id": item.key,
                "text": item.value.get("text", ""),
                "type": item_type,
                "metadata": item.value.get("metadata", {}),
                "score": getattr(item, "score", None)
            })
        
        logger.debug(
            "Memory search completed",
            extra={
                "user_id": user_id,
                "query_preview": query[:50],
                "results_count": len(filtered_results)
            }
        )
        return filtered_results
        
    except Exception as e:
        logger.error(
            "Memory search failed",
            extra={
                "user_id": user_id,
                "query_preview": query[:50],
                "error": str(e),
                "error_type": type(e).__name__
            }
        )
        return []


async def get_user_memories(
    user_id: str,
    memory_type: Optional[MemoryType] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get all memories for a user, optionally filtered by type.
    
    Retrieves memories without semantic filtering. Use search_user_memories()
    when you need relevance-based retrieval.
    
    Args:
        user_id: The user's unique identifier (namespace key).
        memory_type: Optional filter by memory type.
        limit: Maximum number of memories to return (default: 50).
        
    Returns:
        List[Dict]: List of memories, each containing:
            - id: Memory UUID
            - text: Memory content
            - type: Memory type
            - metadata: Additional stored metadata
            
    Note:
        For PostgresStore, this uses a broad semantic query as a workaround
        since list operations aren't directly supported. Consider using
        search_user_memories() for better results.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    store = get_memory_store()
    namespace = ("memories", user_id)
    
    if isinstance(store, dict):
        # In-memory fallback
        if namespace not in store:
            return []
        
        results: List[Dict[str, Any]] = []
        for memory_id, data in list(store[namespace].items())[:limit]:
            if memory_type and data.get("type") != memory_type:
                continue
            results.append({
                "id": memory_id,
                "text": data["text"],
                "type": data.get("type", "general"),
                "metadata": data.get("metadata", {})
            })
        return results
    
    # PostgresStore - use search with broad query
    try:
        # Note: PostgresStore doesn't have a direct list method,
        # so we use a broad semantic query to retrieve memories
        results = store.search(
            namespace, 
            query="user preferences facts information context",
            limit=limit
        )
        
        filtered_results: List[Dict[str, Any]] = []
        for item in results:
            item_type = item.value.get("type", "general")
            if memory_type and item_type != memory_type:
                continue
            filtered_results.append({
                "id": item.key,
                "text": item.value.get("text", ""),
                "type": item_type,
                "metadata": item.value.get("metadata", {})
            })
        return filtered_results
        
    except Exception as e:
        logger.error(
            "Failed to get user memories",
            extra={
                "user_id": user_id,
                "memory_type": memory_type,
                "error": str(e)
            }
        )
        return []


async def delete_user_memory(user_id: str, memory_id: str) -> bool:
    """Delete a specific user memory by ID.
    
    Permanently removes a memory from the user's namespace.
    
    Args:
        user_id: The user's unique identifier (namespace key).
        memory_id: The UUID of the memory to delete.
        
    Returns:
        bool: True if memory was deleted successfully, False if not found or error.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    store = get_memory_store()
    namespace = ("memories", user_id)
    
    if isinstance(store, dict):
        # In-memory fallback
        if namespace in store and memory_id in store[namespace]:
            del store[namespace][memory_id]
            logger.debug(
                "Deleted memory (in-memory)",
                extra={"user_id": user_id, "memory_id": memory_id}
            )
            return True
        return False
    
    # PostgresStore
    try:
        store.delete(namespace, memory_id)
        logger.debug(
            "Deleted memory",
            extra={"user_id": user_id, "memory_id": memory_id}
        )
        return True
    except Exception as e:
        logger.error(
            "Failed to delete memory",
            extra={
                "user_id": user_id,
                "memory_id": memory_id,
                "error": str(e)
            }
        )
        return False


def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """Format memories as a markdown string for system prompt injection.
    
    Converts a list of memory dictionaries into a human-readable format
    suitable for including in LLM system prompts.
    
    Args:
        memories: List of memory dictionaries from search_user_memories().
        
    Returns:
        str: Markdown-formatted memories, or empty string if no memories.
        
    Example:
        >>> memories = [
        ...     {"type": "preference", "text": "Prefers Python"},
        ...     {"type": "fact", "text": "Works as a developer"}
        ... ]
        >>> print(format_memories_for_prompt(memories))
        ## User Memories
        - [Preference] Prefers Python
        - [Fact] Works as a developer
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    if not memories:
        return ""
    
    lines: List[str] = ["## User Memories"]
    for mem in memories:
        memory_type = mem.get("type", "general").capitalize()
        text = mem.get("text", "")
        lines.append(f"- [{memory_type}] {text}")
    
    return "\n".join(lines)


__all__ = [
    "get_memory_store",
    "store_user_memory",
    "search_user_memories",
    "get_user_memories",
    "delete_user_memory",
    "format_memories_for_prompt",
    "MemoryType",
]
