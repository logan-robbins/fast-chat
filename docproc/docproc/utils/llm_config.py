"""
LLM model configuration - Cloud-first design.

This module provides factory functions for creating LLM chat models and
embedding models. Defaults to OpenAI cloud APIs for production reliability.

Configuration (environment variables):
- OPENAI_API_KEY: Required for all operations (cloud mode)
- LLM_MODEL: Chat model to use (default: gpt-4o-mini)
- EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small, 1536 dims)

Local development fallback (optional):
- USE_LOCAL_MODELS: Set to "true" to use Ollama instead of OpenAI
- OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
- OLLAMA_CHAT_MODEL: Local chat model (default: llama3.2:latest)
- OLLAMA_EMBED_MODEL: Local embedding model (default: nomic-embed-text, 768 dims)

WARNING - Embedding Dimension Mismatch:
    Cloud mode uses text-embedding-3-small which produces 1536-dimensional vectors.
    Local mode uses nomic-embed-text which produces 768-dimensional vectors.
    Switching between cloud and local WILL BREAK existing ChromaDB collections
    because HNSW index dimensionality is locked at first insert. You MUST delete
    and recreate all collections when changing embedding modes.

SDK Versions (verified 02/05/2026):
- langchain>=0.1.0
- langchain_openai>=0.0.5
- langchain_ollama>=0.0.1 (optional, for local mode)

Last Grunted: 02/05/2026
"""
import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Cloud-first: Default to OpenAI
USE_LOCAL_MODELS = os.getenv('USE_LOCAL_MODELS', 'false').lower() == 'true'

# Model defaults - using current best price/performance options
DEFAULT_CHAT_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

# Local fallback configuration (only used if USE_LOCAL_MODELS=true)
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'llama3.2:latest')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')  # Dedicated embedding model, not chat


@lru_cache(maxsize=1)
def _get_openai_classes():
    """Lazy import OpenAI LangChain classes."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    return ChatOpenAI, OpenAIEmbeddings


@lru_cache(maxsize=1)
def _get_ollama_classes():
    """Lazy import Ollama LangChain classes (only when needed)."""
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        return ChatOllama, OllamaEmbeddings
    except ImportError:
        raise RuntimeError(
            "langchain_ollama required for local models. "
            "Install with: pip install langchain-ollama"
        )


def get_llm_model(model_name: str = None):
    """
    Get a LangChain chat model. Defaults to OpenAI cloud.
    
    Creates a configured chat model instance for text generation. Uses
    temperature=0 for deterministic outputs suitable for document processing.
    
    Args:
        model_name (str, optional): Model override. If not specified, uses
            DEFAULT_CHAT_MODEL (gpt-4o-mini) for cloud or OLLAMA_CHAT_MODEL
            for local.
    
    Returns:
        BaseChatModel: LangChain chat model instance (ChatOpenAI or ChatOllama)
            with invoke() method for synchronous calls.
    
    Raises:
        RuntimeError: If USE_LOCAL_MODELS=true but langchain_ollama not installed
    
    Examples:
        llm = get_llm_model()  # Uses gpt-4o-mini
        llm = get_llm_model("gpt-4o")  # Uses GPT-4o
        response = llm.invoke("Summarize this text...")
    
    Last Grunted: 02/05/2026
    """
    if USE_LOCAL_MODELS:
        ChatOllama, _ = _get_ollama_classes()
        effective_model = model_name or OLLAMA_CHAT_MODEL
        logger.debug(f"Using local Ollama model: {effective_model}")
        return ChatOllama(
            model=effective_model,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
    
    # Cloud mode (default)
    ChatOpenAI, _ = _get_openai_classes()
    effective_model = model_name or DEFAULT_CHAT_MODEL
    logger.debug(f"Using OpenAI model: {effective_model}")
    return ChatOpenAI(model=effective_model, temperature=0)


def get_embeddings_model():
    """
    Get a LangChain embeddings model. Defaults to OpenAI cloud.
    
    Model details:
    - Cloud: text-embedding-3-small (1536 dims, $0.00002/1K tokens)
      Normalized to unit length. Use cosine similarity for search.
      OpenAI recommends cosine similarity; with normalized vectors,
      cosine similarity equals dot product.
    - Local: nomic-embed-text (768 dims) - dedicated embedding model,
      NOT a chat model repurposed for embeddings.
    
    Returns:
        Embeddings: LangChain embeddings instance with:
            - embed_query(text: str) -> List[float]: Single text embedding
            - embed_documents(texts: List[str]) -> List[List[float]]: Batch embedding
    
    Raises:
        RuntimeError: If USE_LOCAL_MODELS=true but langchain_ollama not installed
    
    Warning:
        Cloud (1536 dims) and local (768 dims) produce INCOMPATIBLE vectors.
        Switching modes requires deleting and recreating all ChromaDB collections.
        HNSW index dimensionality is fixed at first insert and cannot be changed.

    Note:
        For batch embedding, use embed_documents() which is more efficient
        than calling embed_query() in a loop due to API batching.

    Last Grunted: 02/05/2026
    """
    if USE_LOCAL_MODELS:
        _, OllamaEmbeddings = _get_ollama_classes()
        logger.debug(f"Using local Ollama embeddings: {OLLAMA_EMBED_MODEL}")
        return OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
    
    # Cloud mode (default)
    _, OpenAIEmbeddings = _get_openai_classes()
    logger.debug(f"Using OpenAI embeddings: {DEFAULT_EMBEDDING_MODEL}")
    return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)


# Log configuration on module load
if USE_LOCAL_MODELS:
    logger.info(f"LLM Config: Local mode (Ollama at {OLLAMA_BASE_URL})")
else:
    logger.info(f"LLM Config: Cloud mode (OpenAI {DEFAULT_CHAT_MODEL})")
