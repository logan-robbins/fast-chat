"""
Document Processing Library (docproc)

A library for document extraction, summarization, and vector storage.
Supports PDF, PPTX, DOCX, XLSX, CSV, and text files.

Usage:
    from docproc import process_file, search_documents, DocumentStore
    
    # Process a file
    result = await process_file(content, filename)
    
    # Search documents
    results = await search_documents("query text", limit=5)

Configuration:
    OPENAI_API_KEY: Required for embeddings and summarization
    CHROMA_PERSIST_DIR: Optional path to persist vector store
"""

from docproc.services.file_processor import (
    extract_text_from_file,
    download_and_process_url,
    read_file_from_shared_volume,
)
from docproc.services.summarization import summarize_text_async, summarize_single_file
from docproc.services.document_store import (
    store_document,
    store_document_with_chunks,
    check_document_exists,
    delete_document,
    delete_documents_for_file,
    delete_collection,
    get_all_documents,
    get_document_count,
    clear_all_documents,
    ensure_vector_store_initialized,
)
from docproc.utils.vector_store import (
    VectorStoreAdapter,
    ChromaVectorStore,
    InMemoryVectorStore,
    get_vector_store,
    collection_exists,
    list_collections,
)
from docproc.utils.llm_config import get_llm_model, get_embeddings_model
from docproc.utils.file_helpers import calculate_file_hash, get_content_type
from docproc.config import IngestionConfig, get_ingestion_config

__version__ = "2.1.0"

__all__ = [
    # File processing
    "extract_text_from_file",
    "download_and_process_url",
    "read_file_from_shared_volume",
    # Summarization
    "summarize_text_async",
    "summarize_single_file",
    # Document storage
    "store_document",
    "store_document_with_chunks",
    "check_document_exists",
    "delete_document",
    "delete_documents_for_file",
    "delete_collection",
    "get_all_documents",
    "get_document_count",
    "clear_all_documents",
    "ensure_vector_store_initialized",
    # Vector store
    "VectorStoreAdapter",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    "collection_exists",
    "list_collections",
    # LLM
    "get_llm_model",
    "get_embeddings_model",
    # Utilities
    "calculate_file_hash",
    "get_content_type",
    # Configuration
    "IngestionConfig",
    "get_ingestion_config",
]
