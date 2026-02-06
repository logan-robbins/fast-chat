"""
Document storage service for ChromaDB vector database.

Stores document summaries and chunks with embeddings for RAG retrieval.
Uses two collections per document set:
- {collection}: Document chunks for retrieval (semantic search)
- {collection}_summary: Document summaries (metadata lookup)

Chunking Strategy:
- Uses RecursiveCharacterTextSplitter from LangChain
- Default chunk_size: 2000 chars (~500 tokens)
- Default chunk_overlap: 400 chars (20% overlap for context preservation)
- Separators: ["\n\n", "\n", ". ", " ", ""] (paragraph > sentence > word)

Hashing Strategy:
- All document IDs use calculate_file_hash(content, filename, model)
- Produces a 64-char hex SHA256 hash incorporating content + filename + model
- Same file processed with different models gets separate entries
- Deduplication check uses the same hash for consistency across
  check_document_exists(), store_document(), and store_document_with_chunks()

SDK Versions (verified 02/05/2026):
- langchain-text-splitters>=0.0.1: RecursiveCharacterTextSplitter
- chromadb>=0.4.22: Vector storage backend

Last Grunted: 02/05/2026
"""
import asyncio
import logging
import os
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

from docproc.utils.file_helpers import calculate_file_hash, get_content_type
from docproc.utils.llm_config import get_embeddings_model
from docproc.utils.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_COLLECTION = os.getenv("DEFAULT_VECTOR_COLLECTION", "document_summaries")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))

# Module state
_embeddings = None
_store_cache: dict[str, object] = {}
_initialized: set = set()


def _get_embeddings():
    """Lazy-load embeddings model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings_model()
    return _embeddings


def _get_store(collection: str):
    """Get or create cached vector store for collection."""
    if collection not in _store_cache:
        _store_cache[collection] = get_vector_store(collection)
    return _store_cache[collection]


async def _ensure_initialized(collection: str):
    """Ensure collection is initialized."""
    if collection not in _initialized:
        store = _get_store(collection)
        await store.initialize()
        _initialized.add(collection)


async def ensure_vector_store_initialized():
    """Initialize default vector store."""
    await _ensure_initialized(DEFAULT_COLLECTION)
    await _ensure_initialized(f"{DEFAULT_COLLECTION}_summary")


async def check_document_exists(
    file_content: bytes,
    filename: str,
    model: str,
    collection_name: str = "",
) -> dict | None:
    """
    Check if document already exists in vector store by content+filename+model hash.

    Uses calculate_file_hash() for consistency with store_document() and
    store_document_with_chunks(). The hash incorporates content, filename,
    and model so the same file re-processed with a different model is treated
    as a new document.

    Args:
        file_content: Raw file bytes
        filename: Original filename (contributes to hash)
        model: Model used for processing (contributes to hash)
        collection_name: Target collection (default: from env)

    Returns:
        Document info dict if exists, None otherwise
    """
    try:
        file_hash = calculate_file_hash(file_content, filename, model)
        collection = collection_name or DEFAULT_COLLECTION
        summary_collection = f"{collection}_summary"

        await _ensure_initialized(summary_collection)
        store = _get_store(summary_collection)
        doc = await store.get_document(file_hash)

        if doc:
            metadata = doc.get('metadata', {})
            return {
                "filename": filename,
                "status": "exists",
                "summary": doc.get('text', ''),
                "visual_analysis": metadata.get('visual_analysis', ''),
                "processed_at": metadata.get('processed_at'),
                "file_hash": file_hash,
                "error": None,
            }

        return None

    except Exception as e:
        logger.warning(f"Error checking document: {e}")
        return None


async def store_document(
    filename: str,
    file_content: bytes,
    summary: str,
    visual_analysis: str = "",
    model: str = "gpt-4o-mini",
    collection_name: str = "",
) -> str:
    """
    Store document summary in vector database.

    Returns:
        File hash (document ID) - 64-char hex SHA256
    """
    file_hash = calculate_file_hash(file_content, filename, model)
    collection = collection_name or DEFAULT_COLLECTION
    summary_collection = f"{collection}_summary"

    await _ensure_initialized(summary_collection)
    store = _get_store(summary_collection)
    embeddings = _get_embeddings()

    # Generate embedding (run in thread to avoid blocking event loop)
    embedding = await asyncio.to_thread(embeddings.embed_query, summary)

    metadata = {
        "filename": filename,
        "file_size": len(file_content),
        "processed_at": datetime.now().isoformat(),
        "model_used": model,
        "visual_analysis": visual_analysis,
        "content_type": get_content_type(filename),
    }

    # Check for existing and update or add
    existing = await store.get_document(file_hash)
    if existing:
        await store.update_document(
            doc_id=file_hash,
            embedding=embedding,
            text=summary,
            metadata=metadata,
        )
        logger.info(f"Updated document: {filename}")
    else:
        await store.add_document(
            doc_id=file_hash,
            embedding=embedding,
            text=summary,
            metadata=metadata,
        )
        logger.info(f"Stored document: {filename}")

    return file_hash


async def store_document_with_chunks(
    filename: str,
    file_content: bytes,
    full_text: str,
    summary: str,
    visual_analysis: str,
    model: str,
    collection_name: str = "",
) -> tuple:
    """
    Store document summary and text chunks for RAG retrieval.

    Splits document into overlapping chunks using RecursiveCharacterTextSplitter,
    generates embeddings for each chunk, and stores them in ChromaDB for
    semantic search. Also stores the summary separately for quick lookups.

    Args:
        filename: Original filename (e.g., "report.pdf")
        file_content: Raw file bytes (used for hash calculation)
        full_text: Complete extracted text to chunk and embed
        summary: Pre-generated summary text
        visual_analysis: Visual analysis (now inline in text, kept for API compat)
        model: Model name used for processing (stored in metadata, contributes to hash)
        collection_name: Target collection prefix (default: from env)

    Returns:
        tuple: (document_id, file_hash, chunks_stored, summary_stored)
            - document_id (str): 64-char hex SHA256 hash (content + filename + model)
            - file_hash (str): Same as document_id
            - chunks_stored (int): Number of chunks successfully stored
            - summary_stored (bool): Whether summary was stored successfully

    Hashing:
        Uses calculate_file_hash(content, filename, model) for consistent
        document IDs across check_document_exists(), store_document(), and
        this function. The same file processed with a different model gets
        a different hash and is stored as a separate document.

    Chunking Algorithm:
        Uses RecursiveCharacterTextSplitter with separators ["\n\n", "\n", ". ", " ", ""]
        Chunk IDs follow pattern: {document_id}_chunk_{index}

    Last Grunted: 02/05/2026
    """
    # Generate stable IDs using consistent hash
    file_hash = calculate_file_hash(file_content, filename, model)
    document_id = file_hash

    collection = collection_name or DEFAULT_COLLECTION
    summary_collection = f"{collection}_summary"

    # Initialize stores
    chunk_store = _get_store(collection)
    summary_store = _get_store(summary_collection)
    await _ensure_initialized(collection)
    await _ensure_initialized(summary_collection)

    embeddings = _get_embeddings()
    processed_at = datetime.now().isoformat()
    content_type = get_content_type(filename)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    logger.info(f"Split {filename} into {len(chunks)} chunks")

    # Store summary
    summary_stored = False
    try:
        summary_embedding = await asyncio.to_thread(
            embeddings.embed_query, summary or ""
        )
        await summary_store.add_document(
            doc_id=document_id,
            embedding=summary_embedding,
            text=summary,
            metadata={
                "filename": filename,
                "document_id": document_id,
                "chunk_type": "summary",
                "chunk_count": len(chunks),
                "file_hash": file_hash,
                "file_size": len(file_content),
                "content_type": content_type,
                "model_used": model,
                "visual_analysis": visual_analysis,
                "processed_at": processed_at,
            },
        )
        summary_stored = True
    except Exception as e:
        logger.error(f"Failed to store summary: {e}")

    # Store chunks
    chunks_stored = 0
    if chunks:
        try:
            # Batch embed all chunks (run in thread to avoid blocking event loop)
            chunk_embeddings = await asyncio.to_thread(
                embeddings.embed_documents, chunks
            )

            # Prepare batch data
            batch_data = []
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                batch_data.append({
                    "doc_id": f"{document_id}_chunk_{idx}",
                    "embedding": embedding,
                    "text": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "document_id": document_id,
                        "chunk_index": idx,
                        "chunk_count": len(chunks),
                        "chunk_type": "content",
                        "file_hash": file_hash,
                        "content_type": content_type,
                        "model_used": model,
                        "processed_at": processed_at,
                    },
                })

            # Batch insert
            chunks_stored = await chunk_store.add_documents_batch(batch_data)
            logger.info(f"Stored {chunks_stored} chunks for {filename}")

        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")

    return document_id, file_hash, chunks_stored, summary_stored


async def delete_document(
    filename: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> bool:
    """Delete document and chunks by filename."""
    try:
        chunk_collection = collection_name or DEFAULT_COLLECTION
        summary_collection = f"{chunk_collection}_summary"
        deleted = False

        for collection in (chunk_collection, summary_collection):
            store = _get_store(collection)
            await _ensure_initialized(collection)

            docs = await store.get_all_documents()
            doc_ids = [
                d['id'] for d in docs
                if d.get('metadata', {}).get('filename') == filename
            ]

            for doc_id in doc_ids:
                if await store.delete_document(doc_id):
                    deleted = True
                    logger.info(f"Deleted {doc_id} from {collection}")

        return deleted

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False


async def delete_collection(collection_name: str) -> tuple:
    """Delete entire collection and its summary collection."""
    try:
        total = 0
        success = True

        for collection in (collection_name, f"{collection_name}_summary"):
            store = _get_store(collection)
            try:
                await _ensure_initialized(collection)
                count = await store.count()
                total += count
            except Exception:
                count = 0

            if not await store.drop_collection():
                success = False
            else:
                _store_cache.pop(collection, None)
                _initialized.discard(collection)

        return success, total

    except Exception as e:
        logger.error(f"Collection delete failed: {e}")
        return False, 0


async def get_all_documents() -> list[dict]:
    """Get all documents from default collections."""
    docs: list[dict] = []
    for suffix in ("", "_summary"):
        collection = f"{DEFAULT_COLLECTION}{suffix}"
        store = _get_store(collection)
        docs.extend(await store.get_all_documents())
    return docs


async def get_document_count() -> int:
    """Get total document count."""
    total = 0
    for suffix in ("", "_summary"):
        collection = f"{DEFAULT_COLLECTION}{suffix}"
        store = _get_store(collection)
        total += await store.count()
    return total


async def clear_all_documents() -> bool:
    """Clear all documents from default collections."""
    success = True
    for suffix in ("", "_summary"):
        collection = f"{DEFAULT_COLLECTION}{suffix}"
        store = _get_store(collection)
        if not await store.clear_all():
            success = False
    return success
