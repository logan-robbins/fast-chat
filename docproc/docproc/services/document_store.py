"""
Document storage service for ChromaDB vector database.

Stores document summaries and chunks with embeddings for RAG retrieval.
Uses two collections per document set:
- {collection}: Document chunks for retrieval (semantic search)
- {collection}_summary: Document summaries (metadata lookup)

Chunking Strategy:
- Uses RecursiveCharacterTextSplitter.from_tiktoken_encoder() from LangChain
- Token-aware splitting: splits on character boundaries (paragraphs,
  sentences, words) but MEASURES chunk size in tokens via tiktoken
- model_name="gpt-4o": uses the cl200k_base encoding matching OpenAI
  embedding and chat models for accurate token counting
- Default chunk_size: 300 tokens — research-optimized for the 256-512 token
  retrieval sweet spot (see refs below)
- Default chunk_overlap: 50 tokens (~17%) — 2026 research shows overlap
  provides marginal benefit above 10-20%
- Separators: ["\n\n", "\n", ". ", " ", ""] (paragraph > sentence > word)
- add_start_index=True: Tracks each chunk's character offset in the source
  document, enabling citation support and debugging
- Hard constraint: from_tiktoken_encoder recursively splits any chunk
  exceeding the token limit, unlike CharacterTextSplitter.from_tiktoken_encoder
  which only measures but doesn't enforce

Why token-aware over pure semantic chunking:
- SemanticChunker (langchain_experimental) requires embedding API calls
  during chunking, adding cost and latency
- SemanticChunker doesn't support add_start_index for citation tracking
- RecursiveCharacterTextSplitter + tiktoken gives the best balance of
  quality, speed, and cost for RAG retrieval (LangChain recommended default)

Chunking Research References:
- "Document Chunking for RAG: 9 Strategies Tested" (langcopilot.com, 2025):
  256-512 tokens optimal for factoid queries, 1024+ for analytical
- "Chunk Size Optimization" (arxiv.org, 2026): Sentence chunking most
  cost-effective; overlap provides no measurable indexing benefit
- "Chunking Strategies" (weaviate.io): Chunking is the most important
  factor for RAG performance
- LangChain docs (docs.langchain.com, 2026): from_tiktoken_encoder
  provides hard token constraint vs soft measurement

Hashing Strategy:
- All document IDs use calculate_file_hash(content, filename, model)
- Produces a 64-char hex SHA256 hash incorporating content + filename + model
- Same file processed with different models gets separate entries
- Deduplication check uses the same hash for consistency across
  check_document_exists(), store_document(), and store_document_with_chunks()

SDK Versions (verified 02/05/2026):
- langchain-text-splitters>=0.3.0: RecursiveCharacterTextSplitter.from_tiktoken_encoder
- tiktoken>=0.7.0: OpenAI BPE tokenizer (cl200k_base encoding)
- chromadb>=0.4.22: Vector storage backend

Last Grunted: 02/05/2026
"""
import asyncio
import logging
import os
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from docproc.utils.file_helpers import calculate_file_hash, get_content_type
from docproc.utils.llm_config import get_embeddings_model
from docproc.utils.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Configuration -- loaded from centralized IngestionConfig
from docproc.config import get_ingestion_config as _get_config

def _cfg():
    """Lazy config accessor (avoids import-time env read issues in tests)."""
    c = _get_config()
    return c

# Backward-compatible module-level accessors
DEFAULT_COLLECTION = os.getenv("DEFAULT_VECTOR_COLLECTION", "document_summaries")

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

    Splits document into overlapping chunks using token-aware
    RecursiveCharacterTextSplitter (via from_tiktoken_encoder), generates
    embeddings for each chunk, and stores them in ChromaDB for semantic
    search. Also stores the summary separately for quick lookups.

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
        Uses RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4o")
        Token-aware: splits on char boundaries, measures size in tokens
        Separators: ["\n\n", "\n", ". ", " ", ""] (paragraph > sentence > word)
        add_start_index=True for chunk position tracking
        Default: 300 tokens, 50 token overlap (~17%)
        Chunk IDs follow pattern: {document_id}_chunk_{index}
        Each chunk's metadata includes start_index and char_length

    Last Grunted: 02/06/2026
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

    # Split text into chunks with position tracking using token-aware splitting.
    # from_tiktoken_encoder splits on character boundaries (preserving paragraphs,
    # sentences, words) but measures chunk size in tokens via tiktoken. This gives
    # a hard constraint: any chunk exceeding the token limit is recursively split.
    # add_start_index=True embeds each chunk's character offset in the source
    # document as metadata, enabling citation support and chunk tracing.
    cfg = _cfg()
    if cfg.chunk_method == "token":
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
    chunk_docs = splitter.create_documents([full_text])
    chunks = [doc.page_content for doc in chunk_docs]
    chunk_start_indices = [doc.metadata.get("start_index", 0) for doc in chunk_docs]
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

            # Prepare batch data with position metadata for citation support
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
                        "start_index": chunk_start_indices[idx],
                        "char_length": len(chunk_text),
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
    """
    Delete document and chunks by filename.

    Uses ChromaDB's native metadata filter (where clause) for efficient
    bulk deletion. This avoids the O(n) scan of all documents that the
    previous get_all + filter + delete pattern required.

    Args:
        filename: Filename to delete (matches metadata "filename" field)
        collection_name: Target collection prefix (default: from env)

    Returns:
        True if deletion succeeded for at least one collection, False on error.
    """
    try:
        chunk_collection = collection_name or DEFAULT_COLLECTION
        summary_collection = f"{chunk_collection}_summary"
        deleted = False

        for collection in (chunk_collection, summary_collection):
            store = _get_store(collection)
            await _ensure_initialized(collection)

            if await store.delete_by_metadata({"filename": filename}):
                deleted = True
                logger.info(f"Deleted docs with filename={filename} from {collection}")

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


async def delete_documents_for_file(
    filename: str,
    collection_name: str,
) -> bool:
    """Delete all chunks and summary for a specific file from a collection.

    Removes both the chunk documents (from ``collection_name``) and the
    summary document (from ``{collection_name}_summary``) whose metadata
    ``filename`` matches the given value.

    Uses ChromaDB's native metadata filter (where clause) for efficient
    bulk deletion, avoiding an O(n) scan of all documents.

    Args:
        filename: Original filename used during ingestion.
        collection_name: The thread-scoped collection prefix
            (e.g. ``thread_{uuid}``).

    Returns:
        True if deletion succeeded for at least one collection.

    Last Grunted: 02/05/2026
    """
    deleted = False
    summary_collection = f"{collection_name}_summary"

    for coll_name in (collection_name, summary_collection):
        try:
            store = _get_store(coll_name)
            await _ensure_initialized(coll_name)

            if await store.delete_by_metadata({"filename": filename}):
                deleted = True
                logger.info(
                    "Deleted docs for file '%s' from %s",
                    filename,
                    coll_name,
                )
        except Exception as e:
            logger.error(
                "Failed to delete docs for file '%s' from %s: %s",
                filename,
                coll_name,
                e,
            )

    return deleted


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
