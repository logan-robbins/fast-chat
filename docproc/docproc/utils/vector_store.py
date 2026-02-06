"""
Vector Store - ChromaDB Implementation

Simple in-memory vector storage using ChromaDB. No external server needed.
Can optionally persist to disk like SQLite.

ChromaDB uses a two-index system:
- Metadata Index: SQLite database storing document metadata
- Vector Index: HNSW (Hierarchical Navigable Small World) for similarity search

Distance metric: Cosine distance (configured via "hnsw:space": "cosine")
- distance = 1 - cosine_similarity
- distance of 0 means identical vectors
- similarity_score = 1 - distance (converted back to similarity)

Retrieval Features:
- Metadata filtering: search_similar() accepts a `where` dict for ChromaDB
  metadata filters ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or).
  This replaces the need for separate collections for different document types.
- Score thresholding: search_similar() accepts a `score_threshold` float to
  filter out low-relevance results (recommended: 0.3-0.7 depending on use case).
- Efficient deletion: delete_by_metadata() uses ChromaDB's where clause for
  bulk deletion without scanning all documents.

Configuration:
- CHROMA_PERSIST_DIR: Directory for persistence (default: in-memory only)
- CHROMA_COLLECTION: Default collection name (default: "documents")

Threading:
    ChromaDB's Python client is fully synchronous. All collection operations
    are wrapped in asyncio.to_thread() so they run in the default thread pool
    executor and do not block the asyncio event loop. This is important for
    large batch inserts or queries that may take significant time.

SDK Version (verified 02/05/2026):
- chromadb>=0.4.22: PersistentClient, Client, get_or_create_collection

Last Grunted: 02/05/2026
"""
import asyncio
import logging
import os
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

# Configuration
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "")  # Empty = in-memory
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# Module-level client (singleton)
_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        if PERSIST_DIR:
            logger.info(f"ChromaDB: Persistent mode ({PERSIST_DIR})")
            _client = chromadb.PersistentClient(path=PERSIST_DIR)
        else:
            logger.info("ChromaDB: In-memory mode")
            _client = chromadb.Client()
    return _client


class ChromaVectorStore:
    """
    Simple vector store using ChromaDB.

    ChromaDB handles embedding storage and similarity search with minimal setup.
    Runs in-memory by default (like SQLite :memory:) or persists to disk.

    Uses HNSW index with cosine distance metric. Embedding dimensionality is
    set by the first embedding added and cannot be changed afterward.

    All ChromaDB operations are offloaded to threads via asyncio.to_thread()
    to avoid blocking the event loop, since ChromaDB's Python client is
    synchronous.

    Args:
        collection_name: Name of the collection (default: "documents").
            Collections are isolated namespaces within the ChromaDB instance.

    Example:
        store = ChromaVectorStore("my_docs")
        await store.initialize()
        await store.add_document(doc_id="doc1", embedding=[0.1]*1536, ...)
        results = await store.search_similar(query_embedding, limit=5)

    Last Grunted: 02/05/2026
    """

    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        self._collection = None

    async def initialize(self) -> None:
        """Initialize or get the collection."""
        client = _get_client()
        self._collection = await asyncio.to_thread(
            client.get_or_create_collection,
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = await asyncio.to_thread(self._collection.count)
        logger.info(f"ChromaDB: Collection '{self.collection_name}' ready ({count} docs)")

    async def add_document(
        self,
        doc_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> bool:
        """Add a single document."""
        try:
            clean_metadata = self._clean_metadata(metadata)
            await asyncio.to_thread(
                self._collection.add,
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[clean_metadata],
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB add failed: {e}")
            return False

    async def add_documents_batch(self, documents: list[dict[str, Any]]) -> int:
        """Batch add multiple documents."""
        if not documents:
            return 0

        try:
            ids = [d["doc_id"] for d in documents]
            embeddings = [d["embedding"] for d in documents]
            texts = [d["text"] for d in documents]
            metadatas = [self._clean_metadata(d["metadata"]) for d in documents]

            await asyncio.to_thread(
                self._collection.add,
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            logger.info(f"ChromaDB: Added {len(documents)} documents")
            return len(documents)
        except Exception as e:
            logger.error(f"ChromaDB batch add failed: {e}")
            return 0

    async def update_document(
        self,
        doc_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> bool:
        """Update an existing document."""
        try:
            clean_metadata = self._clean_metadata(metadata)
            await asyncio.to_thread(
                self._collection.update,
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[clean_metadata],
            )
            return True
        except Exception as e:
            logger.error(f"ChromaDB update failed: {e}")
            return False

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID."""
        try:
            result = await asyncio.to_thread(
                self._collection.get,
                ids=[doc_id],
                include=["documents", "metadatas"],
            )

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
            return None
        except Exception as e:
            logger.error(f"ChromaDB get failed: {e}")
            return None

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 5,
        where: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.

        ChromaDB returns cosine distance (1 - cosine_similarity), so we convert
        back to similarity score. Results are ordered by increasing distance
        (most similar first).

        Args:
            query_embedding: Query vector. Must match the dimensionality of
                stored embeddings (e.g., 1536 for text-embedding-3-small).
            limit: Maximum results to return (default: 5)
            where: Optional ChromaDB metadata filter dict. Supports operators:
                $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, and logical
                operators $and, $or.
                Example: {"filename": "report.pdf"}
                Example: {"chunk_type": {"$eq": "content"}}
                Example: {"$and": [{"filename": "report.pdf"}, {"chunk_index": {"$lt": 5}}]}
            score_threshold: Optional minimum similarity score (0.0-1.0).
                Results below this threshold are filtered out. Recommended
                values: 0.3 for broad recall, 0.5 for balanced, 0.7 for
                high precision. None (default) returns all results.

        Returns:
            List of matching documents, each containing:
                - id: Document identifier
                - text: Document text content
                - metadata: Document metadata dict
                - distance: Cosine distance (0 = identical, 2 = opposite)
                - similarity_score: 1 - distance (1 = identical, -1 = opposite)

        Last Grunted: 02/05/2026
        """
        try:
            query_kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where

            results = await asyncio.to_thread(
                self._collection.query,
                **query_kwargs,
            )

            docs = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity = 1 - distance

                    # Filter by score threshold if specified
                    if score_threshold is not None and similarity < score_threshold:
                        continue

                    docs.append({
                        "id": doc_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": distance,
                        "similarity_score": similarity,
                    })
            return docs
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a single document."""
        try:
            await asyncio.to_thread(self._collection.delete, ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}")
            return False

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete multiple documents."""
        try:
            await asyncio.to_thread(self._collection.delete, ids=doc_ids)
            return True
        except Exception as e:
            logger.error(f"ChromaDB delete failed: {e}")
            return False

    async def delete_by_metadata(self, where: dict[str, Any]) -> bool:
        """
        Delete documents matching a metadata filter.

        Uses ChromaDB's native `where` clause for efficient bulk deletion
        without scanning all documents first. This is significantly faster
        than get_all + filter + delete for large collections.

        Args:
            where: ChromaDB metadata filter dict.
                Example: {"filename": "report.pdf"}
                Example: {"document_id": "abc123"}
                Example: {"$and": [{"filename": "report.pdf"}, {"chunk_type": "content"}]}

        Returns:
            True if deletion succeeded, False on error.

        Last Grunted: 02/05/2026
        """
        try:
            await asyncio.to_thread(self._collection.delete, where=where)
            return True
        except Exception as e:
            logger.error(f"ChromaDB delete_by_metadata failed: {e}")
            return False

    async def get_all_documents(self) -> list[dict[str, Any]]:
        """Get all documents."""
        try:
            result = await asyncio.to_thread(
                self._collection.get,
                include=["documents", "metadatas"],
            )

            docs = []
            for i, doc_id in enumerate(result["ids"]):
                docs.append({
                    "id": doc_id,
                    "text": result["documents"][i] if result["documents"] else "",
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                })
            return docs
        except Exception as e:
            logger.error(f"ChromaDB get_all failed: {e}")
            return []

    async def count(self) -> int:
        """Get document count."""
        try:
            return await asyncio.to_thread(self._collection.count)
        except Exception as e:
            logger.error(f"ChromaDB count failed: {e}")
            return 0

    async def clear_all(self) -> bool:
        """Clear all documents by dropping and recreating collection."""
        try:
            client = _get_client()
            await asyncio.to_thread(client.delete_collection, self.collection_name)
            self._collection = await asyncio.to_thread(
                client.get_or_create_collection,
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB: Cleared collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"ChromaDB clear failed: {e}")
            return False

    async def drop_collection(self) -> bool:
        """Permanently drop the collection."""
        try:
            client = _get_client()
            await asyncio.to_thread(client.delete_collection, self.collection_name)
            self._collection = None
            logger.info(f"ChromaDB: Dropped collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"ChromaDB drop failed: {e}")
            return False

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Clean metadata for ChromaDB storage.

        ChromaDB only supports str, int, float, bool in metadata.
        None values are converted to empty strings; other unsupported
        types are stringified.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean


async def collection_exists(collection_name: str) -> bool:
    """Check if a collection exists in ChromaDB.

    Args:
        collection_name: Name of the collection to check.

    Returns:
        True if the collection exists, False otherwise.

    Last Grunted: 02/05/2026
    """
    try:
        client = _get_client()
        collections = await asyncio.to_thread(client.list_collections)
        return any(c.name == collection_name for c in collections)
    except Exception as e:
        logger.error(f"ChromaDB collection_exists check failed: {e}")
        return False


async def list_collections() -> list[str]:
    """List all collection names in ChromaDB.

    Returns:
        List of collection name strings.

    Last Grunted: 02/05/2026
    """
    try:
        client = _get_client()
        collections = await asyncio.to_thread(client.list_collections)
        return [c.name for c in collections]
    except Exception as e:
        logger.error(f"ChromaDB list_collections failed: {e}")
        return []


def get_vector_store(collection_name: str = DEFAULT_COLLECTION) -> ChromaVectorStore:
    """
    Factory function to create a ChromaVectorStore.

    Creates a new ChromaVectorStore instance. Must call initialize() before use.
    The underlying ChromaDB client is a module-level singleton.

    Args:
        collection_name: Name of the collection (default: from CHROMA_COLLECTION
            env var or "documents")

    Returns:
        Uninitialized ChromaVectorStore. Call ``await store.initialize()``
        before performing any operations.

    Example:
        store = get_vector_store("my_docs")
        await store.initialize()
        await store.add_document(
            doc_id="doc1",
            embedding=[0.1] * 1536,
            text="Document content",
            metadata={"filename": "report.pdf"}
        )

    Last Grunted: 02/05/2026
    """
    return ChromaVectorStore(collection_name=collection_name)
