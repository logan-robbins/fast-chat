"""
File processor service using docproc library.

Handles document processing including:
    - Text extraction from various file formats
    - Document summarization via LLM
    - Vector storage for semantic search

Uses the docproc package for document processing operations.

Dependencies:
    - docproc: Document processing library (pip install -e ../docproc)

Last Grunted: 02/03/2026 10:30:00 AM UTC
"""
import logging
import os
import uuid
from typing import List, Dict, Any

from src.db.models import FileMetadata

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Document processing service using docproc library.
    
    Processes uploaded files through extraction, summarization, and
    vector storage pipeline. All processing runs in-process without
    external service calls.
    
    Methods:
        process_files: Process a batch of files for a thread
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """

    async def process_files(
        self, 
        thread_id: uuid.UUID, 
        files: List[FileMetadata]
    ) -> List[Dict[str, Any]]:
        """
        Process files using docproc library directly.
        
        Pipeline:
        1. Read file content from storage path
        2. Extract text using docproc.extract_text_from_file()
        3. Summarize using docproc.summarize_single_file()
        4. Store in vector DB using docproc.store_document_with_chunks()
        
        Args:
            thread_id: Thread UUID for vector collection isolation.
                       Creates collection named "thread_{thread_id}".
            files: List of FileMetadata objects with storage_path set.
        
        Returns:
            List[Dict[str, Any]]: Processing results per file:
                - file_id: str - UUID of processed file
                - filename: str - Original filename
                - status: str - "success" or "error"
                - document_id: str - Vector store document ID (on success)
                - chunks_stored: int - Number of chunks stored (on success)
                - error: str - Error message (on failure)
                
        Raises:
            RuntimeError: If docproc library is not installed
            
        Last Grunted: 02/03/2026 10:30:00 AM UTC
        """
        # Import here to allow graceful degradation if docproc not installed
        try:
            from docproc import (
                extract_text_from_file,
                summarize_single_file,
                store_document_with_chunks,
                ensure_vector_store_initialized,
            )
        except ImportError:
            logger.error("docproc library not installed. Run: pip install -e ../docproc")
            raise RuntimeError("docproc library required for file processing")
        
        # Initialize vector store
        await ensure_vector_store_initialized()
        
        results = []
        collection_name = f"thread_{thread_id}"
        
        for file_meta in files:
            try:
                logger.info(f"Processing file: {file_meta.filename}")
                
                # Read file content
                with open(file_meta.storage_path, "rb") as f:
                    content = f.read()
                
                # Create a mock file object for extract_text_from_file
                class MockFile:
                    def __init__(self, name: str, data: bytes):
                        self.filename = name
                        self._data = data
                    
                    async def read(self):
                        return self._data
                
                mock_file = MockFile(file_meta.filename, content)
                
                # Extract text
                filename, text, error, visual_info = await extract_text_from_file(
                    mock_file,
                    visual_analysis=False,
                    model="gpt-4o-mini",
                )
                
                if error:
                    logger.error(f"Extraction failed for {filename}: {error}")
                    results.append({
                        "file_id": str(file_meta.id),
                        "filename": filename,
                        "status": "error",
                        "error": error,
                    })
                    continue
                
                # Summarize
                summary_result = await summarize_single_file(
                    filename, text, visual_info, "gpt-4o-mini"
                )
                
                if summary_result["status"] != "success":
                    results.append({
                        "file_id": str(file_meta.id),
                        "filename": filename,
                        "status": "error",
                        "error": summary_result.get("error", "Summarization failed"),
                    })
                    continue
                
                # Store with thread-scoped collection
                doc_id, file_hash, chunks_stored, summary_stored = await store_document_with_chunks(
                    filename=filename,
                    file_content=content,
                    full_text=text,
                    summary=summary_result["summary"],
                    visual_analysis=summary_result.get("visual_analysis", ""),
                    model="gpt-4o-mini",
                    collection_name=collection_name,
                )
                
                logger.info(f"Processed {filename}: {chunks_stored} chunks stored")
                
                results.append({
                    "file_id": str(file_meta.id),
                    "filename": filename,
                    "status": "success",
                    "document_id": doc_id,
                    "chunks_stored": chunks_stored,
                })
                
            except Exception as e:
                logger.exception(f"Error processing {file_meta.filename}: {e}")
                results.append({
                    "file_id": str(file_meta.id),
                    "filename": file_meta.filename,
                    "status": "error",
                    "error": str(e),
                })
        
        return results
