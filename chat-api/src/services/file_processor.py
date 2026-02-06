"""
File processor service using docproc library.

Handles document processing including:
    - Text extraction from various file formats
    - Document summarization via LLM
    - Vector storage for semantic search

Uses the docproc package for document processing operations.
Uses aiofiles for async file I/O operations.

Dependencies:
    - docproc: Document processing library (pip install -e ../docproc)
    - aiofiles: Async file operations (pip install aiofiles)

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import structlog
import os
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import aiofiles

from src.db.models import FileMetadata

logger = structlog.get_logger(__name__)

# Default model for document processing
DEFAULT_PROCESSING_MODEL: str = "gpt-4o-mini"


@dataclass
class ProcessingResult:
    """
    Result of processing a single file.
    
    Attributes:
        file_id: UUID of the processed file
        filename: Original filename
        status: "success" or "error"
        document_id: Vector store document ID (success only)
        chunks_stored: Number of chunks stored (success only)
        error: Error message (error only)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    file_id: str
    filename: str
    status: str
    document_id: Optional[str] = None
    chunks_stored: Optional[int] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "file_id": self.file_id,
            "filename": self.filename,
            "status": self.status,
        }
        if self.status == "success":
            result["document_id"] = self.document_id
            result["chunks_stored"] = self.chunks_stored
        else:
            result["error"] = self.error
        return result


class AsyncFileAdapter:
    """
    Async file adapter compatible with docproc's file interface.
    
    Provides an async read() method that returns the pre-loaded content.
    
    Attributes:
        filename: Original filename
        _data: File content bytes
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    
    def __init__(self, filename: str, data: bytes) -> None:
        """
        Initialize the file adapter.
        
        Args:
            filename: Original filename
            data: File content as bytes
        """
        self.filename = filename
        self._data = data
    
    async def read(self) -> bytes:
        """
        Read file content asynchronously.
        
        Returns:
            bytes: File content
        """
        return self._data


class FileProcessor:
    """
    Document processing service using docproc library.
    
    Processes uploaded files through extraction, summarization, and
    vector storage pipeline. All processing runs in-process without
    external service calls.
    
    Uses async file I/O for non-blocking file reads.
    
    Attributes:
        model: LLM model for processing (default: gpt-4o-mini)
        visual_analysis: Whether to perform visual analysis on images
        
    Methods:
        process_files: Process a batch of files for a thread
        process_single_file: Process a single file
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    
    def __init__(
        self,
        model: str = DEFAULT_PROCESSING_MODEL,
        visual_analysis: bool = False
    ) -> None:
        """
        Initialize the file processor.
        
        Args:
            model: LLM model to use for extraction/summarization
            visual_analysis: Enable visual analysis for images
        """
        self.model = model
        self.visual_analysis = visual_analysis
        self._docproc_available: Optional[bool] = None
    
    def _check_docproc(self) -> Tuple[Any, ...]:
        """
        Check docproc availability and return required functions.
        
        Returns:
            Tuple of docproc functions
            
        Raises:
            RuntimeError: If docproc is not installed
            
        Last Grunted: 02/04/2026 05:30:00 PM UTC
        """
        try:
            from docproc import (
                extract_text_from_file,
                summarize_single_file,
                store_document_with_chunks,
                ensure_vector_store_initialized,
            )
            self._docproc_available = True
            return (
                extract_text_from_file,
                summarize_single_file,
                store_document_with_chunks,
                ensure_vector_store_initialized,
            )
        except ImportError as e:
            self._docproc_available = False
            logger.error(
                "file_processor.docproc_missing",
                error="docproc library not installed. Run: pip install -e ../docproc"
            )
            raise RuntimeError("docproc library required for file processing") from e

    async def _read_file_async(self, file_path: str) -> bytes:
        """
        Read file content asynchronously.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bytes: File content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
            
        Last Grunted: 02/04/2026 05:30:00 PM UTC
        """
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def process_single_file(
        self,
        file_meta: FileMetadata,
        collection_name: str,
        extract_text: Any,
        summarize: Any,
        store_chunks: Any,
    ) -> ProcessingResult:
        """
        Process a single file through the extraction pipeline.
        
        Args:
            file_meta: File metadata object
            collection_name: Vector store collection name
            extract_text: docproc extract function
            summarize: docproc summarize function
            store_chunks: docproc storage function
            
        Returns:
            ProcessingResult: Processing result
            
        Last Grunted: 02/04/2026 05:30:00 PM UTC
        """
        try:
            logger.info(
                "file_processor.processing",
                filename=file_meta.filename,
                file_id=str(file_meta.id)
            )
            
            # Read file content asynchronously
            content = await self._read_file_async(file_meta.storage_path)
            
            # Create async file adapter for docproc
            file_adapter = AsyncFileAdapter(file_meta.filename, content)
            
            # Extract text
            filename, text, error, visual_info = await extract_text(
                file_adapter,
                visual_analysis=self.visual_analysis,
                model=self.model,
            )
            
            if error:
                logger.warning(
                    "file_processor.extraction_failed",
                    filename=filename,
                    error=error
                )
                return ProcessingResult(
                    file_id=str(file_meta.id),
                    filename=filename,
                    status="error",
                    error=error,
                )
            
            # Summarize
            summary_result = await summarize(
                filename, text, visual_info, self.model
            )
            
            if summary_result.get("status") != "success":
                error_msg = summary_result.get("error", "Summarization failed")
                logger.warning(
                    "file_processor.summarization_failed",
                    filename=filename,
                    error=error_msg
                )
                return ProcessingResult(
                    file_id=str(file_meta.id),
                    filename=filename,
                    status="error",
                    error=error_msg,
                )
            
            # Store with thread-scoped collection
            doc_id, file_hash, chunks_stored, summary_stored = await store_chunks(
                filename=filename,
                file_content=content,
                full_text=text,
                summary=summary_result["summary"],
                visual_analysis=summary_result.get("visual_analysis", ""),
                model=self.model,
                collection_name=collection_name,
            )
            
            logger.info(
                "file_processor.success",
                filename=filename,
                document_id=doc_id,
                chunks_stored=chunks_stored
            )
            
            return ProcessingResult(
                file_id=str(file_meta.id),
                filename=filename,
                status="success",
                document_id=doc_id,
                chunks_stored=chunks_stored,
            )
            
        except FileNotFoundError:
            logger.error(
                "file_processor.file_not_found",
                filename=file_meta.filename,
                path=file_meta.storage_path
            )
            return ProcessingResult(
                file_id=str(file_meta.id),
                filename=file_meta.filename,
                status="error",
                error="File not found on disk",
            )
            
        except Exception as e:
            logger.exception(
                "file_processor.unexpected_error",
                filename=file_meta.filename,
                error=str(e)
            )
            return ProcessingResult(
                file_id=str(file_meta.id),
                filename=file_meta.filename,
                status="error",
                error=str(e),
            )

    async def process_files(
        self, 
        thread_id: uuid.UUID, 
        files: List[FileMetadata]
    ) -> List[Dict[str, Any]]:
        """
        Process files using docproc library directly.
        
        Pipeline:
        1. Read file content from storage path (async)
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
            
        Last Grunted: 02/04/2026 05:30:00 PM UTC
        """
        # Get docproc functions (raises if not available)
        (
            extract_text,
            summarize,
            store_chunks,
            ensure_initialized,
        ) = self._check_docproc()
        
        # Initialize vector store
        await ensure_initialized()
        
        # Create thread-scoped collection name
        collection_name = f"thread_{thread_id}"
        
        # Process each file
        results: List[ProcessingResult] = []
        
        for file_meta in files:
            result = await self.process_single_file(
                file_meta=file_meta,
                collection_name=collection_name,
                extract_text=extract_text,
                summarize=summarize,
                store_chunks=store_chunks,
            )
            results.append(result)
        
        return [r.to_dict() for r in results]
