"""
File utility functions for ingestion service.

Provides hash calculation and content type detection for document files.

Last Grunted: 02/03/2026 02:45:00 PM PST
"""
import hashlib
import os


def calculate_file_hash(file_content: bytes, filename: str, model: str = "gpt-4o-mini") -> str:
    """
    Calculate a unique hash combining file content, filename, and model.
    
    The hash includes the model name to allow re-processing the same file
    with a different model (e.g., upgrading from gpt-4o-mini to gpt-4o)
    and storing both results separately.
    
    Args:
        file_content (bytes): Raw bytes of the file (any size supported)
        filename (str): Original filename including extension (contributes to hash)
        model (str): LLM model name used for processing (default: "gpt-4o-mini")
    
    Returns:
        str: 64-character hexadecimal SHA256 hash string
    
    Raises:
        TypeError: If file_content is not bytes
    
    Algorithm:
        SHA256(file_content || filename.encode('utf-8') || model.encode('utf-8'))
        
        Uses sequential update() calls for memory efficiency with large files.
        The concatenation order matters - changing order produces different hash.
    
    Note:
        For content-only deduplication (ignoring model), use:
        hashlib.sha256(file_content).hexdigest()[:16]
        as done in store_document_with_chunks().
    
    Last Grunted: 02/03/2026 02:45:00 PM PST
    """
    hasher = hashlib.sha256()
    hasher.update(file_content)
    hasher.update(filename.encode('utf-8'))
    hasher.update(model.encode('utf-8'))
    return hasher.hexdigest()


def get_content_type(filename: str) -> str:
    """
    Get MIME content type from filename extension.
    
    Uses os.path.splitext for reliable extension parsing. Extension matching
    is case-insensitive (e.g., .PDF and .pdf both return application/pdf).
    
    Args:
        filename (str): Filename with extension (e.g., "report.pdf", "DOCUMENT.DOCX")
    
    Returns:
        str: MIME type string. Returns 'application/octet-stream' for
            unknown or missing extensions.
    
    Supported types:
        - .txt: text/plain
        - .pdf: application/pdf
        - .pptx: application/vnd.openxmlformats-officedocument.presentationml.presentation
        - .docx: application/vnd.openxmlformats-officedocument.wordprocessingml.document
        - .xlsx: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        - .csv: text/csv
    
    Last Grunted: 02/03/2026 02:45:00 PM PST
    """
    ext = os.path.splitext(filename)[1].lower()
    content_types = {
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.pdf': 'application/pdf', 
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.csv': 'text/csv',
    }
    return content_types.get(ext, 'application/octet-stream')