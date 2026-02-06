"""
OpenAI-compatible Files API router.

Implements the /v1/files endpoints per OpenAI API specification:
    - POST /v1/files - Upload a file
    - GET /v1/files - List all files
    - GET /v1/files/{file_id} - Retrieve file metadata
    - DELETE /v1/files/{file_id} - Delete a file
    - GET /v1/files/{file_id}/content - Retrieve file content

OpenAI File Object Schema:
{
    "id": "file-abc123",
    "object": "file",
    "bytes": 120000,
    "created_at": 1677610602,
    "filename": "mydata.jsonl",
    "purpose": "fine-tune"
}

Reference: https://platform.openai.com/docs/api-reference/files

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import structlog
import os
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.db.engine import get_session
from src.db.models import FileMetadata, Thread
from src.services.errors import (
    create_error_response,
    resource_not_found_error,
    invalid_parameter_error,
    internal_error,
)

logger = structlog.get_logger(__name__)

router = APIRouter()

# ============================================================================
# Configuration
# ============================================================================

UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")
MAX_FILE_SIZE_BYTES: int = 512 * 1024 * 1024  # 512 MB per OpenAI spec

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Regex for validating file ID format (file-{24 hex chars})
FILE_ID_PATTERN = re.compile(r"^file-([a-f0-9]{24})$")


# ============================================================================
# OpenAI-Compliant Pydantic Models
# ============================================================================

class FilePurpose(str, Enum):
    """
    Valid file purposes per OpenAI API.
    
    Attributes:
        assistants: Files used with Assistants API
        assistants_output: Output from Assistants
        batch: Files for Batch API
        batch_output: Output from Batch API
        fine_tune: Training data for fine-tuning
        fine_tune_results: Results from fine-tuning
        vision: Images for vision fine-tuning
        user_data: General purpose files
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    assistants = "assistants"
    assistants_output = "assistants_output"
    batch = "batch"
    batch_output = "batch_output"
    fine_tune = "fine-tune"
    fine_tune_results = "fine-tune-results"
    vision = "vision"
    user_data = "user_data"


class FileObject(BaseModel):
    """
    OpenAI File object representation.
    
    Per https://platform.openai.com/docs/api-reference/files/object
    
    Attributes:
        id: Unique file identifier (file-...)
        object: Always "file"
        bytes: File size in bytes
        created_at: Unix timestamp of creation
        filename: Original filename
        purpose: Intended use (assistants, fine-tune, etc.)
        status: Processing status (uploaded, processed, error) - deprecated
        status_details: Error details if failed - deprecated
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: Optional[str] = None  # Deprecated but included for compatibility
    status_details: Optional[str] = None  # Deprecated


class FileListResponse(BaseModel):
    """
    OpenAI Files list response.
    
    Attributes:
        object: Always "list"
        data: Array of FileObject instances
        has_more: Whether more files exist (pagination)
        first_id: ID of first file in list
        last_id: ID of last file in list
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    object: str = "list"
    data: List[FileObject]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class FileDeleteResponse(BaseModel):
    """
    OpenAI file deletion response.
    
    Attributes:
        id: Deleted file ID
        object: Always "file"
        deleted: Always True on success
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    id: str
    object: str = "file"
    deleted: bool = True


# ============================================================================
# Helper Functions
# ============================================================================

def file_metadata_to_object(meta: FileMetadata) -> FileObject:
    """
    Convert database FileMetadata to OpenAI FileObject.
    
    Args:
        meta: Database FileMetadata model
        
    Returns:
        FileObject: OpenAI-compliant file object
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    valid_purposes = {p.value for p in FilePurpose}
    purpose = meta.content_type if meta.content_type in valid_purposes else "user_data"
    
    return FileObject(
        id=f"file-{meta.id.hex[:24]}",
        object="file",
        bytes=meta.size_bytes,
        created_at=int(meta.created_at.timestamp()),
        filename=meta.filename,
        purpose=purpose,
        status=meta.status,
        status_details=None
    )


def parse_file_id(file_id: str) -> Optional[str]:
    """
    Parse and validate file ID format.
    
    Args:
        file_id: File ID string (format: file-{24 hex chars})
        
    Returns:
        Optional[str]: UUID hex part if valid, None otherwise
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    match = FILE_ID_PATTERN.match(file_id)
    return match.group(1) if match else None


async def find_file_by_id(
    session: AsyncSession,
    uuid_hex: str,
) -> Optional[FileMetadata]:
    """Find a file by its UUID hex prefix using an indexed range query.

    Constructs lower and upper UUID bounds from the 24-char hex prefix
    and performs a primary-key range scan, avoiding a full table scan.

    Args:
        session: Database session.
        uuid_hex: First 24 hex characters of the UUID.

    Returns:
        The matching :class:`FileMetadata` or ``None``.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    if len(uuid_hex) != 24:
        return None

    try:
        lower = uuid.UUID(uuid_hex + "0" * 8)
        upper = uuid.UUID(uuid_hex + "f" * 8)
    except ValueError:
        return None

    result = await session.execute(
        select(FileMetadata).where(
            FileMetadata.id >= lower,
            FileMetadata.id <= upper,
        )
    )
    return result.scalars().first()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem storage.
    
    Removes path separators and other potentially dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Remove path separators
    safe = filename.replace("/", "_").replace("\\", "_")
    # Remove null bytes and other control characters
    safe = "".join(c for c in safe if ord(c) >= 32)
    # Limit length
    return safe[:255] if safe else "unnamed"


async def save_file_async(file_path: str, content: bytes) -> int:
    """
    Save file content asynchronously.
    
    Args:
        file_path: Destination path
        content: File content bytes
        
    Returns:
        int: File size in bytes
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    
    # Get file size
    stat = await aiofiles.os.stat(file_path)
    return stat.st_size


async def delete_file_async(file_path: str) -> bool:
    """
    Delete file asynchronously.
    
    Args:
        file_path: Path to delete
        
    Returns:
        bool: True if deleted, False if not found
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    try:
        await aiofiles.os.remove(file_path)
        return True
    except FileNotFoundError:
        return False


# ============================================================================
# OpenAI-Compatible Endpoints
# ============================================================================

@router.post("/v1/files", response_model=FileObject)
async def upload_file(
    file: UploadFile = File(..., description="The file to upload"),
    purpose: str = Form(..., description="The intended purpose (assistants, fine-tune, etc.)"),
    session: AsyncSession = Depends(get_session)
):
    """
    Upload a file to the API.
    
    Files can be used with features like Assistants, Fine-tuning, and Batch API.
    Individual files can be up to 512 MB.
    
    Args:
        file: The file to upload (multipart/form-data)
        purpose: Intended purpose (assistants, fine-tune, batch, vision, user_data)
        session: Database session (injected)
        
    Returns:
        FileObject: The uploaded file object
        
    Raises:
        HTTPException: 400 for invalid purpose or file issues
        
    Example Request:
        curl https://api.example.com/v1/files \\
            -H "Authorization: Bearer $API_KEY" \\
            -F purpose="assistants" \\
            -F file="@myfile.pdf"
            
    Example Response:
        {
            "id": "file-abc123",
            "object": "file",
            "bytes": 120000,
            "created_at": 1677610602,
            "filename": "myfile.pdf",
            "purpose": "assistants"
        }
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Validate purpose
    valid_purposes = [p.value for p in FilePurpose]
    if purpose not in valid_purposes:
        return invalid_parameter_error(
            param="purpose",
            message=f"Invalid purpose '{purpose}'. Must be one of: {', '.join(valid_purposes)}",
            code="invalid_purpose"
        )
    
    # Validate filename
    if not file.filename:
        return invalid_parameter_error(
            param="file",
            message="File must have a filename",
            code="missing_filename"
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        return invalid_parameter_error(
            param="file",
            message=f"File size ({len(content)} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES} bytes / 512 MB)",
            code="file_too_large"
        )
    
    # Generate file ID and path
    file_id = uuid.uuid4()
    safe_filename = sanitize_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    
    try:
        # Save file asynchronously
        file_size = await save_file_async(file_path, content)
        
        logger.info(
            "files.upload",
            file_id=str(file_id),
            filename=file.filename,
            size_bytes=file_size,
            purpose=purpose
        )
        
        # Create metadata record
        file_meta = FileMetadata(
            id=file_id,
            thread_id=None,  # OpenAI files API doesn't use threads
            filename=file.filename,
            content_type=purpose,  # Store purpose as content_type
            size_bytes=file_size,
            storage_path=file_path,
            status="uploaded"
        )
        session.add(file_meta)
        await session.commit()
        await session.refresh(file_meta)
        
        return file_metadata_to_object(file_meta)
        
    except Exception as e:
        logger.exception(
            "files.upload_failed",
            filename=file.filename,
            error=str(e)
        )
        # Clean up on failure
        await delete_file_async(file_path)
        return internal_error(f"Failed to upload file: {str(e)}")


@router.get("/v1/files", response_model=FileListResponse)
async def list_files(
    purpose: Optional[str] = Query(None, description="Filter by purpose"),
    limit: int = Query(10000, ge=1, le=10000, description="Max files to return"),
    order: str = Query("desc", description="Sort order (asc or desc)"),
    after: Optional[str] = Query(None, description="Cursor for pagination"),
    session: AsyncSession = Depends(get_session)
):
    """
    List all uploaded files.
    
    Returns a list of files that have been uploaded. Supports filtering by
    purpose and pagination.
    
    Args:
        purpose: Only return files with this purpose (optional)
        limit: Maximum number of files to return (1-10000, default 10000)
        order: Sort by created_at (asc or desc, default desc)
        after: Pagination cursor (file ID)
        session: Database session (injected)
        
    Returns:
        FileListResponse: List of file objects
        
    Example Request:
        GET /v1/files?purpose=fine-tune&limit=100
        
    Example Response:
        {
            "object": "list",
            "data": [
                {"id": "file-abc123", "object": "file", ...},
                ...
            ],
            "has_more": false
        }
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Build query
    query = select(FileMetadata)
    
    if purpose:
        query = query.where(FileMetadata.content_type == purpose)
    
    if order == "asc":
        query = query.order_by(FileMetadata.created_at.asc())
    else:
        query = query.order_by(FileMetadata.created_at.desc())
    
    query = query.limit(limit + 1)  # Fetch one extra to check has_more
    
    result = await session.execute(query)
    files = list(result.scalars().all())
    
    has_more = len(files) > limit
    if has_more:
        files = files[:limit]
    
    file_objects = [file_metadata_to_object(f) for f in files]
    
    return FileListResponse(
        object="list",
        data=file_objects,
        has_more=has_more,
        first_id=file_objects[0].id if file_objects else None,
        last_id=file_objects[-1].id if file_objects else None
    )


@router.get("/v1/files/{file_id}", response_model=FileObject)
async def retrieve_file(
    file_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Retrieve file metadata by ID.
    
    Returns information about a specific file.
    
    Args:
        file_id: The file ID (format: file-{uuid})
        session: Database session (injected)
        
    Returns:
        FileObject: The file object
        
    Raises:
        HTTPException: 404 if file not found
        
    Example Request:
        GET /v1/files/file-abc123
        
    Example Response:
        {
            "id": "file-abc123",
            "object": "file",
            "bytes": 120000,
            "created_at": 1677610602,
            "filename": "mydata.jsonl",
            "purpose": "fine-tune"
        }
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Validate and parse file ID
    uuid_hex = parse_file_id(file_id)
    if not uuid_hex:
        return invalid_parameter_error(
            param="file_id",
            message=f"Invalid file ID format: {file_id}. Expected format: file-{{24 hex characters}}",
            code="invalid_file_id"
        )
    
    # Find file
    file_meta = await find_file_by_id(session, uuid_hex)
    
    if not file_meta:
        return resource_not_found_error("file", file_id, "file_id")
    
    return file_metadata_to_object(file_meta)


@router.delete("/v1/files/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Delete a file by ID.
    
    Deletes the file from storage and removes metadata.
    
    Args:
        file_id: The file ID (format: file-{uuid})
        session: Database session (injected)
        
    Returns:
        FileDeleteResponse: Deletion confirmation
        
    Raises:
        HTTPException: 404 if file not found
        
    Example Request:
        DELETE /v1/files/file-abc123
        
    Example Response:
        {
            "id": "file-abc123",
            "object": "file",
            "deleted": true
        }
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Validate and parse file ID
    uuid_hex = parse_file_id(file_id)
    if not uuid_hex:
        return invalid_parameter_error(
            param="file_id",
            message=f"Invalid file ID format: {file_id}",
            code="invalid_file_id"
        )
    
    # Find file
    file_meta = await find_file_by_id(session, uuid_hex)
    
    if not file_meta:
        return resource_not_found_error("file", file_id, "file_id")
    
    # Delete from disk asynchronously
    await delete_file_async(file_meta.storage_path)
    
    # Delete from database
    await session.delete(file_meta)
    await session.commit()
    
    logger.info(
        "files.deleted",
        file_id=file_id,
        filename=file_meta.filename
    )
    
    return FileDeleteResponse(
        id=file_id,
        object="file",
        deleted=True
    )


@router.get("/v1/files/{file_id}/content")
async def retrieve_file_content(
    file_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Retrieve file content by ID.
    
    Returns the raw file content for download.
    
    Args:
        file_id: The file ID (format: file-{uuid})
        session: Database session (injected)
        
    Returns:
        FileResponse: The file content
        
    Raises:
        HTTPException: 404 if file not found
        
    Example Request:
        GET /v1/files/file-abc123/content
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Validate and parse file ID
    uuid_hex = parse_file_id(file_id)
    if not uuid_hex:
        return invalid_parameter_error(
            param="file_id",
            message=f"Invalid file ID format: {file_id}",
            code="invalid_file_id"
        )
    
    # Find file
    file_meta = await find_file_by_id(session, uuid_hex)
    
    if not file_meta:
        return resource_not_found_error("file", file_id, "file_id")
    
    # Check file exists on disk
    if not os.path.exists(file_meta.storage_path):
        logger.error(
            "files.content_missing",
            file_id=file_id,
            path=file_meta.storage_path
        )
        return internal_error("File content not found on disk")
    
    return FileResponse(
        path=file_meta.storage_path,
        filename=file_meta.filename,
        media_type="application/octet-stream"
    )


# ============================================================================
# BFF-Specific Endpoints (Thread-based file uploads)
# ============================================================================

@router.post("/v1/threads/{thread_id}/files")
async def upload_thread_files(
    thread_id: UUID,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session)
):
    """
    Upload files to a specific thread (BFF extension).
    
    Associates uploaded files with a conversation thread for context.
    This is a BFF-specific endpoint, not part of standard OpenAI API.
    
    Args:
        thread_id: UUID of the thread
        files: List of files to upload (multipart/form-data)
        session: Database session (injected)
        
    Returns:
        List[FileObject]: List of uploaded file objects
        
    Raises:
        HTTPException: 404 if thread not found
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Verify thread exists
    thread = await session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail={
            "error": {
                "message": f"Thread '{thread_id}' not found",
                "type": "invalid_request_error",
                "param": "thread_id",
                "code": "thread_not_found"
            }
        })
    
    saved_files: List[FileMetadata] = []
    
    for upload_file in files:
        # Read content
        content = await upload_file.read()
        
        # Validate size
        if len(content) > MAX_FILE_SIZE_BYTES:
            logger.warning(
                "files.thread_upload_too_large",
                thread_id=str(thread_id),
                filename=upload_file.filename,
                size=len(content)
            )
            continue  # Skip this file but continue with others
        
        file_id = uuid.uuid4()
        safe_filename = sanitize_filename(upload_file.filename or "unknown")
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
        
        # Save asynchronously
        file_size = await save_file_async(file_path, content)
        
        # Create metadata
        file_meta = FileMetadata(
            id=file_id,
            thread_id=thread_id,
            filename=upload_file.filename or "unknown",
            content_type=upload_file.content_type or "application/octet-stream",
            size_bytes=file_size,
            storage_path=file_path,
            status="uploaded"
        )
        session.add(file_meta)
        saved_files.append(file_meta)
    
    await session.commit()
    
    logger.info(
        "files.thread_upload",
        thread_id=str(thread_id),
        files_count=len(saved_files)
    )
    
    return [file_metadata_to_object(f) for f in saved_files]
