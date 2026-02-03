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

Last Grunted: 02/03/2026 10:30:00 AM UTC
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel
from enum import Enum
import shutil
import os
import uuid
import time

from src.db.engine import get_session
from src.db.models import FileMetadata, Thread

router = APIRouter()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    id: str
    object: str = "file"
    deleted: bool = True


# ============================================================================
# Helper Functions
# ============================================================================

def create_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create an OpenAI-style error response.
    
    Args:
        message: Human-readable error description
        error_type: Error category
        param: Parameter that caused error
        code: Machine-readable error code
        status_code: HTTP status code
        
    Returns:
        JSONResponse with OpenAI error format
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }
    )


def file_metadata_to_object(meta: FileMetadata) -> FileObject:
    """
    Convert database FileMetadata to OpenAI FileObject.
    
    Args:
        meta: Database FileMetadata model
        
    Returns:
        FileObject: OpenAI-compliant file object
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    return FileObject(
        id=f"file-{meta.id.hex[:24]}",
        object="file",
        bytes=meta.size_bytes,
        created_at=int(meta.created_at.timestamp()),
        filename=meta.filename,
        purpose=meta.content_type if meta.content_type in [p.value for p in FilePurpose] else "user_data",
        status=meta.status,
        status_details=None
    )


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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    # Validate purpose
    valid_purposes = [p.value for p in FilePurpose]
    if purpose not in valid_purposes:
        return create_error_response(
            message=f"Invalid purpose '{purpose}'. Must be one of: {', '.join(valid_purposes)}",
            error_type="invalid_request_error",
            param="purpose",
            code="invalid_purpose",
            status_code=400
        )
    
    # Validate file
    if not file.filename:
        return create_error_response(
            message="File must have a filename",
            error_type="invalid_request_error",
            param="file",
            code="missing_filename",
            status_code=400
        )
    
    # Generate file ID and save
    file_id = uuid.uuid4()
    safe_filename = file.filename.replace("/", "_").replace("\\", "_")
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
    
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size = os.path.getsize(file_path)
        
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
        # Clean up on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        return create_error_response(
            message=f"Failed to upload file: {str(e)}",
            error_type="api_error",
            code="upload_failed",
            status_code=500
        )


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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    # Parse file ID
    if not file_id.startswith("file-"):
        return create_error_response(
            message=f"Invalid file ID format: {file_id}",
            error_type="invalid_request_error",
            param="file_id",
            code="invalid_file_id",
            status_code=400
        )
    
    # Extract UUID from file-{uuid} format
    uuid_part = file_id[5:]  # Remove "file-" prefix
    
    # Query by partial UUID match (we only store first 24 chars in ID)
    result = await session.execute(
        select(FileMetadata)
    )
    files = result.scalars().all()
    
    # Find matching file
    for file_meta in files:
        if file_meta.id.hex[:24] == uuid_part or str(file_meta.id).startswith(uuid_part):
            return file_metadata_to_object(file_meta)
    
    return create_error_response(
        message=f"No such file: {file_id}",
        error_type="invalid_request_error",
        param="file_id",
        code="file_not_found",
        status_code=404
    )


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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    # Parse file ID
    if not file_id.startswith("file-"):
        return create_error_response(
            message=f"Invalid file ID format: {file_id}",
            error_type="invalid_request_error",
            param="file_id",
            code="invalid_file_id",
            status_code=400
        )
    
    uuid_part = file_id[5:]
    
    # Find file
    result = await session.execute(select(FileMetadata))
    files = result.scalars().all()
    
    file_meta = None
    for f in files:
        if f.id.hex[:24] == uuid_part or str(f.id).startswith(uuid_part):
            file_meta = f
            break
    
    if not file_meta:
        return create_error_response(
            message=f"No such file: {file_id}",
            error_type="invalid_request_error",
            param="file_id",
            code="file_not_found",
            status_code=404
        )
    
    # Delete from disk
    if os.path.exists(file_meta.storage_path):
        os.remove(file_meta.storage_path)
    
    # Delete from database
    await session.delete(file_meta)
    await session.commit()
    
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    # Parse file ID
    if not file_id.startswith("file-"):
        return create_error_response(
            message=f"Invalid file ID format: {file_id}",
            error_type="invalid_request_error",
            param="file_id",
            code="invalid_file_id",
            status_code=400
        )
    
    uuid_part = file_id[5:]
    
    # Find file
    result = await session.execute(select(FileMetadata))
    files = result.scalars().all()
    
    file_meta = None
    for f in files:
        if f.id.hex[:24] == uuid_part or str(f.id).startswith(uuid_part):
            file_meta = f
            break
    
    if not file_meta:
        return create_error_response(
            message=f"No such file: {file_id}",
            error_type="invalid_request_error",
            param="file_id",
            code="file_not_found",
            status_code=404
        )
    
    if not os.path.exists(file_meta.storage_path):
        return create_error_response(
            message=f"File content not found on disk",
            error_type="api_error",
            code="file_content_missing",
            status_code=500
        )
    
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
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
    
    saved_files = []
    
    for file in files:
        file_id = uuid.uuid4()
        safe_filename = (file.filename or "unknown").replace("/", "_").replace("\\", "_")
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{safe_filename}")
        
        # Save to disk
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create metadata
        file_meta = FileMetadata(
            id=file_id,
            thread_id=thread_id,
            filename=file.filename or "unknown",
            content_type=file.content_type or "application/octet-stream",
            size_bytes=os.path.getsize(file_path),
            storage_path=file_path,
            status="uploaded"
        )
        session.add(file_meta)
        saved_files.append(file_meta)
    
    await session.commit()
    
    return [file_metadata_to_object(f) for f in saved_files]
