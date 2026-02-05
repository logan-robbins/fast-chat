"""
SQLModel database models for Chat API.

Defines the database schema for:
    - Thread: Conversation threads/sessions
    - Message: Individual messages within threads
    - FileMetadata: Uploaded file metadata and storage info
    - Response: OpenAI Responses API response storage
    - ResponseInputItem: Input items associated with responses

All models use UUID primary keys and UTC timestamps.

Last Grunted: 02/04/2026 04:50:00 PM UTC
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Any
from sqlmodel import Field, SQLModel, Relationship, Column
from sqlalchemy import JSON, Integer, String, Boolean, DateTime, func, select
from sqlalchemy.ext.asyncio import AsyncSession


class ThreadMetadataMixin:
    """Mixin for thread metadata fields.
    
    Provides computed metadata properties that are calculated
    on-demand rather than stored in the database.
    """
    
    async def get_message_count(self, session: AsyncSession) -> int:
        """Get the number of messages in this thread."""
        result = await session.execute(
            select(func.count(Message.id)).where(Message.thread_id == self.id)
        )
        return result.scalar() or 0
    
    async def get_last_message_preview(self, session: AsyncSession, max_length: int = 100) -> Optional[str]:
        """Get a preview of the last message in this thread."""
        result = await session.execute(
            select(Message)
            .where(Message.thread_id == self.id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        message = result.scalar_one_or_none()
        if message and message.content:
            preview = message.content[:max_length]
            if len(message.content) > max_length:
                preview += "..."
            return preview
        return None
    
    async def get_total_tokens(self, session: AsyncSession) -> int:
        """Get total token count for this thread."""
        from sqlalchemy import func
        # Estimate ~4 characters per token
        result = await session.execute(
            select(func.sum(func.length(Message.content) / 4))
            .where(Message.thread_id == self.id)
        )
        total = result.scalar()
        return int(total) if total else 0


class Thread(SQLModel, table=True):
    """
    Conversation thread model.
    
    Represents a single conversation session containing multiple messages.
    Used for maintaining conversation history and context.
    Links to LangGraph checkpoint for conversation state.
    
    Attributes:
        id: Unique thread identifier (UUID)
        user_id: User who owns this thread (UUID)
        title: Optional thread title (typically first message excerpt)
        checkpoint_id: LangGraph checkpoint ID for thread state
        checkpoint_ns: LangGraph checkpoint namespace
        created_at: UTC timestamp of creation
        updated_at: UTC timestamp of last update
        messages: Related Message objects (relationship)
        
        # Metadata fields
        is_pinned: Whether thread is pinned
        title_generated_at: When title was auto-generated
        title_generation_model: Model used for title generation
        
    Table: thread
    
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True)
    title: Optional[str] = Field(default=None)
    # Link to LangGraph checkpoint for conversation state
    checkpoint_id: Optional[str] = Field(default=None, index=True)
    checkpoint_ns: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Metadata fields
    is_pinned: bool = Field(default=False)
    title_generated_at: Optional[datetime] = Field(default=None)
    title_generation_model: Optional[str] = Field(default=None)

    messages: List["Message"] = Relationship(back_populates="thread")


class Message(SQLModel, table=True):
    """
    Chat message model.
    
    Represents a single message in a conversation thread.
    Stores role (user/assistant/system) and content.
    
    Attributes:
        id: Unique message identifier (UUID)
        thread_id: Parent thread ID (foreign key)
        role: Message author role ('user', 'assistant', 'system')
        content: Message text content
        created_at: UTC timestamp of creation
        thread: Parent Thread object (relationship)
        
    Table: message
    
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: uuid.UUID = Field(foreign_key="thread.id", index=True)
    role: str  # 'user', 'assistant', 'system'
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    thread: Optional[Thread] = Relationship(back_populates="messages")


class FileMetadata(SQLModel, table=True):
    """
    File metadata model.
    
    Stores metadata about uploaded files including storage location
    and processing status. Supports both OpenAI-style files and
    thread-associated files.
    
    Attributes:
        id: Unique file identifier (UUID)
        thread_id: Optional associated thread ID (for thread-scoped files)
        filename: Original filename
        content_type: MIME type or purpose (e.g., 'assistants', 'fine-tune')
        size_bytes: File size in bytes
        storage_path: Absolute path to stored file
        status: Processing status ('pending', 'uploaded', 'processed', 'error')
        vector_collection: Optional vector store collection name
        created_at: UTC timestamp of creation
        
    Table: filemetadata
    
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    thread_id: Optional[uuid.UUID] = Field(default=None, foreign_key="thread.id", index=True)
    filename: str
    content_type: str
    size_bytes: int
    storage_path: str
    status: str = Field(default="pending")  # pending, uploaded, processed, error
    vector_collection: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Response(SQLModel, table=True):
    """
    OpenAI Responses API response storage model.
    
    Stores model responses for retrieval, multi-turn conversations,
    and conversation state management per OpenAI Responses API spec.
    
    Attributes:
        id: Unique response identifier (resp_... format string)
        created_at: Unix timestamp (seconds) of creation
        completed_at: Unix timestamp (seconds) of completion (nullable)
        status: Response status (completed, failed, in_progress, cancelled, queued, incomplete)
        model: Model ID used (e.g., "gpt-4o", "o3")
        instructions: System/developer message (nullable)
        output: JSON array of output items (messages, tool calls, etc.)
        usage: JSON object with token usage statistics
        response_metadata: JSON object with user-defined key-value pairs (max 16)
            Note: Named 'response_metadata' to avoid SQLAlchemy reserved name conflict
        previous_response_id: ID of previous response for multi-turn (nullable)
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        max_output_tokens: Token limit for response (nullable)
        parallel_tool_calls: Whether parallel tool calls allowed
        tool_choice: Tool selection strategy
        tools: JSON array of available tools
        truncation: Truncation strategy (disabled, auto)
        background: Whether running in background mode
        error: JSON error object if failed (nullable)
        incomplete_details: JSON details if incomplete (nullable)
        
    Table: response
    
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    id: str = Field(primary_key=True)  # resp_... format
    created_at: int = Field(index=True)  # Unix timestamp seconds
    completed_at: Optional[int] = Field(default=None)
    status: str = Field(default="in_progress", index=True)  # completed, failed, in_progress, cancelled, queued, incomplete
    model: str
    instructions: Optional[str] = Field(default=None)
    output: Optional[List[Any]] = Field(default=None, sa_column=Column(JSON))
    usage: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    response_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    previous_response_id: Optional[str] = Field(default=None, index=True)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    max_output_tokens: Optional[int] = Field(default=None)
    parallel_tool_calls: bool = Field(default=True)
    tool_choice: Optional[str] = Field(default="auto")
    tools: Optional[List[dict]] = Field(default=None, sa_column=Column(JSON))
    truncation: str = Field(default="disabled")
    background: bool = Field(default=False)
    error: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    incomplete_details: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    
    input_items: List["ResponseInputItem"] = Relationship(back_populates="response")


class ResponseInputItem(SQLModel, table=True):
    """
    Input item associated with a Response.
    
    Stores individual input items (messages, files, images) that were
    provided to generate a response. Used for input_items endpoint
    and conversation state reconstruction.
    
    Attributes:
        id: Unique input item identifier (msg_... format string)
        response_id: Parent response ID (foreign key)
        type: Item type (message, file, image, function_call_output, etc.)
        role: Message role if type is message (user, assistant, system)
        content: JSON array of content objects
        created_at: Unix timestamp (seconds) of creation
        
    Table: responseinputitem
    
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    __tablename__ = "responseinputitem"
    
    id: str = Field(primary_key=True)  # msg_... format
    response_id: str = Field(foreign_key="response.id", index=True)
    type: str  # message, file, image, function_call_output, etc.
    role: Optional[str] = Field(default=None)  # user, assistant, system (for messages)
    content: Optional[List[Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: int = Field(index=True)  # Unix timestamp seconds
    
    response: Optional[Response] = Relationship(back_populates="input_items")

