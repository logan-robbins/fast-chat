"""
OpenAI-compatible Chat Completions API router.

Implements POST /v1/chat/completions per OpenAI API specification:
    - Accepts standard OpenAI request format with messages array
    - Returns standard OpenAI response format with choices and usage
    - Supports streaming via Server-Sent Events (SSE)
    - Manages thread/history persistence (PostgreSQL)
    - Proxies to backend chat-app agent

OpenAI Chat Completions Request Schema:
{
    "model": "gpt-4o",                    # Required
    "messages": [                          # Required
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ],
    "temperature": 1.0,                    # Optional (0-2)
    "max_tokens": null,                    # Optional
    "stream": false,                       # Optional
    "n": 1,                                # Optional
    ...
}

OpenAI Chat Completions Response Schema:
{
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "..."},
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

Reference: https://platform.openai.com/docs/api-reference/chat

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import json
import structlog
import time
import uuid as uuid_module
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.db.engine import get_session
from src.db.models import Thread, Message
from src.services.title_generator import generate_title_with_fallback
from src.services.http_client import get_client, CHAT_APP_URL
from src.services.model_registry import is_model_supported, get_context_limit
from src.services.tokens import count_tokens, count_chat_message_tokens
from src.services.errors import (
    create_error_response,
    model_not_found_error,
    missing_parameter_error,
    resource_not_found_error,
    backend_error,
    timeout_error,
    internal_error,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# ============================================================================
# Thread Management Request/Response Models
# ============================================================================

class CreateThreadRequest(BaseModel):
    """Request to create a new thread."""
    user_id: Optional[UUID] = None
    title: Optional[str] = None


class UpdateThreadRequest(BaseModel):
    """Request to update thread metadata."""
    title: Optional[str] = None
    is_pinned: Optional[bool] = None


class EditMessageRequest(BaseModel):
    """Request to edit a message."""
    new_content: str


class ForkThreadRequest(BaseModel):
    """Request to fork a thread."""
    checkpoint_id: Optional[str] = None  # Fork from specific checkpoint


class ThreadMetadataResponse(BaseModel):
    """Extended thread response with metadata."""
    id: UUID
    user_id: UUID
    title: Optional[str]
    is_pinned: bool
    message_count: int
    last_message_preview: Optional[str]
    created_at: datetime
    updated_at: datetime


class ForkThreadResponse(BaseModel):
    """Response after forking a thread."""
    new_thread_id: UUID
    forked_from: UUID
    message_count: int


# ============================================================================
# OpenAI-Compliant Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    """
    OpenAI Chat Message object.
    
    Represents a single message in the conversation, with role indicating
    the message author (system, user, assistant, or tool).
    
    Attributes:
        role: The role of the message author (system, user, assistant, tool)
        content: The content of the message (text or null for tool calls)
        name: Optional name for the participant
        tool_calls: Optional list of tool calls (for assistant messages)
        tool_call_id: Optional tool call ID (for tool response messages)
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class StreamOptions(BaseModel):
    """
    OpenAI streaming options for controlling streaming behavior.
    
    Attributes:
        include_usage: If true, includes token usage in the final chunk.
            When enabled, the final streaming chunk will contain a `usage`
            field with prompt_tokens, completion_tokens, and total_tokens.
        include_status: If true, forwards status events from backend agent.
            Status events provide user-friendly progress updates like
            "Searching the web...", "Looking through your documents...".
            Default True for rich UX, set False for minimal streaming.
            
    Reference: https://platform.openai.com/docs/api-reference/chat/create#stream_options
    
    Last Grunted: 02/04/2026 08:00:00 PM UTC
    """
    include_usage: bool = False
    include_status: bool = True


# ============================================================================
# Status Event Models (ChatGPT/Claude-style streaming status updates)
# ============================================================================

class StatusEventType(str, Enum):
    """
    Status event types for user-friendly streaming updates.
    
    Follows ChatGPT/Claude 2025/2026 patterns for UI status indicators:
    - thinking: Supervisor is reasoning about routing or synthesizing
    - tool_start: Tool execution beginning (with context about what tool)
    - tool_progress: Progress updates during tool execution
    - tool_complete: Tool execution finished successfully
    - agent_handoff: Supervisor delegating to a specialized agent
    - error: Error occurred during execution (user-friendly message)
    
    Reference: OpenAI Responses API patterns (web_search_call, file_search_call, etc.)
    
    Last Grunted: 02/04/2026 08:00:00 PM UTC
    """
    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_COMPLETE = "tool_complete"
    AGENT_HANDOFF = "agent_handoff"
    ERROR = "error"


class StatusEvent(BaseModel):
    """
    User-friendly status event for streaming UI updates.
    
    Sent as `event: status` SSE events to provide progress indicators
    following ChatGPT/Claude patterns.
    
    Attributes:
        type: Status event type (thinking, tool_start, etc.)
        message: User-friendly message to display (required).
            Examples: "Thinking...", "Searching the web for Python 3.12...",
            "Looking through your documents...", "Running Python code..."
        agent: Agent name if applicable (websearch, knowledge_base, code_interpreter)
        tool: Tool name if applicable (perplexity_search, search_document_content)
        details: Optional structured data for rich UI rendering (progress counts, etc.)
        timestamp: ISO timestamp of the event
        
    Example SSE:
        event: status
        data: {"type": "tool_start", "message": "Searching the web...", "agent": "websearch"}
        
    Last Grunted: 02/04/2026 08:00:00 PM UTC
    """
    type: StatusEventType
    message: str
    agent: Optional[str] = None
    tool: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """
    OpenAI Chat Completions API request body.
    
    Full specification per https://platform.openai.com/docs/api-reference/chat/create
    
    Attributes:
        model: Model ID (e.g., "gpt-4o", "gpt-4o-mini"). Required.
        messages: List of messages in the conversation. Required.
        temperature: Sampling temperature (0-2). Default 1.0.
        top_p: Nucleus sampling parameter. Default 1.0.
        n: Number of completions to generate. Default 1.
        stream: Whether to stream responses via SSE. Default False.
        stream_options: Options for streaming (include_usage, etc.). Default None.
        stop: Sequences where the API stops generating. Default None.
        max_tokens: Maximum tokens in completion. Default None (model default).
        max_completion_tokens: Alias for max_tokens (newer API). Default None.
        presence_penalty: Penalty for new topics (-2 to 2). Default 0.
        frequency_penalty: Penalty for repetition (-2 to 2). Default 0.
        logit_bias: Token bias map. Default None.
        user: Unique end-user identifier. Default None.
        
    Last Grunted: 02/04/2026 06:30:00 PM UTC
    """
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # BFF-specific extensions (non-standard but useful)
    thread_id: Optional[UUID] = None


class UsageInfo(BaseModel):
    """
    OpenAI Token usage statistics.
    
    Attributes:
        prompt_tokens: Tokens in the input prompt
        completion_tokens: Tokens in the generated completion
        total_tokens: Sum of prompt + completion tokens
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """
    OpenAI Chat completion choice object.
    
    Attributes:
        index: Choice index (for n > 1)
        message: The generated message
        finish_reason: Why generation stopped (stop, length, tool_calls, etc.)
        logprobs: Log probabilities (if requested)
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """
    OpenAI Chat Completions API response body.
    
    Attributes:
        id: Unique identifier for this completion (chatcmpl-...)
        object: Always "chat.completion"
        created: Unix timestamp of creation
        model: Model used for completion
        choices: List of completion choices
        usage: Token usage statistics
        system_fingerprint: Backend configuration identifier
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkDelta(BaseModel):
    """
    Delta object for streaming chunks.
    
    Attributes:
        role: Role (only in first chunk)
        content: Content piece (incremental)
        tool_calls: Tool calls (if any)
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    """
    Streaming chunk choice object.
    
    Attributes:
        index: Choice index
        delta: Incremental content
        finish_reason: Set when complete
        logprobs: Log probabilities (if requested)
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionChunk(BaseModel):
    """
    OpenAI streaming chunk object.
    
    Sent as SSE data events during streaming.
    
    Attributes:
        id: Completion ID (same across all chunks)
        object: Always "chat.completion.chunk"
        created: Unix timestamp
        model: Model name
        choices: Chunk choices
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ============================================================================
# Valid Models Configuration
# ============================================================================

# Model validation is handled by the shared model registry
# (src.services.model_registry.is_model_supported)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Create a chat completion (OpenAI-compatible).
    
    Generates a model response for the given conversation. Supports both
    streaming (SSE) and non-streaming responses.
    
    This BFF implementation:
    1. Validates the request against OpenAI schema
    2. Manages thread/history persistence in PostgreSQL
    3. Proxies to the backend chat-app agent
    4. Returns OpenAI-compliant response format
    5. Forwards status events for rich UI experience
    
    Streaming SSE Protocol:
    =======================
    When stream=true, the response is a Server-Sent Events stream with:
    
    Standard Events (OpenAI-compatible):
    - `data: {...}` - Token streaming chunks with delta content
    - `data: [DONE]` - Stream completion marker
    
    Extended Events (BFF-specific, controlled via stream_options):
    - `event: status` - User-friendly status updates (ChatGPT/Claude style)
        Types: thinking, tool_start, tool_progress, tool_complete, agent_handoff, error
        Example: {"type": "thinking", "message": "Thinking..."}
        Example: {"type": "tool_start", "message": "Searching the web...", "agent": "websearch"}
    - `event: agent_start` - Agent routing notifications
    - `event: progress` - Tool progress updates (legacy, prefer status)
    - `event: usage` - Token usage updates (when stream_options.include_usage=true)
    
    Stream Options:
    - include_usage: Include usage statistics in stream (default: false)
    - include_status: Forward status events from backend (default: true)
    
    Args:
        request: ChatCompletionRequest with model and messages
        session: Database session (injected)
        
    Returns:
        ChatCompletionResponse (non-streaming) or StreamingResponse (streaming)
        
    Raises:
        HTTPException: 400 for invalid requests, 404 for missing thread
        
    Example Request (streaming with status):
        POST /v1/chat/completions
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Search for Python 3.12 features"}],
            "stream": true,
            "stream_options": {"include_status": true, "include_usage": true}
        }
        
    Example SSE Stream:
        event: status
        data: {"type": "thinking", "message": "Analyzing your request..."}
        
        event: status  
        data: {"type": "agent_handoff", "message": "Searching the web...", "agent": "websearch"}
        
        event: status
        data: {"type": "tool_start", "message": "Searching for Python 3.12 features...", "tool": "perplexity_search"}
        
        data: {"id": "chatcmpl-abc", "choices": [{"delta": {"role": "assistant"}}]}
        
        data: {"id": "chatcmpl-abc", "choices": [{"delta": {"content": "Python 3.12"}}]}
        
        event: status
        data: {"type": "tool_complete", "message": "Web search complete"}
        
        event: usage
        data: {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150, "is_final": true}
        
        data: [DONE]
        
    Example Response (non-streaming):
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help?"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        
    Last Grunted: 02/04/2026 08:00:00 PM UTC
    """
    # Validate model via shared registry (supports dated variants)
    if not is_model_supported(request.model):
        return model_not_found_error(request.model)
    
    # Validate messages
    if not request.messages:
        return missing_parameter_error("messages")
    
    # Generate completion ID
    completion_id = f"chatcmpl-{uuid_module.uuid4().hex[:24]}"
    created_timestamp = int(time.time())
    
    # Thread Management (BFF feature)
    if request.thread_id:
        thread = await session.get(Thread, request.thread_id)
        if not thread:
            return resource_not_found_error("thread", str(request.thread_id), "thread_id")
    else:
        # Create new thread for this conversation
        first_user_msg = next(
            (m.content for m in request.messages if m.role == "user" and m.content),
            None
        )
        
        # Generate title for first message
        if first_user_msg and len(request.messages) == 1:
            # First message - generate proper title
            title = await generate_title_with_fallback(first_user_msg)
        else:
            title = first_user_msg[:50] if first_user_msg else "New Chat"
        
        thread = Thread(
            user_id=uuid_module.uuid4(),
            title=title
        )
        session.add(thread)
        await session.commit()
        await session.refresh(thread)
    
    # Save user messages to history
    for msg in request.messages:
        if msg.role == "user" and msg.content:
            db_msg = Message(
                thread_id=thread.id,
                role=msg.role,
                content=msg.content
            )
            session.add(db_msg)
    await session.commit()
    
    # Calculate prompt tokens using tiktoken
    prompt_tokens = count_chat_message_tokens(request.messages, request.model)
    
    # Prepare backend payload
    backend_payload = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": request.stream,
        "stream_options": request.stream_options.model_dump() if request.stream_options else None,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens or request.max_completion_tokens,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        # BFF extensions for backend
        "thread_id": str(thread.id),
        "user_id": str(thread.user_id)
    }
    
    logger.info(
        "chat.completion.start",
        model=request.model,
        thread_id=str(thread.id),
        stream=request.stream,
        prompt_tokens=prompt_tokens
    )
    
    # Extract stream_options for usage and status tracking
    include_usage = (
        request.stream_options.include_usage 
        if request.stream_options else False
    )
    include_status = (
        request.stream_options.include_status 
        if request.stream_options else True  # Default: include status events
    )
    
    if request.stream:
        return await _handle_streaming_response(
            backend_payload, completion_id, created_timestamp, 
            request.model, prompt_tokens, thread, session,
            include_usage=include_usage,
            include_status=include_status
        )
    else:
        return await _handle_non_streaming_response(
            backend_payload, completion_id, created_timestamp,
            request.model, prompt_tokens, thread, session
        )


async def _handle_streaming_response(
    backend_payload: dict,
    completion_id: str,
    created_timestamp: int,
    model: str,
    prompt_tokens: int,
    thread: Thread,
    session: AsyncSession,
    include_usage: bool = False,
    include_status: bool = True
) -> StreamingResponse:
    """
    Handle streaming chat completion response with extended event forwarding.
    
    Implements Server-Sent Events (SSE) streaming per OpenAI specification.
    Also forwards extended events from chat-app for rich client experience.
    
    SSE Protocol (event types emitted):
    =====================================
    
    Standard OpenAI-Compatible Events:
    ----------------------------------
    - data: {OpenAI chunk JSON}  # Token streaming chunks
    - data: [DONE]               # Stream completion marker
    
    Extended BFF Events:
    -------------------
    - event: status        # User-friendly status updates (ChatGPT/Claude style)
      data: {
          "type": "thinking" | "tool_start" | "tool_progress" | "tool_complete" | "agent_handoff" | "error",
          "message": "User-friendly status message",
          "agent": "websearch" | "knowledge_base" | "code_interpreter" | null,
          "tool": "perplexity_search" | "search_document_content" | null,
          "details": {...} | null,
          "timestamp": "ISO8601"
      }
    
    - event: agent_start   # Agent routing notification (legacy, prefer status)
      data: {"node": "websearch"}
    
    - event: progress      # Tool progress (legacy, prefer status)
      data: {"type": "rag_progress", ...}
    
    - event: usage         # Token usage update (when include_usage=True)
      data: {
          "prompt_tokens": 100,
          "completion_tokens": 50,
          "total_tokens": 150,
          "context_window_limit": 128000,
          "context_utilization_pct": 0.1,
          "is_final": true | false
      }
    
    Status Event Examples:
    ---------------------
    Thinking:
        event: status
        data: {"type": "thinking", "message": "Thinking..."}
    
    Web Search:
        event: status
        data: {"type": "tool_start", "message": "Searching the web for Python 3.12...", "agent": "websearch", "tool": "perplexity_search"}
    
    RAG/Documents:
        event: status
        data: {"type": "tool_progress", "message": "Searching collection 2/3: financial_docs", "agent": "knowledge_base", "details": {"current": 2, "total": 3}}
    
    Code Execution:
        event: status
        data: {"type": "agent_handoff", "message": "Running Python code...", "agent": "code_interpreter"}
    
    Completion:
        event: status
        data: {"type": "tool_complete", "message": "Found 5 relevant documents", "details": {"results_count": 5}}
    
    Args:
        backend_payload: Request to forward to backend
        completion_id: Unique completion identifier
        created_timestamp: Unix timestamp of request
        model: Model name
        prompt_tokens: Calculated prompt token count
        thread: Database thread for persistence
        session: Database session
        include_usage: Whether to include usage events (default False)
        include_status: Whether to forward status events (default True)
        
    Returns:
        StreamingResponse with text/event-stream media type
        
    Frontend TypeScript Integration Example:
        ```typescript
        const eventSource = new EventSource('/v1/chat/completions');
        
        eventSource.addEventListener('status', (e) => {
            const status = JSON.parse(e.data);
            switch (status.type) {
                case 'thinking': showSpinner(); break;
                case 'tool_start': showStatus(status.message); break;
                case 'tool_complete': hideStatus(); break;
            }
        });
        
        eventSource.onmessage = (e) => {
            if (e.data === '[DONE]') return;
            const chunk = JSON.parse(e.data);
            appendToken(chunk.choices[0]?.delta?.content);
        };
        ```
        
    Last Grunted: 02/04/2026 08:00:00 PM UTC
    """
    # Capture values for use in closure
    thread_id = thread.id
    _include_usage = include_usage
    _include_status = include_status
    _prompt_tokens = prompt_tokens
    
    async def stream_generator():
        # Use shared HTTP client for connection pooling
        client = await get_client()
        full_content: list[str] = []
        completion_tokens = 0
        current_event_type: Optional[str] = None
        
        # Track tokens for periodic usage updates
        last_usage_emit = 0
        USAGE_EMIT_INTERVAL = 50  # Emit usage every N completion tokens
        
        try:
            async with client.stream("POST", CHAT_APP_URL, json=backend_payload) as response:
                if response.status_code != 200:
                    logger.warning(
                        "chat.streaming.backend_error",
                        status_code=response.status_code,
                        completion_id=completion_id
                    )
                    error_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created_timestamp,
                        model=model,
                        choices=[ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(content=None),
                            finish_reason="error"
                        )]
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Send initial chunk with role
                initial_chunk = ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created_timestamp,
                    model=model,
                    choices=[ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(role="assistant"),
                        finish_reason=None
                    )]
                )
                yield f"data: {initial_chunk.model_dump_json()}\n\n"
                
                async for line in response.aiter_lines():
                    # Parse SSE format
                    if line.startswith("event: "):
                        current_event_type = line[7:]
                    elif line.startswith("data: "):
                        data = line[6:]
                        
                        # Handle extended events (status, agent_start, progress, etc.)
                        if current_event_type and current_event_type != "message":
                            if data == "[DONE]":
                                continue
                            try:
                                event_data = json.loads(data)
                                
                                # Handle status events (ChatGPT/Claude-style)
                                if current_event_type == "status":
                                    if _include_status:
                                        # Validate and forward status event
                                        status_type = event_data.get("type")
                                        status_message = event_data.get("message")
                                        
                                        if status_type and status_message:
                                            logger.debug(
                                                "chat.streaming.status_event",
                                                completion_id=completion_id,
                                                status_type=status_type,
                                                message=status_message[:50]
                                            )
                                            yield f"event: status\ndata: {json.dumps(event_data)}\n\n"
                                        else:
                                            logger.warning(
                                                "chat.streaming.invalid_status_event",
                                                completion_id=completion_id,
                                                event_data=event_data
                                            )
                                else:
                                    # Forward other extended events (agent_start, progress, etc.)
                                    yield f"event: {current_event_type}\ndata: {json.dumps(event_data)}\n\n"
                                    
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    "chat.streaming.json_decode_error",
                                    completion_id=completion_id,
                                    event_type=current_event_type,
                                    error=str(e)
                                )
                            current_event_type = None
                            continue
                        
                        # Handle standard OpenAI streaming
                        if data == "[DONE]":
                            # Send final chunk with finish_reason
                            final_chunk = ChatCompletionChunk(
                                id=completion_id,
                                object="chat.completion.chunk",
                                created=created_timestamp,
                                model=model,
                                choices=[ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(),
                                    finish_reason="stop"
                                )]
                            )
                            yield f"data: {final_chunk.model_dump_json()}\n\n"
                            
                            # If include_usage is enabled, send usage chunk before [DONE]
                            # Per OpenAI spec: final chunk with empty choices and usage
                            if _include_usage:
                                context_limit = get_context_limit(model)
                                total_tokens = _prompt_tokens + completion_tokens
                                
                                usage_data = {
                                    "prompt_tokens": _prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": total_tokens
                                }
                                usage_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_timestamp,
                                    "model": model,
                                    "choices": [],  # Empty choices for usage-only chunk
                                    "usage": usage_data
                                }
                                yield f"data: {json.dumps(usage_chunk)}\n\n"
                                
                                # Also emit as custom event with context window info for visualization
                                extended_usage = {
                                    "prompt_tokens": _prompt_tokens,
                                    "completion_tokens": completion_tokens,
                                    "total_tokens": total_tokens,
                                    "context_window_limit": context_limit,
                                    "context_utilization_pct": round((total_tokens / context_limit) * 100, 1),
                                    "is_final": True
                                }
                                yield f"event: usage\ndata: {json.dumps(extended_usage)}\n\n"
                            
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            # Check if this is a token event
                            if chunk_data.get("type") == "token":
                                content = chunk_data.get("content", "")
                                if content:
                                    full_content.append(content)
                                    completion_tokens += count_tokens(content, model)
                                    
                                    # Forward as OpenAI-compliant chunk
                                    content_chunk = ChatCompletionChunk(
                                        id=completion_id,
                                        object="chat.completion.chunk",
                                        created=created_timestamp,
                                        model=model,
                                        choices=[ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionChunkDelta(content=content),
                                            finish_reason=None
                                        )]
                                    )
                                    yield f"data: {content_chunk.model_dump_json()}\n\n"
                                    
                                    # Emit periodic usage updates for real-time context visualization
                                    if _include_usage and (completion_tokens - last_usage_emit) >= USAGE_EMIT_INTERVAL:
                                        last_usage_emit = completion_tokens
                                        context_limit = get_context_limit(model)
                                        total = _prompt_tokens + completion_tokens
                                        running_usage = {
                                            "prompt_tokens": _prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total,
                                            "context_window_limit": context_limit,
                                            "context_utilization_pct": round((total / context_limit) * 100, 1),
                                            "is_final": False
                                        }
                                        yield f"event: usage\ndata: {json.dumps(running_usage)}\n\n"
                            else:
                                # Other data events - extract content if present
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        full_content.append(content)
                                        completion_tokens += count_tokens(content, model)
                                        
                                        content_chunk = ChatCompletionChunk(
                                            id=completion_id,
                                            object="chat.completion.chunk",
                                            created=created_timestamp,
                                            model=model,
                                            choices=[ChatCompletionChunkChoice(
                                                index=0,
                                                delta=ChatCompletionChunkDelta(content=content),
                                                finish_reason=None
                                            )]
                                        )
                                        yield f"data: {content_chunk.model_dump_json()}\n\n"
                                        
                                        # Emit periodic usage updates
                                        if _include_usage and (completion_tokens - last_usage_emit) >= USAGE_EMIT_INTERVAL:
                                            last_usage_emit = completion_tokens
                                            ctx_limit = get_context_limit(model)
                                            total_toks = _prompt_tokens + completion_tokens
                                            running_usage = {
                                                "prompt_tokens": _prompt_tokens,
                                                "completion_tokens": completion_tokens,
                                                "total_tokens": total_toks,
                                                "context_window_limit": ctx_limit,
                                                "context_utilization_pct": round((total_toks / ctx_limit) * 100, 1),
                                                "is_final": False
                                            }
                                            yield f"event: usage\ndata: {json.dumps(running_usage)}\n\n"
                        except json.JSONDecodeError:
                            pass
                        
                        current_event_type = None
                            
        except Exception as e:
            logger.exception(
                "chat.streaming.error",
                completion_id=completion_id,
                error=str(e)
            )
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "param": None,
                    "code": "streaming_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Persist assistant response using session context manager
            if full_content:
                try:
                    from src.db.engine import get_session_context
                    async with get_session_context() as persist_session:
                        assistant_msg = Message(
                            thread_id=thread_id,
                            role="assistant",
                            content="".join(full_content)
                        )
                        persist_session.add(assistant_msg)
                        # Commit happens automatically on context exit
                    
                    logger.debug(
                        "chat.streaming.persisted",
                        completion_id=completion_id,
                        thread_id=str(thread_id),
                        completion_tokens=completion_tokens
                    )
                except Exception as e:
                    logger.warning(
                        "chat.streaming.persist_failed",
                        completion_id=completion_id,
                        error=str(e)
                    )
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


async def _handle_non_streaming_response(
    backend_payload: dict,
    completion_id: str,
    created_timestamp: int,
    model: str,
    prompt_tokens: int,
    thread: Thread,
    session: AsyncSession
) -> ChatCompletionResponse:
    """
    Handle non-streaming chat completion response.
    
    Makes synchronous request to backend and returns complete response
    with full token usage statistics.
    
    Args:
        backend_payload: Request to forward to backend
        completion_id: Unique completion identifier
        created_timestamp: Unix timestamp of request
        model: Model name
        prompt_tokens: Calculated prompt token count
        thread: Database thread for persistence
        session: Database session
        
    Returns:
        ChatCompletionResponse with complete response and usage
        
    Raises:
        HTTPException: If backend request fails
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    backend_payload["stream"] = False
    
    # Use shared HTTP client for connection pooling
    client = await get_client()
    
    try:
        response = await client.post(CHAT_APP_URL, json=backend_payload)
        
        if response.status_code != 200:
            logger.warning(
                "chat.completion.backend_error",
                completion_id=completion_id,
                status_code=response.status_code
            )
            return backend_error(response.status_code)
        
        backend_response = response.json()
        
        # Extract content from backend response
        content = ""
        if "choices" in backend_response and backend_response["choices"]:
            message = backend_response["choices"][0].get("message", {})
            content = message.get("content", "")
        
        # Calculate completion tokens using tiktoken
        completion_tokens = count_tokens(content, model)
        
        # Persist assistant response
        if content:
            assistant_msg = Message(
                thread_id=thread.id,
                role="assistant",
                content=content
            )
            session.add(assistant_msg)
            await session.commit()
        
        logger.info(
            "chat.completion.success",
            completion_id=completion_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        # Build OpenAI-compliant response
        return ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=created_timestamp,
            model=model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            ),
            system_fingerprint=f"fp_{uuid_module.uuid4().hex[:12]}"
        )
        
    except httpx.TimeoutException:
        logger.warning(
            "chat.completion.timeout",
            completion_id=completion_id
        )
        return timeout_error("backend request")
        
    except Exception as e:
        logger.exception(
            "chat.completion.error",
            completion_id=completion_id,
            error=str(e)
        )
        return internal_error(str(e))


# ============================================================================
# BFF-Specific Endpoints (Thread Management)
# ============================================================================

class ThreadResponse(BaseModel):
    """
    Thread response for BFF thread listing.
    
    Attributes:
        id: Thread UUID
        title: Thread title (from first message)
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    id: UUID
    title: Optional[str]


class MessageResponse(BaseModel):
    """
    Message response for thread history.
    
    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
        created_at: ISO timestamp
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    role: str
    content: str
    created_at: str


@router.get("/v1/threads", response_model=List[ThreadMetadataResponse])
async def list_threads(session: AsyncSession = Depends(get_session)):
    """
    List all chat threads with metadata (BFF extension).
    
    Returns all conversation threads with computed metadata including:
    - message_count: Number of messages in thread
    - last_message_preview: Preview of most recent message
    - is_pinned: Whether thread is pinned
    
    This is a BFF-specific endpoint, not part of standard OpenAI API.
    
    Args:
        session: Database session (injected)
        
    Returns:
        List[ThreadMetadataResponse]: List of thread objects with metadata
        
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    # Get threads with message counts and last message previews in a single query
    result = await session.execute(
        select(Thread).order_by(Thread.updated_at.desc())
    )
    threads = result.scalars().all()
    
    # Build response with computed metadata
    response_threads = []
    for thread in threads:
        # Get message count
        count_result = await session.execute(
            sa_select(func.count(Message.id)).where(Message.thread_id == thread.id)
        )
        message_count = count_result.scalar() or 0
        
        # Get last message preview
        last_msg_result = await session.execute(
            sa_select(Message)
            .where(Message.thread_id == thread.id)
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        last_message = last_msg_result.scalar_one_or_none()
        last_message_preview = None
        if last_message and last_message.content:
            preview = last_message.content[:100]
            if len(last_message.content) > 100:
                preview += "..."
            last_message_preview = preview
        
        response_threads.append(ThreadMetadataResponse(
            id=thread.id,
            user_id=thread.user_id,
            title=thread.title,
            is_pinned=thread.is_pinned,
            message_count=message_count,
            last_message_preview=last_message_preview,
            created_at=thread.created_at,
            updated_at=thread.updated_at
        ))
    
    return response_threads


@router.post("/v1/threads", response_model=ThreadMetadataResponse)
async def create_thread(
    request: CreateThreadRequest,
    session: AsyncSession = Depends(get_session)
):
    """Create a new conversation thread.
    
    Args:
        request: Thread creation request
        session: Database session
        
    Returns:
        ThreadMetadataResponse: Created thread with metadata
        
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    thread = Thread(
        user_id=request.user_id or uuid_module.uuid4(),
        title=request.title or "New Chat"
    )
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    
    return ThreadMetadataResponse(
        id=thread.id,
        user_id=thread.user_id,
        title=thread.title,
        is_pinned=thread.is_pinned,
        message_count=0,
        last_message_preview=None,
        created_at=thread.created_at,
        updated_at=thread.updated_at
    )


@router.patch("/v1/threads/{thread_id}", response_model=ThreadMetadataResponse)
async def update_thread(
    thread_id: UUID,
    request: UpdateThreadRequest,
    session: AsyncSession = Depends(get_session)
):
    """Update thread metadata.
    
    Args:
        thread_id: Thread UUID
        request: Update request with fields to change
        session: Database session
        
    Returns:
        ThreadMetadataResponse: Updated thread with metadata
        
    Raises:
        HTTPException: 404 if thread not found
        
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    thread = await session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    if request.title is not None:
        thread.title = request.title
    if request.is_pinned is not None:
        thread.is_pinned = request.is_pinned
    
    thread.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(thread)
    
    # Get current metadata
    count_result = await session.execute(
        sa_select(func.count(Message.id)).where(Message.thread_id == thread.id)
    )
    message_count = count_result.scalar() or 0
    
    last_msg_result = await session.execute(
        sa_select(Message)
        .where(Message.thread_id == thread.id)
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    last_message = last_msg_result.scalar_one_or_none()
    last_message_preview = None
    if last_message and last_message.content:
        preview = last_message.content[:100]
        if len(last_message.content) > 100:
            preview += "..."
        last_message_preview = preview
    
    return ThreadMetadataResponse(
        id=thread.id,
        user_id=thread.user_id,
        title=thread.title,
        is_pinned=thread.is_pinned,
        message_count=message_count,
        last_message_preview=last_message_preview,
        created_at=thread.created_at,
        updated_at=thread.updated_at
    )


@router.delete("/v1/threads/{thread_id}")
async def delete_thread(
    thread_id: UUID,
    session: AsyncSession = Depends(get_session)
):
    """Delete a thread and all its messages.
    
    Args:
        thread_id: Thread UUID
        session: Database session
        
    Returns:
        dict: Deletion confirmation
        
    Raises:
        HTTPException: 404 if thread not found
        
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    thread = await session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Delete all messages first
    await session.execute(
        delete(Message).where(Message.thread_id == thread_id)
    )
    
    # Delete the thread
    await session.delete(thread)
    await session.commit()
    
    return {"deleted": True, "thread_id": str(thread_id)}


@router.post("/v1/threads/{thread_id}/messages/{message_id}/edit")
async def edit_message(
    thread_id: UUID,
    message_id: UUID,
    request: EditMessageRequest,
    session: AsyncSession = Depends(get_session)
):
    """Edit a message and truncate subsequent conversation.
    
    Only user messages can be edited. Editing truncates all subsequent
    messages in the conversation.
    
    Args:
        thread_id: Thread UUID
        message_id: Message UUID
        request: Edit request with new content
        session: Database session
        
    Returns:
        dict: Edit confirmation
        
    Raises:
        HTTPException: 404 if thread/message not found, 400 if not user message
        
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    thread = await session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    message = await session.get(Message, message_id)
    if not message or message.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Message not found")
    
    if message.role != "user":
        raise HTTPException(status_code=400, detail="Can only edit user messages")
    
    # Update the message
    message.content = request.new_content
    message.created_at = datetime.now(timezone.utc)  # Reset timestamp
    
    # Delete all subsequent messages (truncate conversation)
    await session.execute(
        delete(Message).where(
            Message.thread_id == thread_id,
            Message.created_at > message.created_at
        )
    )
    
    await session.commit()
    return {"edited": True, "truncated": True, "message_id": str(message_id)}


@router.post("/v1/threads/{thread_id}/fork", response_model=ForkThreadResponse)
async def fork_thread(
    thread_id: UUID,
    request: ForkThreadRequest,
    session: AsyncSession = Depends(get_session)
):
    """Create a new thread from an existing checkpoint.
    
    Forks a conversation at a specific point, copying all messages
    up to that point into a new thread.
    
    Args:
        thread_id: Original thread UUID
        request: Fork request with optional checkpoint_id
        session: Database session
        
    Returns:
        ForkThreadResponse: New thread details
        
    Raises:
        HTTPException: 404 if thread not found
        
    Last Grunted: 02/04/2026 04:50:00 PM UTC
    """
    original = await session.get(Thread, thread_id)
    if not original:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Create new thread
    new_thread = Thread(
        user_id=original.user_id,
        title=f"{original.title or 'Chat'} (fork)"
    )
    session.add(new_thread)
    await session.flush()  # Get ID without committing
    
    # Build query for messages to copy
    query = (
        sa_select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.created_at)
    )

    # If checkpoint_id is provided, treat it as a message ID and copy
    # only messages up to (and including) that message's timestamp.
    if request.checkpoint_id:
        try:
            checkpoint_uuid = UUID(request.checkpoint_id)
            checkpoint_msg = await session.get(Message, checkpoint_uuid)
            if checkpoint_msg and checkpoint_msg.thread_id == thread_id:
                query = query.where(Message.created_at <= checkpoint_msg.created_at)
            else:
                logger.warning(
                    "fork.checkpoint_not_found",
                    checkpoint_id=request.checkpoint_id,
                    thread_id=str(thread_id),
                )
        except ValueError:
            logger.warning(
                "fork.invalid_checkpoint_id",
                checkpoint_id=request.checkpoint_id,
            )

    result = await session.execute(query)
    messages = result.scalars().all()
    
    for msg in messages:
        new_msg = Message(
            thread_id=new_thread.id,
            role=msg.role,
            content=msg.content
        )
        session.add(new_msg)
    
    await session.commit()
    
    return ForkThreadResponse(
        new_thread_id=new_thread.id,
        forked_from=thread_id,
        message_count=len(messages)
    )


@router.get("/v1/threads/{thread_id}/messages", response_model=List[MessageResponse])
async def get_thread_messages(
    thread_id: UUID,
    session: AsyncSession = Depends(get_session)
):
    """
    Get messages for a specific thread (BFF extension).
    
    Returns the conversation history for a thread in chronological order.
    This is a BFF-specific endpoint, not part of standard OpenAI API.
    
    Args:
        thread_id: UUID of the thread
        session: Database session (injected)
        
    Returns:
        List[MessageResponse]: List of messages in the thread
        
    Raises:
        HTTPException: 404 if thread not found
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    thread = await session.get(Thread, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    result = await session.execute(
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    
    return [
        MessageResponse(
            role=m.role,
            content=m.content,
            created_at=m.created_at.isoformat()
        )
        for m in messages
    ]
