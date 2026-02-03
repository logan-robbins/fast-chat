"""
OpenAI-compatible Chat Completions API router.

Implements POST /v1/chat/completions per OpenAI API specification:
    - Accepts standard OpenAI request format with messages array
    - Returns standard OpenAI response format with choices and usage
    - Supports streaming via Server-Sent Events (SSE)
    - Manages thread/history persistence (SQLite)
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

Last Grunted: 02/03/2026 10:30:00 AM UTC
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from typing import List, Optional, Union, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field
import httpx
import os
import json
import time
import uuid as uuid_module

from src.db.engine import get_session
from src.db.models import Thread, Message

router = APIRouter()

# Backend configuration
CHAT_APP_URL = os.getenv("CHAT_APP_URL", "http://localhost:8001/v1/chat/completions")


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
        stop: Sequences where the API stops generating. Default None.
        max_tokens: Maximum tokens in completion. Default None (model default).
        max_completion_tokens: Alias for max_tokens (newer API). Default None.
        presence_penalty: Penalty for new topics (-2 to 2). Default 0.
        frequency_penalty: Penalty for repetition (-2 to 2). Default 0.
        logit_bias: Token bias map. Default None.
        user: Unique end-user identifier. Default None.
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    stream: Optional[bool] = False
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
# Token Counting Utilities
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using character-based heuristic.
    
    OpenAI models use ~4 characters per token on average for English text.
    This is a rough estimate - for production, use tiktoken library.
    
    Formula: tokens ≈ characters / 4
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        int: Estimated token count (minimum 1)
        
    Note:
        For accurate counts, integrate tiktoken:
        `import tiktoken; enc = tiktoken.encoding_for_model("gpt-4o")`
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    if not text:
        return 0
    # Average ~4 characters per token for English
    # Add overhead for message structure (~4 tokens per message)
    return max(1, len(text) // 4)


def count_message_tokens(messages: List[ChatMessage]) -> int:
    """
    Count tokens in a list of chat messages.
    
    Includes overhead for message structure per OpenAI's token counting:
    - ~4 tokens overhead per message for role/structure
    - Content tokens based on character estimation
    
    Args:
        messages: List of ChatMessage objects
        
    Returns:
        int: Total estimated prompt tokens
        
    Reference:
        https://platform.openai.com/docs/guides/text-generation/managing-tokens
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    total = 0
    for msg in messages:
        # Message overhead (~4 tokens)
        total += 4
        if msg.content:
            total += estimate_tokens(msg.content)
        if msg.name:
            total += estimate_tokens(msg.name)
    # Add 2 tokens for priming
    total += 2
    return total


# ============================================================================
# OpenAI Error Response Helpers
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
    
    OpenAI Error Format:
    {
        "error": {
            "message": "Description of the error",
            "type": "error_type",
            "param": "parameter_name",
            "code": "error_code"
        }
    }
    
    Args:
        message: Human-readable error description
        error_type: Error category (invalid_request_error, authentication_error, etc.)
        param: The parameter that caused the error (if applicable)
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
    2. Manages thread/history persistence in SQLite
    3. Proxies to the backend chat-app agent
    4. Returns OpenAI-compliant response format
    
    Args:
        request: ChatCompletionRequest with model and messages
        session: Database session (injected)
        
    Returns:
        ChatCompletionResponse (non-streaming) or StreamingResponse (streaming)
        
    Raises:
        HTTPException: 400 for invalid requests, 404 for missing thread
        
    Example Request:
        POST /v1/chat/completions
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream": false
        }
        
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    # Validate model
    valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    if request.model not in valid_models:
        return create_error_response(
            message=f"The model '{request.model}' does not exist or you do not have access to it.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404
        )
    
    # Validate messages
    if not request.messages:
        return create_error_response(
            message="'messages' is a required property",
            error_type="invalid_request_error",
            param="messages",
            code="missing_required_parameter",
            status_code=400
        )
    
    # Generate completion ID
    completion_id = f"chatcmpl-{uuid_module.uuid4().hex[:24]}"
    created_timestamp = int(time.time())
    
    # Thread Management (BFF feature)
    if request.thread_id:
        thread = await session.get(Thread, request.thread_id)
        if not thread:
            return create_error_response(
                message=f"Thread '{request.thread_id}' not found",
                error_type="invalid_request_error",
                param="thread_id",
                code="thread_not_found",
                status_code=404
            )
    else:
        # Create new thread for this conversation
        first_user_msg = next(
            (m.content for m in request.messages if m.role == "user" and m.content),
            "New Chat"
        )
        thread = Thread(
            user_id=uuid_module.uuid4(),
            title=first_user_msg[:50] if first_user_msg else "New Chat"
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
    
    # Calculate prompt tokens
    prompt_tokens = count_message_tokens(request.messages)
    
    # Prepare backend payload
    backend_payload = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "stream": request.stream,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens or request.max_completion_tokens,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        # BFF extensions for backend
        "thread_id": str(thread.id),
        "user_id": str(thread.user_id)
    }
    
    if request.stream:
        return await _handle_streaming_response(
            backend_payload, completion_id, created_timestamp, 
            request.model, prompt_tokens, thread, session
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
    session: AsyncSession
) -> StreamingResponse:
    """
    Handle streaming chat completion response.
    
    Implements Server-Sent Events (SSE) streaming per OpenAI specification.
    Each chunk follows the format: `data: {json}\n\n`
    Stream ends with: `data: [DONE]\n\n`
    
    Args:
        backend_payload: Request to forward to backend
        completion_id: Unique completion identifier
        created_timestamp: Unix timestamp of request
        model: Model name
        prompt_tokens: Calculated prompt token count
        thread: Database thread for persistence
        session: Database session
        
    Returns:
        StreamingResponse with text/event-stream media type
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    async def stream_generator():
        client = httpx.AsyncClient(timeout=120.0)
        full_content = []
        completion_tokens = 0
        
        try:
            async with client.stream("POST", CHAT_APP_URL, json=backend_payload) as response:
                if response.status_code != 200:
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
                    if line.startswith("data: "):
                        data = line[6:]
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
                            yield "data: [DONE]\n\n"
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            # Extract content from backend response
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    full_content.append(content)
                                    completion_tokens += estimate_tokens(content)
                                    
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
                        except json.JSONDecodeError:
                            pass
                            
        except Exception as e:
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
            await client.aclose()
            
            # Persist assistant response
            if full_content:
                from src.db.engine import get_session as get_new_session
                # Note: In production, use a proper async session factory
                # For now, we skip persistence in streaming to avoid session issues
                pass
    
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
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    backend_payload["stream"] = False
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(CHAT_APP_URL, json=backend_payload)
            
            if response.status_code != 200:
                return create_error_response(
                    message=f"Backend error: {response.status_code}",
                    error_type="api_error",
                    code="backend_error",
                    status_code=response.status_code
                )
            
            backend_response = response.json()
            
            # Extract content from backend response
            content = ""
            if "choices" in backend_response and backend_response["choices"]:
                message = backend_response["choices"][0].get("message", {})
                content = message.get("content", "")
            
            # Calculate completion tokens
            completion_tokens = estimate_tokens(content)
            
            # Persist assistant response
            if content:
                assistant_msg = Message(
                    thread_id=thread.id,
                    role="assistant",
                    content=content
                )
                session.add(assistant_msg)
                await session.commit()
            
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
            return create_error_response(
                message="Request to backend timed out",
                error_type="timeout_error",
                code="timeout",
                status_code=504
            )
        except Exception as e:
            return create_error_response(
                message=str(e),
                error_type="api_error",
                code="internal_error",
                status_code=500
            )


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


@router.get("/v1/threads", response_model=List[ThreadResponse])
async def list_threads(session: AsyncSession = Depends(get_session)):
    """
    List all chat threads (BFF extension).
    
    Returns all conversation threads ordered by most recent activity.
    This is a BFF-specific endpoint, not part of standard OpenAI API.
    
    Args:
        session: Database session (injected)
        
    Returns:
        List[ThreadResponse]: List of thread objects
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    result = await session.execute(
        select(Thread).order_by(Thread.updated_at.desc())
    )
    return result.scalars().all()


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
