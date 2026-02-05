"""OpenAI-compatible REST API adapter for Chat-App with human-in-the-loop support.

Provides a FastAPI server that exposes the LangGraph supervisor as an
OpenAI-compatible chat completions endpoint. Supports human-in-the-loop
workflows with interrupt/resume functionality.

Endpoints:
    POST /v1/chat/completions: Streaming chat completion (OpenAI-compatible)
    POST /v1/chat/resume: Resume interrupted conversation with human input
    GET /v1/chat/interrupt/{thread_id}: Get pending interrupt status
    GET /health: Health check endpoint

Streaming Protocol (ChatGPT/Claude 2025/2026 patterns):
    The API uses Server-Sent Events (SSE) with multiple event types:
    - token: LLM output tokens with agent metadata
    - agent_start: Agent routing notifications  
    - status: User-friendly status updates (ChatGPT/Claude-style)
        - "Thinking..." during reasoning
        - "Searching the web..." during web search
        - "Looking through your documents..." during RAG
        - "Running Python code..." during code execution
    - progress: Legacy tool progress updates (for backward compatibility)
    - complete: Stream completion marker
    - usage: Token usage statistics

Human-in-the-Loop Flow:
    1. Graph encounters interrupt() - returns __interrupt__ in response
    2. Client polls /v1/chat/interrupt/{thread_id} to see pending interrupts
    3. Client POSTs to /v1/chat/resume with human response
    4. Graph continues execution from interrupt point

Last Grunted: 02/04/2026 07:30:00 PM PST
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from chat_app.graphs.app import app as graph

# Setup Logging
logger = logging.getLogger(__name__)

app_api = FastAPI(
    title="Chat App (Agent)",
    version="2.0.0",
    description="LangGraph multi-agent orchestrator with OpenAI-compatible API"
)


# ------------------------------------------------------------------
# OpenAI-Compatible Schema Models
# ------------------------------------------------------------------

class ChatMessage(BaseModel):
    """OpenAI-format chat message."""
    
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional sender name")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is one of the allowed values."""
        allowed_roles = {"user", "assistant", "system"}
        if v.lower() not in allowed_roles:
            raise ValueError(f"role must be one of {allowed_roles}, got '{v}'")
        return v.lower()
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v


class StreamOptions(BaseModel):
    """Streaming options for controlling token usage reporting."""
    
    include_usage: bool = Field(default=False, description="Include token usage in streaming response")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    
    model: str = Field(default="gpt-4o", description="Model to use (ignored, uses supervisor model)")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming response")
    stream_options: Optional[StreamOptions] = Field(default=None, description="Streaming options (include_usage, etc.)")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for conversation continuity")
    user_id: Optional[str] = Field(default=None, description="User ID for memory and personalization")
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        """Validate messages list is not empty."""
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ResumeRequest(BaseModel):
    """Request to resume an interrupted conversation."""
    
    thread_id: str = Field(..., min_length=1, description="Thread ID of the interrupted conversation")
    resume_data: Dict[str, Any] = Field(..., description="Human response to the interrupt")
    stream: bool = Field(default=True, description="Enable streaming response")


class InterruptStatus(BaseModel):
    """Status of pending interrupts for a thread."""
    
    thread_id: str
    has_interrupt: bool
    interrupt_data: Optional[Dict[str, Any]] = None
    interrupt_type: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Single choice in a streaming chunk."""
    
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ------------------------------------------------------------------
# Adapter Logic
# ------------------------------------------------------------------

def _convert_messages(messages: List[ChatMessage]) -> List[BaseMessage]:
    """Convert OpenAI-format messages to LangChain message objects.
    
    Transforms a list of ChatMessage objects (OpenAI format) into the
    corresponding LangChain message types for use with the supervisor graph.
    
    Args:
        messages: List of OpenAI-format chat messages.
        
    Returns:
        List[BaseMessage]: List of LangChain message objects (HumanMessage,
            AIMessage, or SystemMessage).
            
    Note:
        Unknown roles are treated as user messages.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    lc_messages: List[BaseMessage] = []
    
    for msg in messages:
        role = msg.role.lower()
        if role == "user":
            lc_messages.append(HumanMessage(content=msg.content, name=msg.name))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=msg.content, name=msg.name))
        elif role == "system":
            lc_messages.append(SystemMessage(content=msg.content, name=msg.name))
        else:
            # Fallback: treat unknown roles as user messages
            logger.warning(f"Unknown role '{msg.role}', treating as user message")
            lc_messages.append(HumanMessage(content=msg.content, name=msg.name))
    
    return lc_messages


async def _stream_response(
    request: ChatCompletionRequest, 
    run_id: str
) -> AsyncGenerator[str, None]:
    """Stream LangGraph responses as OpenAI-compatible Server-Sent Events.
    
    Uses multiple stream modes simultaneously for rich client experience:
    - messages: Token-by-token streaming with metadata
    - updates: Agent routing decisions (agent_start events)
    - custom: Progress updates from tools (RAG progress, etc.)
    
    Event Types Emitted:
        - token: LLM output tokens with agent metadata
        - agent_start: Agent routing notifications
        - progress: Tool progress updates
        - usage: Token usage updates (when stream_options.include_usage=true)
        - complete: Stream completion marker
    
    Token Usage Tracking:
        When stream_options.include_usage is enabled, this function:
        1. Counts prompt tokens from the input messages
        2. Tracks completion tokens as they stream
        3. Emits periodic usage events during streaming
        4. Includes final usage in the completion event
    
    Args:
        request: The incoming chat completion request.
        run_id: Unique identifier for this run.
        
    Yields:
        str: SSE-formatted strings with event types and data.
        
    Last Grunted: 02/04/2026 06:30:00 PM UTC
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Token usage tracking
    include_usage = request.stream_options.include_usage if request.stream_options else False
    prompt_tokens = 0
    completion_tokens = 0
    
    if include_usage:
        # Count prompt tokens from input messages
        from chat_app.summarization import count_tokens, MESSAGE_TOKEN_OVERHEAD
        for msg in request.messages:
            prompt_tokens += MESSAGE_TOKEN_OVERHEAD
            if msg.content:
                prompt_tokens += count_tokens(msg.content)
    
    # Prepare graph input
    inputs: Dict[str, Any] = {
        "messages": _convert_messages(request.messages),
    }
    
    config: Dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": request.user_id
        }
    }

    logger.info(
        "Starting streaming run",
        extra={
            "run_id": run_id,
            "thread_id": thread_id,
            "message_count": len(request.messages),
            "user_id": request.user_id
        }
    )
    
    # Token counting for usage tracking
    last_usage_emit_tokens = 0
    USAGE_EMIT_INTERVAL = 50  # Emit usage every N completion tokens
    
    try:
        # Use multiple stream modes simultaneously for rich experience
        stream_modes = ["messages", "updates", "custom"]
        
        async for mode, chunk in graph.astream(inputs, config, stream_mode=stream_modes):
            if mode == "messages":
                # chunk is (message_chunk, metadata)
                msg_chunk, metadata = chunk
                if msg_chunk.content:
                    # Track completion tokens
                    if include_usage:
                        from chat_app.summarization import count_tokens as _count_tokens
                        completion_tokens += _count_tokens(msg_chunk.content)
                        
                        # Emit periodic usage updates for real-time visualization
                        if (completion_tokens - last_usage_emit_tokens) >= USAGE_EMIT_INTERVAL:
                            last_usage_emit_tokens = completion_tokens
                            usage_event = {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                                "is_final": False
                            }
                            yield f"event: usage\ndata: {json.dumps(usage_event)}\n\n"
                    
                    event = {
                        "type": "token",
                        "content": msg_chunk.content,
                        "agent": metadata.get("langgraph_node"),
                        "tags": metadata.get("tags", [])
                    }
                    yield f"event: token\ndata: {json.dumps(event)}\n\n"
                    
            elif mode == "updates":
                # chunk is state update dict
                node_name = list(chunk.keys())[0] if chunk else None
                if node_name and node_name != "__end__":
                    event = {
                        "type": "agent_start",
                        "node": node_name
                    }
                    yield f"event: agent_start\ndata: {json.dumps(event)}\n\n"
                elif node_name == "__end__":
                    # Send completion event with final usage if enabled
                    completion_event: Dict[str, Any] = {"type": "complete"}
                    if include_usage:
                        completion_event["usage"] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    yield f"event: complete\ndata: {json.dumps(completion_event)}\n\n"
                    
            elif mode == "custom":
                # chunk is custom data from get_stream_writer()
                # Check if it's a status event (ChatGPT/Claude-style)
                if isinstance(chunk, dict) and "status" in chunk:
                    # Extract the StatusEvent and emit as 'status' event type
                    status_event = chunk["status"]
                    yield f"event: status\ndata: {json.dumps(status_event)}\n\n"
                else:
                    # Legacy progress event (for backward compatibility)
                    event = {
                        "type": "progress",
                        "data": chunk
                    }
                    yield f"event: progress\ndata: {json.dumps(event)}\n\n"
        
        # Send OpenAI-compatible final chunk
        final_chunk = ChatCompletionChunk(
            id=run_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        
        # Emit final usage if enabled
        if include_usage:
            final_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "is_final": True
            }
            yield f"event: usage\ndata: {json.dumps(final_usage)}\n\n"
        
        yield "data: [DONE]\n\n"
        
        logger.info(
            "Streaming run completed",
            extra={"run_id": run_id, "thread_id": thread_id}
        )

    except Exception as e:
        logger.error(
            "Error in stream",
            extra={
                "run_id": run_id,
                "thread_id": thread_id,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        
        # Send error chunk to client
        error_chunk = ChatCompletionChunk(
            id=run_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta={"error": str(e)},
                    finish_reason="error"
                )
            ]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


async def _stream_resume_response(
    resume_request: ResumeRequest,
    run_id: str
) -> AsyncGenerator[str, None]:
    """Stream resumed LangGraph responses after interrupt.
    
    Resumes execution from an interrupt point with human-provided data.
    Uses multiple stream modes for rich client experience.
    
    Args:
        resume_request: The resume request with human response data.
        run_id: Unique identifier for this run.
        
    Yields:
        str: SSE-formatted strings with event types and data.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    config: Dict[str, Any] = {
        "configurable": {
            "thread_id": resume_request.thread_id
        }
    }
    
    # Create Command to resume with human input
    resume_command = Command(resume=resume_request.resume_data)
    
    logger.info(
        "Resuming interrupted conversation",
        extra={
            "run_id": run_id,
            "thread_id": resume_request.thread_id,
            "resume_data_keys": list(resume_request.resume_data.keys())
        }
    )
    
    try:
        # Use multiple stream modes for rich experience
        stream_modes = ["messages", "updates", "custom"]
        
        async for mode, chunk in graph.astream(resume_command, config, stream_mode=stream_modes):
            if mode == "messages":
                # chunk is (message_chunk, metadata)
                msg_chunk, metadata = chunk
                if msg_chunk.content:
                    event = {
                        "type": "token",
                        "content": msg_chunk.content,
                        "agent": metadata.get("langgraph_node"),
                        "tags": metadata.get("tags", [])
                    }
                    yield f"event: token\ndata: {json.dumps(event)}\n\n"
                    
            elif mode == "updates":
                # chunk is state update dict
                node_name = list(chunk.keys())[0] if chunk else None
                if node_name and node_name != "__end__":
                    event = {
                        "type": "agent_start",
                        "node": node_name
                    }
                    yield f"event: agent_start\ndata: {json.dumps(event)}\n\n"
                    
            elif mode == "custom":
                # chunk is custom data from get_stream_writer()
                # Check if it's a status event (ChatGPT/Claude-style)
                if isinstance(chunk, dict) and "status" in chunk:
                    # Extract the StatusEvent and emit as 'status' event type
                    status_event = chunk["status"]
                    yield f"event: status\ndata: {json.dumps(status_event)}\n\n"
                else:
                    # Legacy progress event (for backward compatibility)
                    event = {
                        "type": "progress",
                        "data": chunk
                    }
                    yield f"event: progress\ndata: {json.dumps(event)}\n\n"
        
        # Send final completion chunk
        final_chunk = ChatCompletionChunk(
            id=run_id,
            created=int(time.time()),
            model="gpt-4o",
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info(
            "Resume completed",
            extra={"run_id": run_id, "thread_id": resume_request.thread_id}
        )

    except Exception as e:
        logger.error(
            "Error in resume stream",
            extra={
                "run_id": run_id,
                "thread_id": resume_request.thread_id,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app_api.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> StreamingResponse:
    """OpenAI-compatible chat completions endpoint.
    
    Accepts chat completion requests in OpenAI format and returns streaming
    responses via Server-Sent Events. Supports human-in-the-loop workflows.
    
    Args:
        request: OpenAI-format request containing messages, model, thread_id, etc.
        
    Returns:
        StreamingResponse: SSE stream containing:
            - token events: LLM output with agent metadata
            - agent_start events: Agent routing notifications
            - progress events: Tool progress updates
            - OpenAI-compatible completion chunks
            
    Raises:
        HTTPException 501: If stream=False (non-streaming not implemented).
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    run_id = f"run_{uuid.uuid4()}"
    
    if request.stream:
        return StreamingResponse(
            _stream_response(request, run_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Run-ID": run_id
            }
        )
    else:
        # Non-streaming implementation
        raise HTTPException(
            status_code=501, 
            detail="Non-streaming not implemented yet. Set stream=True in your request."
        )


@app_api.post("/v1/chat/resume")
async def resume_conversation(resume_request: ResumeRequest) -> StreamingResponse:
    """Resume an interrupted conversation with human input.
    
    This endpoint allows clients to provide human responses to interrupt()
    calls, enabling human-in-the-loop workflows.
    
    Args:
        resume_request: Contains thread_id and human response data.
        
    Returns:
        StreamingResponse: SSE stream of completion chunks from the point
            of interruption.
            
    Raises:
        HTTPException 501: If stream=False (non-streaming not implemented).
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    run_id = f"run_{uuid.uuid4()}"
    
    if resume_request.stream:
        return StreamingResponse(
            _stream_resume_response(resume_request, run_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Run-ID": run_id
            }
        )
    else:
        raise HTTPException(
            status_code=501, 
            detail="Non-streaming not implemented yet. Set stream=True in your request."
        )


@app_api.get("/v1/chat/interrupt/{thread_id}", response_model=InterruptStatus)
async def get_interrupt_status(thread_id: str) -> InterruptStatus:
    """Get the interrupt status for a thread.
    
    Allows clients to poll for pending interrupts without streaming.
    Use this to check if a conversation is waiting for human input.
    
    Args:
        thread_id: The thread ID to check for pending interrupts.
        
    Returns:
        InterruptStatus: Contains:
            - has_interrupt: Whether an interrupt is pending
            - interrupt_data: Data provided to interrupt() call
            - interrupt_type: Type of interrupt (code_confirmation, etc.)
            
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    try:
        config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        
        # Check for interrupts in state
        interrupts = state.tasks[0].interrupts if state.tasks else []
        
        if interrupts:
            interrupt_value = interrupts[0].value if interrupts else None
            return InterruptStatus(
                thread_id=thread_id,
                has_interrupt=True,
                interrupt_data=interrupt_value,
                interrupt_type=interrupt_value.get("action_type") if isinstance(interrupt_value, dict) else None
            )
        else:
            return InterruptStatus(
                thread_id=thread_id,
                has_interrupt=False
            )
            
    except Exception as e:
        logger.error(
            "Error checking interrupt status",
            extra={"thread_id": thread_id, "error": str(e)}
        )
        return InterruptStatus(
            thread_id=thread_id,
            has_interrupt=False
        )


@app_api.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint for load balancers and monitoring.
    
    Returns:
        dict: Health status with service identifier and version.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    return {
        "status": "ok",
        "service": "chat-app",
        "version": "2.0.0"
    }
