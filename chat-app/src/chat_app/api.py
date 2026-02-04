"""OpenAI-compatible REST API adapter for Chat-App with human-in-the-loop support.

Provides a FastAPI server that exposes the LangGraph supervisor as an
OpenAI-compatible chat completions endpoint. Supports human-in-the-loop
workflows with interrupt/resume functionality.

Endpoints:
    POST /v1/chat/completions: Streaming chat completion (OpenAI-compatible)
    POST /v1/chat/resume: Resume interrupted conversation with human input
    GET /v1/chat/interrupt: Get pending interrupt status
    GET /health: Health check endpoint

The adapter converts OpenAI message formats to LangChain messages,
invokes the supervisor graph, and streams responses back in the
OpenAI Server-Sent Events (SSE) format.

Human-in-the-loop flow:
1. Graph encounters interrupt() - returns __interrupt__ in response
2. Client polls /v1/chat/interrupt to see pending interrupts
3. Client POSTs to /v1/chat/resume with human response
4. Graph continues execution from interrupt point

Last Grunted: 02/04/2026 03:30:00 PM PST
"""

import time
import uuid
import json
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from chat_app.graphs.app import app as graph

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-app.api")

app_api = FastAPI(title="Chat App (Agent)", version="2.0.0")


# OpenAI Schema Models
class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[ChatMessage]
    stream: bool = False
    # Extra fields for specialized agent control (optional)
    thread_id: Optional[str] = None 
    user_id: Optional[str] = None


class ResumeRequest(BaseModel):
    """Request to resume an interrupted conversation."""
    thread_id: str = Field(..., description="Thread ID of the interrupted conversation")
    resume_data: Dict[str, Any] = Field(..., description="Human response to the interrupt")
    stream: bool = True


class InterruptStatus(BaseModel):
    """Status of pending interrupts."""
    thread_id: str
    has_interrupt: bool
    interrupt_data: Optional[Dict[str, Any]] = None
    interrupt_type: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


# ------------------------------------------------------------------
# Adapter Logic
# ------------------------------------------------------------------

def _convert_messages(messages: List[ChatMessage]):
    """Convert OpenAI-format messages to LangChain message objects.
    
    Transforms a list of ChatMessage objects (OpenAI format) into the
    corresponding LangChain message types for use with the supervisor graph.
    
    Args:
        messages: List of OpenAI-format chat messages
        
    Returns:
        list: List of LangChain message objects
    """
    lc_messages = []
    for msg in messages:
        if msg.role == "user":
            lc_messages.append(HumanMessage(content=msg.content, name=msg.name))
        elif msg.role == "assistant":
            lc_messages.append(AIMessage(content=msg.content, name=msg.name))
        elif msg.role == "system":
            lc_messages.append(SystemMessage(content=msg.content, name=msg.name))
    return lc_messages


async def _stream_response(
    request: ChatCompletionRequest, 
    run_id: str
) -> AsyncGenerator[str, None]:
    """Stream LangGraph responses as OpenAI-compatible Server-Sent Events.
    
    Async generator that invokes the supervisor graph with the user's messages
    and yields response chunks in the OpenAI SSE format (data: {json}\n\n).
    
    Handles interrupts by yielding special interrupt chunks that the client
    can detect and respond to via the /resume endpoint.
    
    Args:
        request: The incoming chat completion request
        run_id: Unique identifier for this run
        
    Yields:
        str: SSE-formatted strings in format "data: {json}\n\n"
    """
    # Prepare Input
    inputs = {
        "messages": _convert_messages(request.messages),
    }
    
    config = {
        "configurable": {
            "thread_id": request.thread_id or str(uuid.uuid4()),
            "user_id": request.user_id
        }
    }

    logger.info(f"Starting run {run_id} for thread {config['configurable']['thread_id']}")
    
    try:
        async for event in graph.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            
            # Handle interrupts
            if kind == "on_interrupt":
                interrupt_data = event.get("data", {})
                interrupt_chunk = ChatCompletionChunk(
                    id=run_id,
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta={
                                "role": "assistant",
                                "content": "",
                                "__interrupt__": interrupt_data
                            },
                            finish_reason="interrupt"
                        )
                    ]
                )
                yield f"data: {interrupt_chunk.model_dump_json()}\n\n"
                yield "data: [INTERRUPT]\n\n"
                return
            
            # Regular chat model streaming
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    chunk = ChatCompletionChunk(
                        id=run_id,
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta={"content": content, "role": "assistant"},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Finish
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
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in stream: {e}", exc_info=True)
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
    
    Args:
        resume_request: The resume request with human response
        run_id: Unique identifier for this run
        
    Yields:
        str: SSE-formatted strings
    """
    config = {
        "configurable": {
            "thread_id": resume_request.thread_id
        }
    }
    
    # Create Command to resume with human input
    resume_command = Command(resume=resume_request.resume_data)
    
    try:
        async for event in graph.astream_events(resume_command, config=config, version="v1"):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    chunk = ChatCompletionChunk(
                        id=run_id,
                        created=int(time.time()),
                        model="gpt-4o",
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta={"content": content, "role": "assistant"},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
        
        # Finish
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

    except Exception as e:
        logger.error(f"Error in resume stream: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app_api.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.
    
    Accepts chat completion requests in OpenAI format and returns streaming
    responses via Server-Sent Events. Supports human-in-the-loop workflows.
    
    Args:
        request: OpenAI-format request containing messages, model, etc.
        
    Returns:
        StreamingResponse: SSE stream of ChatCompletionChunk objects.
        May include interrupt markers for human-in-the-loop.
    """
    run_id = f"run_{uuid.uuid4()}"
    
    if request.stream:
        return StreamingResponse(
            _stream_response(request, run_id),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming implementation
        raise HTTPException(status_code=501, detail="Non-streaming not implemented yet (use stream=True)")


@app_api.post("/v1/chat/resume")
async def resume_conversation(resume_request: ResumeRequest):
    """Resume an interrupted conversation with human input.
    
    This endpoint allows clients to provide human responses to interrupt()
    calls, enabling human-in-the-loop workflows.
    
    Args:
        resume_request: Contains thread_id and human response data
        
    Returns:
        StreamingResponse: SSE stream of completion chunks
    """
    run_id = f"run_{uuid.uuid4()}"
    
    if resume_request.stream:
        return StreamingResponse(
            _stream_resume_response(resume_request, run_id),
            media_type="text/event-stream"
        )
    else:
        raise HTTPException(status_code=501, detail="Non-streaming not implemented yet (use stream=True)")


@app_api.get("/v1/chat/interrupt/{thread_id}")
async def get_interrupt_status(thread_id: str):
    """Get the interrupt status for a thread.
    
    Allows clients to poll for pending interrupts without streaming.
    
    Args:
        thread_id: The thread ID to check
        
    Returns:
        InterruptStatus: Status of pending interrupts
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        
        # Check for interrupts in state
        interrupts = state.tasks[0].interrupts if state.tasks else []
        
        if interrupts:
            return InterruptStatus(
                thread_id=thread_id,
                has_interrupt=True,
                interrupt_data=interrupts[0].value if interrupts else None,
                interrupt_type=interrupts[0].value.get("action_type") if interrupts else None
            )
        else:
            return InterruptStatus(
                thread_id=thread_id,
                has_interrupt=False
            )
            
    except Exception as e:
        logger.error(f"Error checking interrupt status: {e}")
        return InterruptStatus(
            thread_id=thread_id,
            has_interrupt=False
        )


@app_api.get("/health")
async def health():
    """Health check endpoint for load balancers and monitoring.
    
    Returns:
        dict: Health status with service identifier
    """
    return {"status": "ok", "service": "chat-app", "version": "2.0.0"}
