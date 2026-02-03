"""OpenAI-compatible REST API adapter for Chat-App.

Provides a FastAPI server that exposes the LangGraph supervisor as an
OpenAI-compatible chat completions endpoint. This enables integration
with tools and clients that expect the OpenAI API format.

Endpoints:
    POST /v1/chat/completions: Streaming chat completion (OpenAI-compatible)
    GET /health: Health check endpoint

The adapter converts OpenAI message formats to LangChain messages,
invokes the supervisor graph, and streams responses back in the
OpenAI Server-Sent Events (SSE) format.

Dependencies:
    - fastapi>=0.100.0 for REST API framework
    - pydantic>=2.5.0 for request/response models
    - langchain_core for message types

Last Grunted: 02/03/2026 04:00:00 PM PST
"""

import time
import uuid
import json
import logging
from typing import AsyncGenerator, List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chat_app.graphs.app import app as graph

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-app.api")

app_api = FastAPI(title="Chat App (Agent)", version="1.0.0")

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
        messages (List[ChatMessage]): List of OpenAI-format chat messages
            with role ("user", "assistant", "system") and content fields.

    Returns:
        list: List of LangChain message objects:
            - HumanMessage for role="user"
            - AIMessage for role="assistant"
            - SystemMessage for role="system"
            Messages with unrecognized roles are silently skipped.

    Example:
        >>> msgs = [ChatMessage(role="user", content="Hello")]
        >>> _convert_messages(msgs)
        [HumanMessage(content="Hello", name=None)]

    Last Grunted: 02/03/2026 04:00:00 PM PST
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

async def _stream_response(request: ChatCompletionRequest, run_id: str) -> AsyncGenerator[str, None]:
    """Stream LangGraph responses as OpenAI-compatible Server-Sent Events.

    Async generator that invokes the supervisor graph with the user's messages
    and yields response chunks in the OpenAI SSE format (data: {json}\n\n).

    The function:
    1. Converts OpenAI messages to LangChain format
    2. Streams events from the supervisor graph via astream_events
    3. Converts LLM stream chunks to OpenAI ChatCompletionChunk format
    4. Yields SSE-formatted strings ending with [DONE] marker

    Args:
        request (ChatCompletionRequest): The incoming chat completion request
            containing messages, model selection, and optional thread_id/user_id.
        run_id (str): Unique identifier for this run (format: "run_{uuid}").

    Yields:
        str: SSE-formatted strings in format "data: {json}\n\n" where json
            is a ChatCompletionChunk. Final yield is "data: [DONE]\n\n".

    Raises:
        No exceptions raised - errors are caught and yielded as error chunks.

    Example:
        >>> async for chunk in _stream_response(request, "run_123"):
        ...     print(chunk)
        data: {"id":"run_123","object":"chat.completion.chunk",...}

    Last Grunted: 02/03/2026 04:00:00 PM PST
    """
    
    # Prepare Input
    inputs = {
        "messages": _convert_messages(request.messages),
        # Pass user_id if available to the graph config or state
    }
    
    config = {
        "configurable": {
            "thread_id": request.thread_id or str(uuid.uuid4()),
            "user_id": request.user_id
        }
    }

    logger.info(f"Starting run {run_id} for thread {config['configurable']['thread_id']}")
    
    # Stream from Graph
    # We assume the graph streams AIMessageChunk or similar
    try:
        async for event in graph.astream_events(inputs, config=config, version="v1"):
            kind = event["event"]
            
            # We are interested in "on_chat_model_stream" or output from the final node
            # Adjust logic based on how 'supervisor' graph emits events.
            # For simplicity, let's catch 'on_chat_model_stream' from the active agent.
            
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
        # OpenAI doesn't have a clean error chunk format, usually just closes connection or yields error json
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app_api.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Accepts chat completion requests in OpenAI format and returns streaming
    responses via Server-Sent Events. The request is processed by the
    LangGraph supervisor which orchestrates specialized agents.

    Args:
        request (ChatCompletionRequest): OpenAI-format request containing:
            - model: Model identifier (passed through, supervisor uses configured model)
            - messages: List of chat messages
            - stream: Must be True (non-streaming not implemented)
            - thread_id: Optional conversation thread identifier
            - user_id: Optional user identifier

    Returns:
        StreamingResponse: SSE stream of ChatCompletionChunk objects.

    Raises:
        HTTPException: 501 if stream=False (not implemented).

    Note:
        Non-streaming mode is not yet implemented. Always set stream=True.

    Last Grunted: 02/03/2026 04:00:00 PM PST
    """
    run_id = f"run_{uuid.uuid4()}"
    
    if request.stream:
        return StreamingResponse(
            _stream_response(request, run_id),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming implementation (accumulate)
        # TODO: Implement invoke() wait
        raise HTTPException(status_code=501, detail="Non-streaming not implemented yet (use stream=True)")

@app_api.get("/health")
async def health():
    """Health check endpoint for load balancers and monitoring.

    Returns a simple JSON status indicating the service is running.
    Used by Kubernetes probes and external health monitoring systems.

    Returns:
        dict: Health status with keys:
            - status: "ok" if service is healthy
            - service: "chat-app" service identifier

    Last Grunted: 02/03/2026 04:00:00 PM PST
    """
    return {"status": "ok", "service": "chat-app"}
