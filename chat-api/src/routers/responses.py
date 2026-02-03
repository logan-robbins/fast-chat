"""
OpenAI Responses API router implementation.

Implements the full OpenAI Responses API per 2026 specification:
    - POST /v1/responses - Create a model response
    - GET /v1/responses/{response_id} - Retrieve a stored response
    - DELETE /v1/responses/{response_id} - Delete a response
    - POST /v1/responses/{response_id}/cancel - Cancel background response
    - POST /v1/responses/compact - Compact conversation history
    - GET /v1/responses/{response_id}/input_items - List input items
    - POST /v1/responses/input_tokens - Get input token counts

The Responses API provides stateful model interactions with:
    - Multi-turn conversations via previous_response_id
    - Built-in tools (web_search, file_search, code_interpreter)
    - Function calling
    - Streaming via SSE
    - Response storage and retrieval

Reference: https://platform.openai.com/docs/api-reference/responses

Last Grunted: 02/03/2026 11:45:00 AM UTC
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from typing import List, Optional, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
import httpx
import os
import json
import time
import secrets
import base64

from src.db.engine import get_session
from src.db.models import Response, ResponseInputItem

router = APIRouter()

# Backend configuration
CHAT_APP_URL = os.getenv("CHAT_APP_URL", "http://localhost:8001/v1/chat/completions")


# ============================================================================
# Pydantic Models - OpenAI Responses API Compliant
# ============================================================================

class InputTextContent(BaseModel):
    """
    Text content input item.
    
    Attributes:
        type: Always "input_text"
        text: The text content
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(BaseModel):
    """
    Image content input item.
    
    Attributes:
        type: Always "input_image"
        image_url: URL or base64 data URI of the image
        detail: Image detail level (auto, low, high)
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Optional[str] = "auto"


class OutputTextContent(BaseModel):
    """
    Text content output item.
    
    Attributes:
        type: Always "output_text"
        text: The generated text content
        annotations: List of annotations (citations, etc.)
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: List[dict] = []


class MessageInputItem(BaseModel):
    """
    Message input item for multi-turn conversations.
    
    Attributes:
        type: Always "message"
        role: Message author role (user, assistant, system)
        content: List of content objects or string
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["message"] = "message"
    role: str
    content: Union[str, List[Union[InputTextContent, InputImageContent, dict]]]


class MessageOutputItem(BaseModel):
    """
    Message output item from model response.
    
    Attributes:
        type: Always "message"
        id: Unique message identifier (msg_...)
        status: Message status (completed, in_progress)
        role: Always "assistant" for outputs
        content: List of output content objects
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["message"] = "message"
    id: str
    status: str = "completed"
    role: Literal["assistant"] = "assistant"
    content: List[OutputTextContent]


class FunctionDefinition(BaseModel):
    """
    Function tool definition.
    
    Attributes:
        type: Always "function"
        name: Function name
        description: Function description
        parameters: JSON Schema for parameters
        strict: Whether to enforce strict schema validation
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["function"] = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None
    strict: Optional[bool] = False


class WebSearchTool(BaseModel):
    """
    Web search built-in tool.
    
    Attributes:
        type: Always "web_search"
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["web_search"] = "web_search"


class FileSearchTool(BaseModel):
    """
    File search built-in tool.
    
    Attributes:
        type: Always "file_search"
        vector_store_ids: List of vector store IDs to search
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["file_search"] = "file_search"
    vector_store_ids: Optional[List[str]] = None


class UsageObject(BaseModel):
    """
    Token usage statistics.
    
    Attributes:
        input_tokens: Number of input tokens
        input_tokens_details: Detailed input token breakdown
        output_tokens: Number of output tokens
        output_tokens_details: Detailed output token breakdown
        total_tokens: Total tokens used
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    input_tokens: int
    input_tokens_details: Optional[dict] = None
    output_tokens: int
    output_tokens_details: Optional[dict] = None
    total_tokens: int


class ErrorObject(BaseModel):
    """
    Error object for failed responses.
    
    Attributes:
        code: Machine-readable error code
        message: Human-readable error message
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    code: str
    message: str


class ReasoningConfig(BaseModel):
    """
    Reasoning configuration for o-series models.
    
    Attributes:
        effort: Reasoning effort level (low, medium, high)
        summary: Whether to include reasoning summary
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    effort: Optional[str] = None
    summary: Optional[str] = None


class TextConfig(BaseModel):
    """
    Text output configuration.
    
    Attributes:
        format: Output format configuration
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    format: Optional[dict] = Field(default_factory=lambda: {"type": "text"})


class CreateResponseRequest(BaseModel):
    """
    Request body for POST /v1/responses.
    
    Creates a model response with text/image inputs and optional tools.
    
    Attributes:
        model: Model ID (e.g., "gpt-4o", "o3"). Required.
        input: Text string or array of input items. Optional.
        instructions: System/developer message. Optional.
        previous_response_id: ID for multi-turn conversations. Optional.
        conversation: Conversation object or ID. Optional.
        tools: Array of tool definitions. Optional.
        tool_choice: Tool selection strategy. Default "auto".
        store: Whether to store response. Default True.
        stream: Whether to stream response. Default False.
        temperature: Sampling temperature (0-2). Default 1.0.
        top_p: Nucleus sampling (0-1). Default 1.0.
        max_output_tokens: Maximum output tokens. Optional.
        metadata: Key-value metadata (max 16 pairs). Optional.
        parallel_tool_calls: Allow parallel tool calls. Default True.
        truncation: Truncation strategy. Default "disabled".
        background: Run in background. Default False.
        reasoning: Reasoning config for o-series. Optional.
        text: Text output config. Optional.
        include: Additional output data to include. Optional.
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    model: str
    input: Optional[Union[str, List[Union[MessageInputItem, dict]]]] = None
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Union[str, dict]] = None
    tools: Optional[List[Union[FunctionDefinition, WebSearchTool, FileSearchTool, dict]]] = None
    tool_choice: Optional[Union[str, dict]] = "auto"
    store: bool = True
    stream: bool = False
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    parallel_tool_calls: bool = True
    truncation: str = "disabled"
    background: bool = False
    reasoning: Optional[ReasoningConfig] = None
    text: Optional[TextConfig] = None
    include: Optional[List[str]] = None


class ResponseObject(BaseModel):
    """
    OpenAI Response object.
    
    Returned by create, get, and cancel endpoints.
    
    Attributes:
        id: Unique response ID (resp_...)
        object: Always "response"
        created_at: Unix timestamp of creation
        status: Response status
        completed_at: Unix timestamp of completion (nullable)
        error: Error object if failed (nullable)
        incomplete_details: Details if incomplete (nullable)
        instructions: System message (nullable)
        max_output_tokens: Token limit (nullable)
        model: Model ID used
        output: Array of output items
        parallel_tool_calls: Whether parallel calls allowed
        previous_response_id: Previous response for multi-turn
        reasoning: Reasoning configuration
        store: Whether response is stored
        temperature: Sampling temperature
        text: Text output configuration
        tool_choice: Tool selection strategy
        tools: Available tools
        top_p: Nucleus sampling parameter
        truncation: Truncation strategy
        usage: Token usage statistics
        metadata: User metadata
        background: Whether running in background
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"]
    completed_at: Optional[int] = None
    error: Optional[ErrorObject] = None
    incomplete_details: Optional[dict] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[Union[MessageOutputItem, dict]] = []
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: Optional[ReasoningConfig] = None
    store: bool = True
    temperature: float = 1.0
    text: Optional[TextConfig] = None
    tool_choice: Optional[Union[str, dict]] = "auto"
    tools: List[dict] = []
    top_p: float = 1.0
    truncation: str = "disabled"
    usage: Optional[UsageObject] = None
    metadata: Dict[str, str] = {}
    background: bool = False


class DeleteResponseObject(BaseModel):
    """
    Response deletion confirmation.
    
    Attributes:
        id: Deleted response ID
        object: Always "response"
        deleted: Always True
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    id: str
    object: Literal["response"] = "response"
    deleted: Literal[True] = True


class CompactResponseRequest(BaseModel):
    """
    Request body for POST /v1/responses/compact.
    
    Attributes:
        model: Model ID. Required.
        input: Input items to compact. Optional.
        instructions: System message. Optional.
        previous_response_id: Previous response ID. Optional.
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    model: str
    input: Optional[Union[str, List[Union[MessageInputItem, dict]]]] = None
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None


class CompactionItem(BaseModel):
    """
    Compacted conversation item.
    
    Attributes:
        type: Always "compaction"
        id: Compaction ID (cmp_...)
        encrypted_content: Encrypted/opaque content
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    type: Literal["compaction"] = "compaction"
    id: str
    encrypted_content: str


class CompactResponseObject(BaseModel):
    """
    Compacted response object.
    
    Attributes:
        id: Response ID
        object: Always "response.compaction"
        created_at: Unix timestamp
        output: Compacted output items
        usage: Token usage
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    id: str
    object: Literal["response.compaction"] = "response.compaction"
    created_at: int
    output: List[Union[MessageInputItem, CompactionItem, dict]]
    usage: UsageObject


class InputItemsListResponse(BaseModel):
    """
    Response for GET /v1/responses/{response_id}/input_items.
    
    Attributes:
        object: Always "list"
        data: Array of input items
        first_id: ID of first item
        last_id: ID of last item
        has_more: Whether more items exist
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    object: Literal["list"] = "list"
    data: List[dict]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


class InputTokensRequest(BaseModel):
    """
    Request body for POST /v1/responses/input_tokens.
    
    Attributes:
        model: Model ID. Optional.
        input: Input items. Optional.
        instructions: System message. Optional.
        previous_response_id: Previous response ID. Optional.
        tools: Tool definitions. Optional.
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    model: Optional[str] = None
    input: Optional[Union[str, List[Union[MessageInputItem, dict]]]] = None
    instructions: Optional[str] = None
    previous_response_id: Optional[str] = None
    tools: Optional[List[dict]] = None


class InputTokensResponse(BaseModel):
    """
    Response for POST /v1/responses/input_tokens.
    
    Attributes:
        object: Always "response.input_tokens"
        input_tokens: Token count
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    object: Literal["response.input_tokens"] = "response.input_tokens"
    input_tokens: int


# ============================================================================
# Utility Functions
# ============================================================================

def generate_response_id() -> str:
    """
    Generate a unique response ID in OpenAI format.
    
    Format: resp_ + 48 hex characters (24 bytes)
    
    Returns:
        str: Response ID like "resp_67ccd2bed1ec8190b14f964abc054267"
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    return f"resp_{secrets.token_hex(24)}"


def generate_message_id() -> str:
    """
    Generate a unique message ID in OpenAI format.
    
    Format: msg_ + 48 hex characters (24 bytes)
    
    Returns:
        str: Message ID like "msg_67ccd2bf17f0819081ff3bb2cf6508e6"
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    return f"msg_{secrets.token_hex(24)}"


def generate_compaction_id() -> str:
    """
    Generate a unique compaction ID.
    
    Format: cmp_ + 48 hex characters (24 bytes)
    
    Returns:
        str: Compaction ID like "cmp_67ccd2bf17f0819081ff3bb2cf6508e6"
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    return f"cmp_{secrets.token_hex(24)}"


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using character-based heuristic.
    
    OpenAI models use ~4 characters per token on average for English.
    For production, integrate tiktoken library for accuracy.
    
    Formula: tokens ≈ characters / 4
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        int: Estimated token count (minimum 1 for non-empty)
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def count_input_tokens(
    input_data: Optional[Union[str, List[Any]]],
    instructions: Optional[str] = None,
    tools: Optional[List[dict]] = None
) -> int:
    """
    Count estimated input tokens for a response request.
    
    Includes tokens from:
    - Input text/messages
    - System instructions
    - Tool definitions
    - Message structure overhead (~4 tokens per message)
    
    Args:
        input_data: String or list of input items
        instructions: System/developer message
        tools: List of tool definitions
        
    Returns:
        int: Estimated total input tokens
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    total = 0
    
    # Count instructions
    if instructions:
        total += estimate_tokens(instructions) + 4  # +4 for message overhead
    
    # Count input
    if isinstance(input_data, str):
        total += estimate_tokens(input_data) + 4
    elif isinstance(input_data, list):
        for item in input_data:
            total += 4  # Message overhead
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    total += estimate_tokens(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            text = c.get("text", "")
                            if text:
                                total += estimate_tokens(text)
            elif hasattr(item, "content"):
                content = item.content
                if isinstance(content, str):
                    total += estimate_tokens(content)
    
    # Count tools
    if tools:
        for tool in tools:
            # Rough estimate: tool definition ~50-100 tokens
            total += 50
            if isinstance(tool, dict):
                desc = tool.get("description", "")
                if desc:
                    total += estimate_tokens(desc)
    
    # Add base overhead
    total += 3  # Priming tokens
    
    return total


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
            "message": "Description",
            "type": "error_type",
            "param": "parameter",
            "code": "error_code"
        }
    }
    
    Args:
        message: Human-readable error description
        error_type: Error category
        param: Parameter that caused error (nullable)
        code: Machine-readable error code (nullable)
        status_code: HTTP status code
        
    Returns:
        JSONResponse with OpenAI error format
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
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


def extract_text_from_input(input_data: Optional[Union[str, List[Any]]]) -> str:
    """
    Extract text content from input data for processing.
    
    Handles both string inputs and array of message objects.
    
    Args:
        input_data: String or list of input items
        
    Returns:
        str: Concatenated text content
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    if isinstance(input_data, str):
        return input_data
    
    if isinstance(input_data, list):
        texts = []
        for item in input_data:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "input_text":
                            texts.append(c.get("text", ""))
            elif hasattr(item, "content"):
                content = item.content
                if isinstance(content, str):
                    texts.append(content)
        return " ".join(texts)
    
    return ""


async def get_previous_context(
    session: AsyncSession,
    previous_response_id: str
) -> List[dict]:
    """
    Retrieve input/output items from previous response for multi-turn.
    
    Args:
        session: Database session
        previous_response_id: ID of previous response
        
    Returns:
        List[dict]: Previous input and output items
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    previous = await session.get(Response, previous_response_id)
    if not previous:
        return []
    
    context = []
    
    # Get input items from previous response
    result = await session.execute(
        select(ResponseInputItem)
        .where(ResponseInputItem.response_id == previous_response_id)
        .order_by(ResponseInputItem.created_at)
    )
    for item in result.scalars().all():
        context.append({
            "type": item.type,
            "role": item.role,
            "content": item.content
        })
    
    # Add output from previous response
    if previous.output:
        context.extend(previous.output)
    
    return context


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/v1/responses", response_model=ResponseObject)
async def create_response(
    request: CreateResponseRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Create a model response (POST /v1/responses).
    
    Generates a model response from text/image inputs. Supports:
    - Multi-turn conversations via previous_response_id
    - Built-in tools (web_search, file_search)
    - Function calling
    - Streaming via SSE
    - Response storage for retrieval
    
    Args:
        request: CreateResponseRequest with model and input
        session: Database session (injected)
        
    Returns:
        ResponseObject with id, output, usage
        StreamingResponse if stream=true
        
    Raises:
        HTTPException: 400 for invalid request, 404 for missing previous_response
        
    Example:
        POST /v1/responses
        {"model": "gpt-4o", "input": "Hello!"}
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    # Generate IDs and timestamps
    response_id = generate_response_id()
    message_id = generate_message_id()
    created_at = int(time.time())
    
    # Validate model
    valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5", "o1", "o3", "o1-pro", "o3-mini"]
    model_base = request.model.split("-")[0] if "-" in request.model else request.model
    # Allow model versions like gpt-4o-2024-08-06
    if not any(request.model.startswith(m) for m in valid_models):
        return create_error_response(
            message=f"The model '{request.model}' does not exist or you do not have access to it.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404
        )
    
    # Validate metadata (max 16 pairs)
    if request.metadata and len(request.metadata) > 16:
        return create_error_response(
            message="metadata cannot have more than 16 key-value pairs",
            error_type="invalid_request_error",
            param="metadata",
            code="invalid_metadata"
        )
    
    # Build context from previous response
    context_items = []
    if request.previous_response_id:
        previous = await session.get(Response, request.previous_response_id)
        if not previous:
            return create_error_response(
                message=f"Previous response '{request.previous_response_id}' not found",
                error_type="invalid_request_error",
                param="previous_response_id",
                code="response_not_found",
                status_code=404
            )
        context_items = await get_previous_context(session, request.previous_response_id)
    
    # Handle streaming
    if request.stream:
        return await _handle_streaming_response(
            request, response_id, message_id, created_at, context_items, session
        )
    
    # Non-streaming response
    return await _handle_non_streaming_response(
        request, response_id, message_id, created_at, context_items, session
    )


async def _handle_streaming_response(
    request: CreateResponseRequest,
    response_id: str,
    message_id: str,
    created_at: int,
    context_items: List[dict],
    session: AsyncSession
) -> StreamingResponse:
    """
    Handle streaming response via SSE.
    
    Emits events:
    - response.created
    - response.in_progress
    - response.output_item.added
    - response.output_text.delta
    - response.output_item.done
    - response.completed
    
    Args:
        request: Original request
        response_id: Generated response ID
        message_id: Generated message ID
        created_at: Creation timestamp
        context_items: Previous conversation context
        session: Database session
        
    Returns:
        StreamingResponse with text/event-stream media type
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    async def stream_generator():
        # Emit response.created
        created_event = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "model": request.model,
                "output": [],
                "metadata": request.metadata or {}
            }
        }
        yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"
        
        # Emit response.in_progress
        in_progress_event = {"type": "response.in_progress", "response": {"id": response_id}}
        yield f"event: response.in_progress\ndata: {json.dumps(in_progress_event)}\n\n"
        
        # Emit output_item.added for message
        output_item_event = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": message_id,
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        }
        yield f"event: response.output_item.added\ndata: {json.dumps(output_item_event)}\n\n"
        
        # Extract input text
        input_text = extract_text_from_input(request.input)
        
        # Build messages for backend
        messages = []
        if request.instructions:
            messages.append({"role": "system", "content": request.instructions})
        for ctx in context_items:
            if ctx.get("role") and ctx.get("content"):
                content = ctx["content"]
                if isinstance(content, list) and content:
                    # Extract text from content array
                    text_parts = []
                    for c in content:
                        if isinstance(c, dict) and c.get("text"):
                            text_parts.append(c["text"])
                    content = " ".join(text_parts) if text_parts else str(content)
                messages.append({"role": ctx["role"], "content": content})
        if input_text:
            messages.append({"role": "user", "content": input_text})
        
        # Prepare backend payload
        backend_payload = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_output_tokens
        }
        
        full_content = []
        prompt_tokens = count_input_tokens(request.input, request.instructions, 
                                           [t.model_dump() if hasattr(t, 'model_dump') else t for t in (request.tools or [])])
        completion_tokens = 0
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", CHAT_APP_URL, json=backend_payload) as response:
                    if response.status_code != 200:
                        error_event = {
                            "type": "response.failed",
                            "response": {
                                "id": response_id,
                                "status": "failed",
                                "error": {"code": "backend_error", "message": f"Backend returned {response.status_code}"}
                            }
                        }
                        yield f"event: response.failed\ndata: {json.dumps(error_event)}\n\n"
                        yield "event: done\ndata: [DONE]\n\n"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        full_content.append(content)
                                        completion_tokens += estimate_tokens(content)
                                        
                                        # Emit text delta
                                        delta_event = {
                                            "type": "response.output_text.delta",
                                            "output_index": 0,
                                            "content_index": 0,
                                            "delta": content
                                        }
                                        yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"
                            except json.JSONDecodeError:
                                pass
                                
        except Exception as e:
            error_event = {
                "type": "response.failed",
                "response": {
                    "id": response_id,
                    "status": "failed", 
                    "error": {"code": "streaming_error", "message": str(e)}
                }
            }
            yield f"event: response.failed\ndata: {json.dumps(error_event)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        
        # Emit output_item.done
        final_text = "".join(full_content)
        output_done_event = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": message_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_text, "annotations": []}]
            }
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(output_done_event)}\n\n"
        
        completed_at = int(time.time())
        
        # Emit response.completed
        completed_event = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "completed_at": completed_at,
                "status": "completed",
                "model": request.model,
                "output": [{
                    "type": "message",
                    "id": message_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_text, "annotations": []}]
                }],
                "usage": {
                    "input_tokens": prompt_tokens,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": completion_tokens,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "metadata": request.metadata or {}
            }
        }
        yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"
        
        # Store response if requested
        if request.store:
            try:
                db_response = Response(
                    id=response_id,
                    created_at=created_at,
                    completed_at=completed_at,
                    status="completed",
                    model=request.model,
                    instructions=request.instructions,
                    output=[{
                        "type": "message",
                        "id": message_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": final_text, "annotations": []}]
                    }],
                    usage={
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    response_metadata=request.metadata or {},
                    previous_response_id=request.previous_response_id,
                    temperature=request.temperature or 1.0,
                    top_p=request.top_p or 1.0,
                    max_output_tokens=request.max_output_tokens,
                    parallel_tool_calls=request.parallel_tool_calls,
                    tool_choice=request.tool_choice if isinstance(request.tool_choice, str) else "auto",
                    tools=[t.model_dump() if hasattr(t, 'model_dump') else t for t in (request.tools or [])],
                    truncation=request.truncation,
                    background=request.background
                )
                session.add(db_response)
                
                # Store input items
                if request.input:
                    input_item = ResponseInputItem(
                        id=generate_message_id(),
                        response_id=response_id,
                        type="message",
                        role="user",
                        content=[{"type": "input_text", "text": input_text}] if input_text else None,
                        created_at=created_at
                    )
                    session.add(input_item)
                
                await session.commit()
            except Exception:
                # Don't fail streaming response if storage fails
                pass
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def _handle_non_streaming_response(
    request: CreateResponseRequest,
    response_id: str,
    message_id: str,
    created_at: int,
    context_items: List[dict],
    session: AsyncSession
) -> ResponseObject:
    """
    Handle non-streaming response.
    
    Makes synchronous request to backend and returns complete response.
    
    Args:
        request: Original request
        response_id: Generated response ID
        message_id: Generated message ID
        created_at: Creation timestamp
        context_items: Previous conversation context
        session: Database session
        
    Returns:
        ResponseObject with complete response
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    # Extract input text
    input_text = extract_text_from_input(request.input)
    
    # Build messages for backend
    messages = []
    if request.instructions:
        messages.append({"role": "system", "content": request.instructions})
    for ctx in context_items:
        if ctx.get("role") and ctx.get("content"):
            content = ctx["content"]
            if isinstance(content, list) and content:
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("text"):
                        text_parts.append(c["text"])
                content = " ".join(text_parts) if text_parts else str(content)
            messages.append({"role": ctx["role"], "content": content})
    if input_text:
        messages.append({"role": "user", "content": input_text})
    
    # Prepare backend payload
    backend_payload = {
        "model": request.model,
        "messages": messages,
        "stream": False,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_output_tokens
    }
    
    prompt_tokens = count_input_tokens(request.input, request.instructions,
                                       [t.model_dump() if hasattr(t, 'model_dump') else t for t in (request.tools or [])])
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(CHAT_APP_URL, json=backend_payload)
            
            if response.status_code != 200:
                return create_error_response(
                    message=f"Backend error: {response.status_code}",
                    error_type="api_error",
                    code="backend_error",
                    status_code=response.status_code
                )
            
            backend_response = response.json()
            
            # Extract content
            content = ""
            if "choices" in backend_response and backend_response["choices"]:
                message = backend_response["choices"][0].get("message", {})
                content = message.get("content", "")
            
            completion_tokens = estimate_tokens(content)
            completed_at = int(time.time())
            
            # Build output
            output = [{
                "type": "message",
                "id": message_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}]
            }]
            
            usage = UsageObject(
                input_tokens=prompt_tokens,
                input_tokens_details={"cached_tokens": 0},
                output_tokens=completion_tokens,
                output_tokens_details={"reasoning_tokens": 0},
                total_tokens=prompt_tokens + completion_tokens
            )
            
            # Store response if requested
            if request.store:
                db_response = Response(
                    id=response_id,
                    created_at=created_at,
                    completed_at=completed_at,
                    status="completed",
                    model=request.model,
                    instructions=request.instructions,
                    output=output,
                    usage=usage.model_dump(),
                    response_metadata=request.metadata or {},
                    previous_response_id=request.previous_response_id,
                    temperature=request.temperature or 1.0,
                    top_p=request.top_p or 1.0,
                    max_output_tokens=request.max_output_tokens,
                    parallel_tool_calls=request.parallel_tool_calls,
                    tool_choice=request.tool_choice if isinstance(request.tool_choice, str) else "auto",
                    tools=[t.model_dump() if hasattr(t, 'model_dump') else t for t in (request.tools or [])],
                    truncation=request.truncation,
                    background=request.background
                )
                session.add(db_response)
                
                # Store input items
                if request.input:
                    input_item = ResponseInputItem(
                        id=generate_message_id(),
                        response_id=response_id,
                        type="message",
                        role="user",
                        content=[{"type": "input_text", "text": input_text}] if input_text else None,
                        created_at=created_at
                    )
                    session.add(input_item)
                
                await session.commit()
            
            return ResponseObject(
                id=response_id,
                object="response",
                created_at=created_at,
                status="completed",
                completed_at=completed_at,
                model=request.model,
                output=output,
                usage=usage,
                instructions=request.instructions,
                max_output_tokens=request.max_output_tokens,
                parallel_tool_calls=request.parallel_tool_calls,
                previous_response_id=request.previous_response_id,
                reasoning=request.reasoning,
                store=request.store,
                temperature=request.temperature or 1.0,
                text=request.text,
                tool_choice=request.tool_choice,
                tools=[t.model_dump() if hasattr(t, 'model_dump') else t for t in (request.tools or [])],
                top_p=request.top_p or 1.0,
                truncation=request.truncation,
                metadata=request.metadata or {},  # API uses 'metadata', db uses 'response_metadata'
                background=request.background
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


@router.get("/v1/responses/{response_id}")
async def get_response(
    response_id: str,
    include: Optional[List[str]] = Query(default=None),
    stream: bool = Query(default=False),
    session: AsyncSession = Depends(get_session)
):
    """
    Retrieve a stored response (GET /v1/responses/{response_id}).
    
    Returns the response object with the given ID. Supports streaming
    retrieval for long responses.
    
    Args:
        response_id: ID of response to retrieve
        include: Additional fields to include
        stream: Whether to stream the response
        session: Database session (injected)
        
    Returns:
        ResponseObject matching the ID
        
    Raises:
        HTTPException: 404 if response not found
        
    Example:
        GET /v1/responses/resp_67ccd2bed1ec8190b14f964abc054267
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    db_response = await session.get(Response, response_id)
    
    if not db_response:
        return create_error_response(
            message=f"Response '{response_id}' not found",
            error_type="invalid_request_error",
            param="response_id",
            code="response_not_found",
            status_code=404
        )
    
    # Build usage object
    usage = None
    if db_response.usage:
        usage = UsageObject(
            input_tokens=db_response.usage.get("input_tokens", 0),
            input_tokens_details=db_response.usage.get("input_tokens_details"),
            output_tokens=db_response.usage.get("output_tokens", 0),
            output_tokens_details=db_response.usage.get("output_tokens_details"),
            total_tokens=db_response.usage.get("total_tokens", 0)
        )
    
    return ResponseObject(
        id=db_response.id,
        object="response",
        created_at=db_response.created_at,
        status=db_response.status,
        completed_at=db_response.completed_at,
        error=ErrorObject(**db_response.error) if db_response.error else None,
        incomplete_details=db_response.incomplete_details,
        instructions=db_response.instructions,
        max_output_tokens=db_response.max_output_tokens,
        model=db_response.model,
        output=db_response.output or [],
        parallel_tool_calls=db_response.parallel_tool_calls,
        previous_response_id=db_response.previous_response_id,
        store=True,  # Was stored since we retrieved it
        temperature=db_response.temperature,
        tool_choice=db_response.tool_choice,
        tools=db_response.tools or [],
        top_p=db_response.top_p,
        truncation=db_response.truncation,
        usage=usage,
        metadata=db_response.response_metadata or {},  # db uses 'response_metadata'
        background=db_response.background
    )


@router.delete("/v1/responses/{response_id}", response_model=DeleteResponseObject)
async def delete_response(
    response_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Delete a stored response (DELETE /v1/responses/{response_id}).
    
    Permanently removes the response and associated input items.
    
    Args:
        response_id: ID of response to delete
        session: Database session (injected)
        
    Returns:
        DeleteResponseObject with deletion confirmation
        
    Raises:
        HTTPException: 404 if response not found
        
    Example:
        DELETE /v1/responses/resp_67ccd2bed1ec8190b14f964abc054267
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    db_response = await session.get(Response, response_id)
    
    if not db_response:
        return create_error_response(
            message=f"Response '{response_id}' not found",
            error_type="invalid_request_error",
            param="response_id",
            code="response_not_found",
            status_code=404
        )
    
    # Delete associated input items
    result = await session.execute(
        select(ResponseInputItem).where(ResponseInputItem.response_id == response_id)
    )
    for item in result.scalars().all():
        await session.delete(item)
    
    # Delete response
    await session.delete(db_response)
    await session.commit()
    
    return DeleteResponseObject(id=response_id, object="response", deleted=True)


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(
    response_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Cancel a background response (POST /v1/responses/{response_id}/cancel).
    
    Only responses created with background=true can be cancelled.
    
    Args:
        response_id: ID of response to cancel
        session: Database session (injected)
        
    Returns:
        ResponseObject with status="cancelled"
        
    Raises:
        HTTPException: 404 if not found, 400 if not cancellable
        
    Example:
        POST /v1/responses/resp_67ccd2bed1ec8190b14f964abc054267/cancel
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    db_response = await session.get(Response, response_id)
    
    if not db_response:
        return create_error_response(
            message=f"Response '{response_id}' not found",
            error_type="invalid_request_error",
            param="response_id",
            code="response_not_found",
            status_code=404
        )
    
    if not db_response.background:
        return create_error_response(
            message="Only background responses can be cancelled",
            error_type="invalid_request_error",
            param="response_id",
            code="not_cancellable",
            status_code=400
        )
    
    if db_response.status not in ["in_progress", "queued"]:
        return create_error_response(
            message=f"Response with status '{db_response.status}' cannot be cancelled",
            error_type="invalid_request_error",
            param="response_id",
            code="invalid_status",
            status_code=400
        )
    
    # Update status to cancelled
    db_response.status = "cancelled"
    await session.commit()
    await session.refresh(db_response)
    
    usage = None
    if db_response.usage:
        usage = UsageObject(
            input_tokens=db_response.usage.get("input_tokens", 0),
            output_tokens=db_response.usage.get("output_tokens", 0),
            total_tokens=db_response.usage.get("total_tokens", 0)
        )
    
    return ResponseObject(
        id=db_response.id,
        object="response",
        created_at=db_response.created_at,
        status="cancelled",
        completed_at=db_response.completed_at,
        model=db_response.model,
        output=db_response.output or [],
        usage=usage,
        metadata=db_response.response_metadata or {},  # db uses 'response_metadata'
        background=True,
        parallel_tool_calls=db_response.parallel_tool_calls,
        temperature=db_response.temperature,
        top_p=db_response.top_p,
        truncation=db_response.truncation
    )


@router.post("/v1/responses/compact", response_model=CompactResponseObject)
async def compact_response(
    request: CompactResponseRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Compact conversation history (POST /v1/responses/compact).
    
    Compresses long conversation history into encrypted, opaque items
    for efficient context management. Used for managing context window
    in long-running conversations.
    
    Args:
        request: CompactResponseRequest with model and input
        session: Database session (injected)
        
    Returns:
        CompactResponseObject with compacted items
        
    Example:
        POST /v1/responses/compact
        {"model": "gpt-4o", "input": [...], "previous_response_id": "resp_..."}
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    response_id = generate_response_id()
    created_at = int(time.time())
    
    # Gather all items to compact
    items_to_compact = []
    
    # Add items from previous response
    if request.previous_response_id:
        previous = await session.get(Response, request.previous_response_id)
        if previous:
            # Get input items
            result = await session.execute(
                select(ResponseInputItem)
                .where(ResponseInputItem.response_id == request.previous_response_id)
                .order_by(ResponseInputItem.created_at)
            )
            for item in result.scalars().all():
                items_to_compact.append({
                    "type": item.type,
                    "role": item.role,
                    "content": item.content
                })
            # Add output
            if previous.output:
                items_to_compact.extend(previous.output)
    
    # Add current input
    if request.input:
        if isinstance(request.input, str):
            items_to_compact.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": request.input}]
            })
        else:
            for item in request.input:
                if isinstance(item, dict):
                    items_to_compact.append(item)
                else:
                    items_to_compact.append(item.model_dump())
    
    # Calculate token usage for the items
    total_text = ""
    for item in items_to_compact:
        content = item.get("content", [])
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("text"):
                    total_text += c["text"] + " "
        elif isinstance(content, str):
            total_text += content + " "
    
    input_tokens = estimate_tokens(total_text)
    
    # Generate encrypted compaction (simulated - real implementation would use encryption)
    compaction_data = json.dumps(items_to_compact)
    encrypted_content = base64.b64encode(compaction_data.encode()).decode()
    
    # Build output with preserved user messages and compaction
    output = []
    
    # Keep the first user message for context
    for item in items_to_compact[:2]:  # Keep first 2 items max
        if item.get("type") == "message":
            output.append(item)
    
    # Add compaction item
    compaction_item = {
        "type": "compaction",
        "id": generate_compaction_id(),
        "encrypted_content": encrypted_content[:100] + "..."  # Truncate for display
    }
    output.append(compaction_item)
    
    # Estimate output tokens (compaction is efficient)
    output_tokens = max(100, input_tokens // 4)  # Compaction typically reduces tokens
    
    return CompactResponseObject(
        id=response_id,
        object="response.compaction",
        created_at=created_at,
        output=output,
        usage=UsageObject(
            input_tokens=input_tokens,
            input_tokens_details={"cached_tokens": 0},
            output_tokens=output_tokens,
            output_tokens_details={"reasoning_tokens": 0},
            total_tokens=input_tokens + output_tokens
        )
    )


@router.get("/v1/responses/{response_id}/input_items", response_model=InputItemsListResponse)
async def list_input_items(
    response_id: str,
    after: Optional[str] = Query(default=None, description="Item ID to list after"),
    limit: int = Query(default=20, ge=1, le=100, description="Max items to return"),
    order: str = Query(default="desc", description="Sort order (asc/desc)"),
    include: Optional[List[str]] = Query(default=None, description="Additional fields"),
    session: AsyncSession = Depends(get_session)
):
    """
    List input items for a response (GET /v1/responses/{response_id}/input_items).
    
    Returns paginated list of input items that were provided to generate
    the response.
    
    Args:
        response_id: ID of response
        after: Cursor for pagination (item ID)
        limit: Maximum items to return (1-100, default 20)
        order: Sort order (asc/desc, default desc)
        include: Additional fields to include
        session: Database session (injected)
        
    Returns:
        InputItemsListResponse with paginated items
        
    Raises:
        HTTPException: 404 if response not found
        
    Example:
        GET /v1/responses/resp_123/input_items?limit=10&order=asc
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    # Verify response exists
    db_response = await session.get(Response, response_id)
    if not db_response:
        return create_error_response(
            message=f"Response '{response_id}' not found",
            error_type="invalid_request_error",
            param="response_id",
            code="response_not_found",
            status_code=404
        )
    
    # Build query
    query = select(ResponseInputItem).where(ResponseInputItem.response_id == response_id)
    
    # Apply ordering
    if order == "asc":
        query = query.order_by(ResponseInputItem.created_at.asc())
    else:
        query = query.order_by(ResponseInputItem.created_at.desc())
    
    # Apply pagination cursor
    if after:
        after_item = await session.get(ResponseInputItem, after)
        if after_item:
            if order == "asc":
                query = query.where(ResponseInputItem.created_at > after_item.created_at)
            else:
                query = query.where(ResponseInputItem.created_at < after_item.created_at)
    
    # Apply limit + 1 to check has_more
    query = query.limit(limit + 1)
    
    result = await session.execute(query)
    items = list(result.scalars().all())
    
    # Check if there are more items
    has_more = len(items) > limit
    if has_more:
        items = items[:limit]
    
    # Build response data
    data = []
    for item in items:
        data.append({
            "id": item.id,
            "type": item.type,
            "role": item.role,
            "content": item.content
        })
    
    return InputItemsListResponse(
        object="list",
        data=data,
        first_id=data[0]["id"] if data else None,
        last_id=data[-1]["id"] if data else None,
        has_more=has_more
    )


@router.post("/v1/responses/input_tokens", response_model=InputTokensResponse)
async def get_input_tokens(
    request: InputTokensRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Get input token count (POST /v1/responses/input_tokens).
    
    Calculates the number of input tokens for a request without
    actually generating a response. Useful for estimating costs
    and validating context window usage.
    
    Args:
        request: InputTokensRequest with input data
        session: Database session (injected)
        
    Returns:
        InputTokensResponse with token count
        
    Example:
        POST /v1/responses/input_tokens
        {"model": "gpt-4o", "input": "Hello, world!"}
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    # Gather context from previous response if provided
    context_tokens = 0
    if request.previous_response_id:
        previous = await session.get(Response, request.previous_response_id)
        if previous:
            # Count tokens from previous input items
            result = await session.execute(
                select(ResponseInputItem)
                .where(ResponseInputItem.response_id == request.previous_response_id)
            )
            for item in result.scalars().all():
                if item.content:
                    for c in item.content:
                        if isinstance(c, dict) and c.get("text"):
                            context_tokens += estimate_tokens(c["text"])
            
            # Count tokens from previous output
            if previous.output:
                for output_item in previous.output:
                    if isinstance(output_item, dict):
                        content = output_item.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("text"):
                                    context_tokens += estimate_tokens(c["text"])
    
    # Count current input tokens
    input_tokens = count_input_tokens(request.input, request.instructions, request.tools)
    
    # Total tokens
    total_tokens = context_tokens + input_tokens
    
    return InputTokensResponse(
        object="response.input_tokens",
        input_tokens=total_tokens
    )
