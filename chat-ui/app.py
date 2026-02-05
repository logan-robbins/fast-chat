"""
Chainlit Frontend for Fast-Chat Platform.

Production-ready Chainlit application with:
- OAuth/SSO authentication (configurable providers)
- SSE streaming from chat-api with status event handling
- Chat history persistence via SQLAlchemy/PostgreSQL
- File upload support integrated with chat-api
- Modern dark theme with professional styling

Architecture:
    chat-ui (Chainlit) → chat-api (BFF) → chat-app (LangGraph)

SSE Stream Format from chat-api:
    event: status
    data: {"type": "agent_handoff", "message": "Searching the web...", "agent": "websearch"}

    data: {"choices": [{"delta": {"content": "Hello"}}]}
    data: {"choices": [{"delta": {"content": " world"}}]}

    event: usage
    data: {"prompt_tokens": 150, "completion_tokens": 50}

    data: [DONE]

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
import json
import logging
import os
from typing import Dict, List, Optional, Any

import chainlit as cl
import httpx
from httpx_sse import aconnect_sse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# =============================================================================
# Configuration
# =============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Chat API Configuration
    chat_api_url: str = "http://localhost:8000"
    chat_api_timeout: float = 120.0
    
    # Authentication
    chainlit_auth_secret: Optional[str] = None
    
    # Database (for SQLAlchemy data layer)
    database_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    
    # Default model
    default_model: str = "gpt-4o"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# HTTP Client
# =============================================================================

# Global HTTP client with connection pooling
_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Get or create the HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=settings.chat_api_url,
            timeout=httpx.Timeout(settings.chat_api_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http_client


# =============================================================================
# Data Models
# =============================================================================


class StatusEvent(BaseModel):
    """Status event from chat-api SSE stream."""
    type: str  # thinking, tool_start, tool_progress, tool_complete, agent_handoff, error
    message: str
    agent: Optional[str] = None
    tool: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class UsageEvent(BaseModel):
    """Token usage event from chat-api SSE stream."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_window_limit: Optional[int] = None
    context_utilization_pct: Optional[float] = None
    is_final: bool = False


# =============================================================================
# SQLAlchemy Data Layer (Optional - for persistence)
# =============================================================================

if settings.database_url:
    try:
        from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
        
        @cl.data_layer
        def get_data_layer():
            """Configure SQLAlchemy data layer for chat persistence."""
            # Ensure asyncpg is used for async PostgreSQL
            conninfo = settings.database_url
            if conninfo.startswith("postgresql://"):
                conninfo = conninfo.replace("postgresql://", "postgresql+asyncpg://", 1)
            
            return SQLAlchemyDataLayer(conninfo=conninfo)
        
        logger.info("SQLAlchemy data layer configured for persistence")
    except ImportError as e:
        logger.warning(f"SQLAlchemy data layer not available: {e}")


# =============================================================================
# OAuth Authentication
# =============================================================================


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    """
    Handle OAuth authentication callback.
    
    Configure OAuth providers via environment variables:
    - OAUTH_GITHUB_CLIENT_ID, OAUTH_GITHUB_CLIENT_SECRET
    - OAUTH_GOOGLE_CLIENT_ID, OAUTH_GOOGLE_CLIENT_SECRET
    - OAUTH_AZURE_AD_CLIENT_ID, OAUTH_AZURE_AD_CLIENT_SECRET, OAUTH_AZURE_AD_TENANT_ID
    - OAUTH_OKTA_CLIENT_ID, OAUTH_OKTA_CLIENT_SECRET, OAUTH_OKTA_DOMAIN
    
    Args:
        provider_id: OAuth provider identifier (github, google, azure-ad, etc.)
        token: OAuth access token
        raw_user_data: Raw user data from the provider
        default_user: Default user object created by Chainlit
        
    Returns:
        cl.User object if authentication is successful, None to reject
    """
    logger.info(
        "oauth.callback",
        extra={
            "provider": provider_id,
            "user_id": default_user.identifier,
        }
    )
    
    # Add provider-specific validation here if needed
    # Example: Only allow users from specific domains
    # if provider_id == "google":
    #     if raw_user_data.get("hd") != "example.com":
    #         return None
    
    return default_user


# =============================================================================
# Chat Lifecycle Hooks
# =============================================================================


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize a new chat session.
    
    Sets up:
    - Message history storage in user session
    - Welcome message for new users
    - Thread ID tracking for chat-api persistence
    """
    # Initialize message history
    cl.user_session.set("message_history", [])
    cl.user_session.set("thread_id", None)
    cl.user_session.set("file_ids", [])
    
    # Get user info if authenticated
    user = cl.user_session.get("user")
    if user:
        logger.info(f"chat.start user={user.identifier}")
        
    # Send welcome message with starters
    await cl.Message(
        content="Hello! I'm your AI assistant. I can help you with:\n\n"
                "- **Web Search** - Find current information online\n"
                "- **Document Analysis** - Upload and analyze files\n"
                "- **Code Assistance** - Help with programming tasks\n\n"
                "How can I help you today?",
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.ThreadDict):
    """
    Resume a previous chat session.
    
    Restores message history from the persisted thread.
    
    Args:
        thread: Dictionary containing thread data and messages
    """
    logger.info(f"chat.resume thread_id={thread.get('id')}")
    
    # Restore message history from thread
    message_history: List[Dict[str, str]] = []
    
    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            message_history.append({
                "role": "user",
                "content": step.get("output", "")
            })
        elif step.get("type") == "assistant_message":
            message_history.append({
                "role": "assistant", 
                "content": step.get("output", "")
            })
    
    cl.user_session.set("message_history", message_history)
    cl.user_session.set("thread_id", thread.get("id"))
    cl.user_session.set("file_ids", [])


@cl.on_stop
async def on_stop():
    """Handle user clicking the stop button during generation."""
    logger.info("chat.stopped by user")
    # The current streaming will be interrupted by Chainlit


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat session ends."""
    logger.info("chat.end")


# =============================================================================
# File Upload Handling
# =============================================================================


async def upload_file_to_api(file: cl.File) -> Optional[str]:
    """
    Upload a file to chat-api and return the file ID.
    
    Args:
        file: Chainlit File object with path and metadata
        
    Returns:
        File ID string if successful, None otherwise
    """
    try:
        client = await get_http_client()
        
        # Read file content
        with open(file.path, "rb") as f:
            file_content = f.read()
        
        # Upload to chat-api
        files = {"file": (file.name, file_content)}
        data = {"purpose": "assistants"}
        
        response = await client.post("/v1/files", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            file_id = result.get("id")
            logger.info(f"file.uploaded id={file_id} name={file.name}")
            return file_id
        else:
            logger.error(f"file.upload_failed status={response.status_code}")
            return None
            
    except Exception as e:
        logger.exception(f"file.upload_error: {e}")
        return None


# =============================================================================
# Message Handling with SSE Streaming
# =============================================================================


async def handle_status_event(status: StatusEvent, step: Optional[cl.Step] = None) -> cl.Step:
    """
    Handle status events from chat-api and display in UI.
    
    Creates or updates Chainlit Steps to show:
    - Agent handoffs ("Searching the web...")
    - Tool usage ("Looking through your documents...")
    - Progress updates ("Searching collection 2/3...")
    
    Args:
        status: StatusEvent from SSE stream
        step: Existing step to update, or None to create new
        
    Returns:
        The Step object (created or updated)
    """
    # Map status types to step names and styling
    status_config = {
        "thinking": {"name": "Thinking", "type": "llm"},
        "agent_handoff": {"name": status.agent or "Agent", "type": "tool"},
        "tool_start": {"name": status.tool or "Tool", "type": "tool"},
        "tool_progress": {"name": status.tool or "Tool", "type": "tool"},
        "tool_complete": {"name": status.tool or "Tool", "type": "tool"},
        "error": {"name": "Error", "type": "tool"},
    }
    
    config = status_config.get(status.type, {"name": "Processing", "type": "tool"})
    
    if step is None:
        step = cl.Step(name=config["name"], type=config["type"])
        await step.send()
    
    # Update step content
    step.output = status.message
    
    # Mark complete if finished
    if status.type in ("tool_complete", "error"):
        await step.update()
    else:
        await step.stream_token("")  # Trigger UI update
    
    return step


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages with SSE streaming from chat-api.
    
    Flow:
    1. Handle any file uploads
    2. Build message history for context
    3. Stream response from chat-api via SSE
    4. Parse and display status events, content tokens, and usage
    5. Update message history
    
    Args:
        message: Incoming Chainlit Message with content and optional files
    """
    # Get message history from session
    message_history: List[Dict[str, str]] = cl.user_session.get("message_history", [])
    file_ids: List[str] = cl.user_session.get("file_ids", [])
    thread_id: Optional[str] = cl.user_session.get("thread_id")
    
    # Handle file uploads
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_id = await upload_file_to_api(element)
                if file_id:
                    file_ids.append(file_id)
                    await cl.Message(
                        content=f"Uploaded file: **{element.name}**",
                        author="system"
                    ).send()
        
        cl.user_session.set("file_ids", file_ids)
    
    # Add user message to history
    user_content = message.content
    if file_ids:
        # Include file references in the message
        user_content += f"\n\n[Attached files: {', '.join(file_ids)}]"
    
    message_history.append({"role": "user", "content": user_content})
    
    # Build request payload for chat-api
    payload = {
        "model": settings.default_model,
        "messages": message_history,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "include_status": True,
        },
    }
    
    if thread_id:
        payload["thread_id"] = thread_id
    
    # Create response message for streaming
    response_message = cl.Message(content="")
    await response_message.send()
    
    # Track current status step
    current_step: Optional[cl.Step] = None
    full_content: List[str] = []
    current_event_type: Optional[str] = None
    
    try:
        client = await get_http_client()
        
        async with aconnect_sse(
            client, 
            "POST", 
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as event_source:
            async for sse in event_source.aiter_sse():
                # Handle different event types
                event_type = sse.event or "message"
                data = sse.data
                
                if data == "[DONE]":
                    # Stream complete
                    if current_step:
                        await current_step.update()
                    break
                
                try:
                    event_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                
                # Handle status events
                if event_type == "status":
                    status = StatusEvent(**event_data)
                    
                    # Close previous step if starting new one
                    if status.type in ("agent_handoff", "tool_start") and current_step:
                        await current_step.update()
                        current_step = None
                    
                    current_step = await handle_status_event(status, current_step)
                    
                    # Close step on completion
                    if status.type in ("tool_complete", "error"):
                        current_step = None
                
                # Handle usage events
                elif event_type == "usage":
                    usage = UsageEvent(**event_data)
                    if usage.is_final and usage.context_utilization_pct:
                        # Display context utilization in footer
                        await cl.Message(
                            content=f"*Tokens: {usage.total_tokens:,} ({usage.context_utilization_pct:.1f}% context)*",
                            author="system",
                        ).send()
                
                # Handle content streaming (default event type)
                else:
                    choices = event_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        
                        if content:
                            full_content.append(content)
                            await response_message.stream_token(content)
        
        # Finalize message
        response_message.content = "".join(full_content)
        await response_message.update()
        
        # Update message history
        if full_content:
            message_history.append({
                "role": "assistant",
                "content": "".join(full_content)
            })
            cl.user_session.set("message_history", message_history)
        
        logger.info(
            "chat.response_complete",
            extra={"content_length": len(response_message.content)}
        )
        
    except httpx.TimeoutException:
        logger.error("chat.timeout")
        await cl.Message(
            content="Request timed out. Please try again.",
            author="system"
        ).send()
        
    except httpx.HTTPStatusError as e:
        logger.error(f"chat.http_error status={e.response.status_code}")
        await cl.Message(
            content=f"API error: {e.response.status_code}. Please try again.",
            author="system"
        ).send()
        
    except Exception as e:
        logger.exception(f"chat.error: {e}")
        await cl.Message(
            content="An error occurred. Please try again.",
            author="system"
        ).send()


# =============================================================================
# Chat Starters (Quick Actions)
# =============================================================================


@cl.set_starters
async def set_starters():
    """
    Define conversation starters shown to new users.
    
    Returns:
        List of Starter objects with labels, messages, and icons
    """
    return [
        cl.Starter(
            label="Search the web",
            message="Search for the latest news about artificial intelligence",
            icon="/public/icons/search.svg",
        ),
        cl.Starter(
            label="Analyze a document",
            message="I'd like to upload a document for analysis",
            icon="/public/icons/document.svg",
        ),
        cl.Starter(
            label="Help with code",
            message="Can you help me write a Python function?",
            icon="/public/icons/code.svg",
        ),
        cl.Starter(
            label="General question",
            message="What can you help me with?",
            icon="/public/icons/question.svg",
        ),
    ]


# =============================================================================
# Application Shutdown
# =============================================================================


@cl.on_logout
async def on_logout(request, response):
    """Clean up when user logs out."""
    logger.info("user.logout")


# Cleanup HTTP client on shutdown
import atexit

def cleanup():
    """Cleanup resources on shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_http_client.aclose())
            else:
                loop.run_until_complete(_http_client.aclose())
        except Exception:
            pass

atexit.register(cleanup)
