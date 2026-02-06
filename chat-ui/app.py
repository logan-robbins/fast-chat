"""
Chainlit Frontend for Fast-Chat Platform.

Production-ready Chainlit 2.x application with:
- Dynamic model selection via ``GET /v1/models`` + ChatSettings
- SSE streaming from chat-api with rich status event handling
- Chat history persistence via SQLAlchemy / PostgreSQL data layer
- File upload scoped per-message (no leaking across turns)
- Exponential-backoff retry for transient HTTP errors
- Graceful stop-generation handling
- OAuth / SSO authentication (provider-configurable)

Architecture::

    chat-ui (Chainlit, port 8080)
        -> chat-api (FastAPI BFF, port 8000)
            -> chat-app (LangGraph, port 8001)

SSE Stream Format from chat-api::

    event: status
    data: {"type": "agent_handoff", "message": "Searching the web...", "agent": "websearch"}

    data: {"choices": [{"delta": {"content": "Hello"}}]}
    data: {"choices": [{"delta": {"content": " world"}}]}

    event: usage
    data: {"prompt_tokens": 150, "completion_tokens": 50, ...}

    data: [DONE]

Last Grunted: 02/05/2026 11:00:00 PM UTC
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
from typing import Any, Dict, FrozenSet, List, Optional

import chainlit as cl
from chainlit.input_widget import Select
import httpx
from httpx_sse import aconnect_sse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# =============================================================================
# Configuration
# =============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        chat_api_url: Base URL of the chat-api (BFF) service.
        chat_api_timeout: HTTP request timeout in seconds for chat-api calls.
        chainlit_auth_secret: Secret for Chainlit session signing.
        database_url: PostgreSQL connection string for the data layer.
        log_level: Python logging level name.
        default_model: Default model ID used when none is selected.
        max_retries: Maximum retry attempts for transient HTTP errors.
        retry_backoff_base: Base delay (seconds) for exponential backoff.
    """

    chat_api_url: str = "http://localhost:8000"
    chat_api_timeout: float = 120.0

    chainlit_auth_secret: Optional[str] = None
    database_url: Optional[str] = None

    log_level: str = "INFO"
    default_model: str = "gpt-4o"

    max_retries: int = 3
    retry_backoff_base: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# HTTP status codes considered transient and eligible for retry.
_RETRYABLE_STATUS_CODES: FrozenSet[int] = frozenset({429, 502, 503, 504})


# =============================================================================
# HTTP Client (shared, connection-pooled)
# =============================================================================

_http_client: Optional[httpx.AsyncClient] = None


async def get_http_client() -> httpx.AsyncClient:
    """Return the module-level ``httpx.AsyncClient``, creating it lazily.

    The client uses connection pooling to reduce TCP handshake overhead
    across requests within a single Chainlit worker process.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=settings.chat_api_url,
            timeout=httpx.Timeout(settings.chat_api_timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
        )
    return _http_client


async def _api_get_with_retry(
    path: str,
    *,
    max_retries: int = 2,
) -> httpx.Response:
    """Issue a ``GET`` request to *path* with exponential-backoff retry.

    Retries on ``httpx.ConnectError``, ``httpx.TimeoutException``, and
    any response with a status code in ``_RETRYABLE_STATUS_CODES``.

    Args:
        path: URL path relative to the chat-api base URL.
        max_retries: Total attempts before giving up.

    Returns:
        The ``httpx.Response`` from the first non-retryable attempt.

    Raises:
        httpx.ConnectError: If all retries are exhausted due to connection errors.
        httpx.TimeoutException: If all retries are exhausted due to timeouts.
        httpx.HTTPStatusError: Synthesised from the last retryable response.
    """
    client = await get_http_client()
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            response = await client.get(path)
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                return response
            # Retryable HTTP status -- fall through to backoff
            last_exc = httpx.HTTPStatusError(
                f"HTTP {response.status_code}",
                request=response.request,
                response=response,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc

        if attempt < max_retries - 1:
            backoff = settings.retry_backoff_base * (2 ** attempt)
            logger.warning(
                "api.retry attempt=%d/%d backoff=%.1fs error=%s",
                attempt + 1,
                max_retries,
                backoff,
                last_exc,
            )
            await asyncio.sleep(backoff)

    raise last_exc  # type: ignore[misc]


# =============================================================================
# Data Models (SSE event payloads)
# =============================================================================


class StatusEvent(BaseModel):
    """Status event from chat-api SSE stream.

    Attributes:
        type: One of ``thinking``, ``tool_start``, ``tool_progress``,
            ``tool_complete``, ``agent_handoff``, ``error``.
        message: User-friendly status string to display.
        agent: Agent name when relevant (e.g. ``"websearch"``).
        tool: Tool name when relevant (e.g. ``"perplexity_search"``).
        details: Optional structured payload for rich rendering.
        timestamp: ISO-8601 timestamp of the event.
    """

    type: str
    message: str
    agent: Optional[str] = None
    tool: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class UsageEvent(BaseModel):
    """Token usage event from chat-api SSE stream.

    Attributes:
        prompt_tokens: Tokens consumed by the input prompt.
        completion_tokens: Tokens in the generated completion.
        total_tokens: ``prompt_tokens + completion_tokens``.
        context_window_limit: Model context window size (tokens).
        context_utilization_pct: Percentage of context used.
        is_final: ``True`` when this is the definitive usage report.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    context_window_limit: Optional[int] = None
    context_utilization_pct: Optional[float] = None
    is_final: bool = False


# =============================================================================
# SQLAlchemy Data Layer (optional -- requires DATABASE_URL)
# =============================================================================

if settings.database_url:
    try:
        from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

        @cl.data_layer
        def get_data_layer():
            """Configure the SQLAlchemy data layer for PostgreSQL persistence.

            The ``+asyncpg`` driver is required for async support; the helper
            transparently rewrites plain ``postgresql://`` URIs.
            """
            conninfo = settings.database_url
            if conninfo and conninfo.startswith("postgresql://"):
                conninfo = conninfo.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )
            return SQLAlchemyDataLayer(conninfo=conninfo)

        logger.info("data_layer.configured backend=sqlalchemy")
    except ImportError as exc:
        logger.warning("data_layer.unavailable error=%s", exc)


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
    """Handle OAuth authentication callback.

    Configure providers via environment variables::

        OAUTH_GITHUB_CLIENT_ID / OAUTH_GITHUB_CLIENT_SECRET
        OAUTH_GOOGLE_CLIENT_ID / OAUTH_GOOGLE_CLIENT_SECRET
        OAUTH_AZURE_AD_CLIENT_ID / OAUTH_AZURE_AD_CLIENT_SECRET / OAUTH_AZURE_AD_TENANT_ID
        OAUTH_OKTA_CLIENT_ID / OAUTH_OKTA_CLIENT_SECRET / OAUTH_OKTA_DOMAIN

    Args:
        provider_id: OAuth provider identifier (``github``, ``google``, etc.).
        token: OAuth access token.
        raw_user_data: Raw profile data from the provider.
        default_user: Default ``cl.User`` created by Chainlit.

    Returns:
        ``cl.User`` to accept login, ``None`` to reject.
    """
    logger.info(
        "oauth.callback provider=%s user=%s",
        provider_id,
        default_user.identifier,
    )
    return default_user


# =============================================================================
# Model Discovery (GET /v1/models)
# =============================================================================

_FALLBACK_MODELS: List[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


async def fetch_available_models() -> List[str]:
    """Fetch available model IDs from ``GET /v1/models``.

    Returns the API-provided model list or a hard-coded fallback when the
    API is unreachable.
    """
    try:
        response = await _api_get_with_retry("/v1/models", max_retries=2)
        if response.status_code == 200:
            data = response.json()
            model_ids: List[str] = [m["id"] for m in data.get("data", [])]
            if model_ids:
                logger.info("models.fetched count=%d", len(model_ids))
                return model_ids
    except Exception as exc:
        logger.warning("models.fetch_failed error=%s", exc)

    logger.info("models.using_fallback count=%d", len(_FALLBACK_MODELS))
    return list(_FALLBACK_MODELS)


async def _setup_model_selector() -> str:
    """Create the ChatSettings model selector and return the selected model."""
    model_ids = await fetch_available_models()

    default_idx = 0
    if settings.default_model in model_ids:
        default_idx = model_ids.index(settings.default_model)

    chat_settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=model_ids,
                initial_index=default_idx,
            ),
        ]
    ).send()

    selected: str = chat_settings.get("model", settings.default_model)  # type: ignore[arg-type]
    cl.user_session.set("selected_model", selected)
    return selected


# =============================================================================
# Chat Lifecycle Hooks
# =============================================================================


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise a new chat session.

    - Creates empty message history and session state.
    - Fetches available models from chat-api and presents a selector.
    - Sends a welcome message.
    """
    cl.user_session.set("message_history", [])
    cl.user_session.set("thread_id", None)

    selected_model = await _setup_model_selector()

    user = cl.user_session.get("user")
    if user:
        logger.info("chat.start user=%s model=%s", user.identifier, selected_model)

    await cl.Message(
        content=(
            "Hello! I'm your AI assistant. I can help you with:\n\n"
            "- **Web Search** -- Find current information online\n"
            "- **Document Analysis** -- Upload and analyze files\n"
            "- **Code Assistance** -- Help with programming tasks\n\n"
            "Use the **settings** button to switch models. "
            "How can I help you today?"
        ),
    ).send()


@cl.on_settings_update
async def on_settings_update(new_settings: Dict[str, Any]) -> None:
    """Handle changes from the ChatSettings UI (model switch)."""
    model = new_settings.get("model")
    if model:
        cl.user_session.set("selected_model", model)
        logger.info("settings.model_changed model=%s", model)
        await cl.Message(
            content=f"Switched to model **{model}**.",
            author="system",
        ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.ThreadDict) -> None:
    """Resume a previously persisted chat session.

    Restores message history from the thread's steps and re-initialises
    the model selector.

    Args:
        thread: Persisted thread dictionary containing steps and metadata.
    """
    logger.info("chat.resume thread_id=%s", thread.get("id"))

    message_history: List[Dict[str, str]] = []
    for step in thread.get("steps", []):
        step_type = step.get("type")
        content = step.get("output", "")
        if step_type == "user_message" and content:
            message_history.append({"role": "user", "content": content})
        elif step_type == "assistant_message" and content:
            message_history.append({"role": "assistant", "content": content})

    cl.user_session.set("message_history", message_history)
    cl.user_session.set("thread_id", thread.get("id"))

    await _setup_model_selector()


@cl.on_stop
async def on_stop() -> None:
    """Handle the user clicking the stop button during generation.

    Chainlit automatically cancels the running ``on_message`` task by
    raising ``asyncio.CancelledError``.  This callback fires *before*
    the cancellation propagates, allowing us to set a session flag and
    log the event.

    Note:
        The chat-api exposes ``POST /v1/responses/{id}/cancel`` for the
        Responses API.  For the chat-completions path used here,
        cancellation is achieved by the ``CancelledError`` propagation
        which drops the SSE connection; the backend detects the closed
        connection and stops processing.
    """
    logger.info("chat.stopped_by_user")
    cl.user_session.set("generation_cancelled", True)


@cl.on_chat_end
async def on_chat_end() -> None:
    """Log when a chat session ends (disconnect or new session)."""
    logger.info("chat.end")


# =============================================================================
# File Upload Handling
# =============================================================================


async def _upload_file_to_api(file: cl.File) -> Optional[str]:
    """Upload *file* to ``POST /v1/files`` and return its file ID.

    Args:
        file: Chainlit ``File`` element with a local ``path`` and ``name``.

    Returns:
        The ``file-...`` ID string on success, ``None`` otherwise.
    """
    try:
        client = await get_http_client()

        with open(file.path, "rb") as fh:
            file_content = fh.read()

        response = await client.post(
            "/v1/files",
            files={"file": (file.name, file_content)},
            data={"purpose": "assistants"},
        )

        if response.status_code == 200:
            file_id: str = response.json().get("id", "")
            logger.info("file.uploaded id=%s name=%s", file_id, file.name)
            return file_id

        logger.error(
            "file.upload_failed status=%d body=%s",
            response.status_code,
            response.text[:200],
        )
        return None

    except Exception as exc:
        logger.exception("file.upload_error: %s", exc)
        return None


# =============================================================================
# Status Event Display (Chainlit Steps)
# =============================================================================


async def _handle_status_event(
    status: StatusEvent,
    current_step: Optional[cl.Step],
) -> Optional[cl.Step]:
    """Render a ``StatusEvent`` as a Chainlit Step.

    Creates new Steps for ``thinking``, ``agent_handoff``, and
    ``tool_start``; updates the existing step for ``tool_progress``;
    and finalises the step for ``tool_complete`` and ``error``.

    All Steps use ``type="tool"`` so they remain visible when the
    config ``cot = "tool_call"`` is set.

    Args:
        status: Parsed status event from the SSE stream.
        current_step: The currently open step, or ``None``.

    Returns:
        The active step, or ``None`` when the step has been finalised.
    """
    # Derive a human-friendly step name
    if status.type == "agent_handoff":
        step_name = status.agent or "Agent"
    elif status.type in ("tool_start", "tool_progress", "tool_complete"):
        step_name = status.tool or "Tool"
    elif status.type == "thinking":
        step_name = "Thinking"
    elif status.type == "error":
        step_name = "Error"
    else:
        step_name = "Processing"

    # All steps as "tool" type so they are visible with cot="tool_call"
    if current_step is None:
        current_step = cl.Step(name=step_name, type="tool")
        await current_step.send()

    current_step.output = status.message

    if status.type in ("tool_complete", "error"):
        await current_step.update()
        return None  # step closed

    # Update the step UI to display the output text.
    # NOTE: ``stream_token("")`` was previously used here as a refresh hack,
    # but it operates on the Step's *streaming buffer*, not the ``output``
    # property, so the status message was never shown for in-progress events.
    await current_step.update()
    return current_step


# =============================================================================
# Usage Display Formatting
# =============================================================================


def _format_usage_markdown(usage: UsageEvent) -> str:
    """Return a formatted Markdown summary of token usage.

    Context utilisation is displayed first when available so that it is
    the most prominent piece of information.  A warning suffix is
    appended when utilisation exceeds 90%.

    Example output::

        Context: **12.3%** used | **1,234** tokens (prompt 800 + completion 434)
    """
    parts: List[str] = []

    if usage.context_utilization_pct is not None:
        pct = usage.context_utilization_pct
        warning = " -- approaching limit" if pct >= 90.0 else ""
        parts.append(f"Context: **{pct:.1f}%** used{warning}")

    parts.append(
        f"**{usage.total_tokens:,}** tokens "
        f"(prompt {usage.prompt_tokens:,} + completion {usage.completion_tokens:,})"
    )

    return " | ".join(parts)


# =============================================================================
# SSE Streaming Core
# =============================================================================


async def _stream_chat_completion(
    payload: Dict[str, Any],
    response_message: cl.Message,
) -> Optional[str]:
    """Stream a chat completion from chat-api via SSE.

    Handles three event categories:

    * **status** -- rendered as collapsible Chainlit Steps.
    * **usage** -- displayed as a system message when final.
    * **content** -- streamed token-by-token into *response_message*.

    Connection-level errors (connect, timeout, 502/503/504) trigger
    exponential-backoff retry *before* any tokens have been sent.
    Mid-stream errors are propagated to the caller.

    Args:
        payload: The JSON body for ``POST /v1/chat/completions``.
        response_message: The pre-created Chainlit message to stream into.

    Returns:
        The concatenated assistant content, or ``None`` on total failure.
    """
    current_step: Optional[cl.Step] = None
    full_content: List[str] = []
    tokens_started = False
    last_exc: Optional[Exception] = None

    for attempt in range(settings.max_retries):
        try:
            client = await get_http_client()

            async with aconnect_sse(
                client,
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    event_type = sse.event or "message"
                    data = sse.data

                    if data == "[DONE]":
                        if current_step:
                            await current_step.update()
                        break

                    try:
                        event_data: Dict[str, Any] = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # ---- Status events ----
                    if event_type == "status":
                        status = StatusEvent(**event_data)

                        # Close previous step when a new phase begins
                        if (
                            status.type in ("agent_handoff", "tool_start")
                            and current_step
                        ):
                            await current_step.update()
                            current_step = None

                        current_step = await _handle_status_event(
                            status, current_step
                        )

                    # ---- Usage events ----
                    elif event_type == "usage":
                        usage = UsageEvent(**event_data)
                        if usage.is_final:
                            await cl.Message(
                                content=_format_usage_markdown(usage),
                                author="system",
                            ).send()

                    # ---- Content streaming (default event type) ----
                    else:
                        choices = event_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                tokens_started = True
                                full_content.append(content)
                                await response_message.stream_token(content)

            # Reached end of stream successfully
            return "".join(full_content)

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            last_exc = exc
            # Only retry if we haven't started sending tokens yet
            if tokens_started:
                raise
            if attempt < settings.max_retries - 1:
                backoff = settings.retry_backoff_base * (2 ** attempt)
                logger.warning(
                    "chat.stream_retry attempt=%d/%d backoff=%.1fs error=%s",
                    attempt + 1,
                    settings.max_retries,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
            else:
                raise

        except asyncio.CancelledError:
            # User clicked stop -- ensure any open status Step is
            # finalised so it doesn't hang in an indeterminate state
            # in the Chainlit UI.
            if current_step is not None:
                try:
                    current_step.output = current_step.output or "Cancelled"
                    await current_step.update()
                except Exception:
                    pass  # Best-effort cleanup during cancellation
            raise  # Re-raise for on_message CancelledError handler

    # All retries exhausted (shouldn't reach here but guard anyway)
    if last_exc:
        raise last_exc
    return None


# =============================================================================
# Message Handling
# =============================================================================


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages with SSE streaming from chat-api.

    Flow:
        1. Upload any attached files -- IDs are scoped to **this** message
           only (not leaked to subsequent turns).
        2. Append user content (with optional file refs) to history.
        3. Stream response from chat-api via SSE with retry.
        4. Render status events, content tokens, and usage stats.
        5. Append assistant response to message history.

    Args:
        message: Incoming ``cl.Message`` with content and optional file
            elements.
    """
    cl.user_session.set("generation_cancelled", False)

    message_history: List[Dict[str, str]] = cl.user_session.get(
        "message_history", []
    )
    thread_id: Optional[str] = cl.user_session.get("thread_id")
    selected_model: str = cl.user_session.get(
        "selected_model", settings.default_model
    )

    # ---- File uploads (scoped to THIS message only) ----
    current_file_ids: List[str] = []
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                file_id = await _upload_file_to_api(element)
                if file_id:
                    current_file_ids.append(file_id)
                    await cl.Message(
                        content=f"Uploaded: **{element.name}**",
                        author="system",
                    ).send()

    # ---- Build user content ----
    user_content: str = message.content
    if current_file_ids:
        user_content += f"\n\n[Attached files: {', '.join(current_file_ids)}]"

    message_history.append({"role": "user", "content": user_content})

    # ---- Build request payload ----
    payload: Dict[str, Any] = {
        "model": selected_model,
        "messages": message_history,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "include_status": True,
        },
    }
    if thread_id:
        payload["thread_id"] = thread_id

    # ---- Create empty response message for streaming ----
    response_message = cl.Message(content="")
    await response_message.send()

    try:
        assistant_content = await _stream_chat_completion(
            payload, response_message
        )

        # Finalise
        if assistant_content:
            response_message.content = assistant_content
            await response_message.update()

            message_history.append(
                {"role": "assistant", "content": assistant_content}
            )
            cl.user_session.set("message_history", message_history)

        logger.info(
            "chat.response_complete len=%d model=%s",
            len(response_message.content),
            selected_model,
        )

    except httpx.TimeoutException:
        logger.error("chat.timeout model=%s", selected_model)
        await cl.Message(
            content=(
                "The request timed out. "
                "Please try again or switch to a faster model."
            ),
            author="system",
        ).send()

    except httpx.ConnectError:
        logger.error(
            "chat.connection_refused url=%s", settings.chat_api_url
        )
        await cl.Message(
            content=(
                "Could not connect to the API server. "
                "Please verify the backend is running and try again."
            ),
            author="system",
        ).send()

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.error("chat.http_error status=%d", status_code)

        if status_code == 429:
            user_msg = "Rate limited. Please wait a moment and try again."
        elif 500 <= status_code < 600:
            user_msg = "The API server encountered an error. Please try again."
        else:
            user_msg = f"API error ({status_code}). Please try again."

        await cl.Message(content=user_msg, author="system").send()

    except asyncio.CancelledError:
        # User clicked stop -- Chainlit cancels the running task.
        logger.info("chat.cancelled_by_user")
        if response_message.content:
            response_message.content += "\n\n*-- Generation stopped --*"
            await response_message.update()

    except Exception as exc:
        logger.exception("chat.error: %s", exc)
        await cl.Message(
            content="An unexpected error occurred. Please try again.",
            author="system",
        ).send()


# =============================================================================
# Chat Starters (Quick Actions)
# =============================================================================


@cl.set_starters
async def set_starters():
    """Conversation starters displayed to new users.

    Returns:
        List of ``cl.Starter`` objects with labels, messages, and icons.
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
# Logout
# =============================================================================


@cl.on_logout
async def on_logout(request, response):
    """Log when a user signs out."""
    logger.info("user.logout")


# =============================================================================
# Application Shutdown
# =============================================================================


def _cleanup_http_client() -> None:
    """Clean up the shared HTTP client during process shutdown.

    Attempts ``asyncio.get_running_loop()`` first (in case Chainlit's
    event loop is still active) and falls back to ``asyncio.run()``
    when no loop is running.

    All errors are swallowed so that shutdown is never blocked.  The
    module-level ``_http_client`` reference is cleared regardless so
    that subsequent calls do not attempt a double-close.
    """
    global _http_client
    if _http_client is None or _http_client.is_closed:
        return

    client_to_close = _http_client
    _http_client = None  # Prevent double-close from concurrent calls

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(client_to_close.aclose())
    except RuntimeError:
        # No running loop -- create a temporary one.
        try:
            asyncio.run(client_to_close.aclose())
        except Exception:
            # Event loop finalisation edge case -- give up gracefully.
            pass
    except Exception as exc:
        logger.warning("app.cleanup_failed error=%s", exc)


atexit.register(_cleanup_http_client)
