"""
Chat API - OpenAI-Compatible BFF Service.

This service provides an OpenAI-compatible API layer that acts as a BFF
(Backend for Frontend) for the Fast-Chat system. It implements the standard
OpenAI API endpoints for chat completions, files, models, and the Responses API.

Endpoints:
    Chat Completions API:
        - POST /v1/chat/completions - Create chat completion (streaming/non-streaming)
    
    Models API:
        - GET /v1/models - List available models
        - GET /v1/models/{model} - Retrieve model info
    
    Files API:
        - POST /v1/files - Upload file
        - GET /v1/files - List files
        - GET /v1/files/{file_id} - Retrieve file metadata
        - DELETE /v1/files/{file_id} - Delete file
    
    Responses API (2026 Specification):
        - POST /v1/responses - Create a model response
        - GET /v1/responses/{response_id} - Retrieve a stored response
        - DELETE /v1/responses/{response_id} - Delete a response
        - POST /v1/responses/{response_id}/cancel - Cancel background response
        - POST /v1/responses/compact - Compact conversation history
        - GET /v1/responses/{response_id}/input_items - List input items
        - POST /v1/responses/input_tokens - Get input token counts
    
    Search API:
        - POST /v1/search - Semantic search over document collections
    
    Health:
        - GET /health - Health check

Last Grunted: 02/04/2026 05:30:00 PM UTC
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Callable

import structlog
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.db.engine import init_db, close_db, check_db_health
from src.services.http_client import close_client
from src.routers import admin, chat, files, models, responses, search
from src.services.observability import get_metric_snapshot, get_audit_events


# ============================================================================
# Logging Configuration
# ============================================================================

def configure_logging() -> None:
    """
    Configure structured logging with structlog.
    
    Sets up structlog with JSON output for production and pretty printing
    for development (when LOG_FORMAT=console).
    
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO),
    )
    
    # Shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_format == "console":
        # Development: pretty console output
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True)
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Production: JSON output
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


# Initialize logging before creating logger
configure_logging()
logger = structlog.get_logger("chat-api")


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    
    Startup:
        - Initializes database connection pool
        - Initializes HTTP client for backend communication
    
    Shutdown:
        - Closes HTTP client connections
        - Closes database connections
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None - Control returns to the application
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Startup
    logger.info("chat_api.startup")
    
    try:
        logger.info("chat_api.database.init")
        await init_db()
        logger.info("chat_api.database.ready")
    except Exception as e:
        logger.error("chat_api.database.error", error=str(e))
        raise
    
    logger.info("chat_api.ready")
    
    yield
    
    # Shutdown
    logger.info("chat_api.shutdown")
    
    await close_client()
    await close_db()
    
    logger.info("chat_api.shutdown.complete")


# ============================================================================
# Application Instance
# ============================================================================

app = FastAPI(
    title="Chat API",
    description="OpenAI-compatible API for Fast-Chat",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ============================================================================
# CORS Middleware
# ============================================================================

# Allowed origins -- defaults to the Chainlit UI on port 8080.
# Override via CORS_ORIGINS env var (comma-separated) for production.
_cors_origins: list[str] = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Response-Time"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors with OpenAI-style responses.
    
    Args:
        request: The incoming HTTP request
        exc: The validation error
        
    Returns:
        JSONResponse with OpenAI error format (400 status)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        loc = first_error.get("loc", [])
        param = ".".join(str(l) for l in loc if l != "body")
        message = first_error.get("msg", "Validation error")
    else:
        param = None
        message = "Request validation failed"
    
    logger.warning(
        "chat_api.validation_error",
        path=request.url.path,
        param=param,
        message=message,
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": "validation_error"
            }
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """
    Handle HTTPException with OpenAI-style responses.
    
    Preserves structured error details if provided in exc.detail.
    
    Args:
        request: The incoming HTTP request
        exc: The HTTP exception
        
    Returns:
        JSONResponse with OpenAI error format
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Check if detail is already structured
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    
    # Map HTTP status to error type
    error_type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }
    error_type = error_type_map.get(exc.status_code, "api_error")
    
    logger.warning(
        "chat_api.http_error",
        path=request.url.path,
        status_code=exc.status_code,
        detail=str(exc.detail),
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": error_type,
                "param": None,
                "code": None
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Global exception handler for unhandled errors.
    
    Logs the full exception and returns a safe OpenAI-style error response
    without leaking sensitive information.
    
    Args:
        request: The incoming HTTP request
        exc: The exception that was raised
        
    Returns:
        JSONResponse with OpenAI-style error format (500 status)
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    # Log the full error for debugging
    logger.exception(
        "chat_api.unhandled_error",
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    
    # Return safe error response (don't leak internal details in production)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An internal server error occurred",
                "type": "api_error",
                "param": None,
                "code": "internal_error"
            }
        }
    )


# ============================================================================
# Request Logging Middleware
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next: Callable):
    """
    Middleware to log all requests with timing information.
    
    Logs request start and completion with duration for monitoring
    and debugging.
    
    Args:
        request: The incoming HTTP request
        call_next: Next middleware/handler in chain
        
    Returns:
        Response from the handler
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    request_id = request.headers.get("X-Request-ID", "-")
    start_time = time.perf_counter()
    
    # Bind request context for all logs in this request
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    
    logger.info("chat_api.request.start")
    
    response = await call_next(request)
    
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    logger.info(
        "chat_api.request.complete",
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    
    # Add timing header for debugging
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    
    return response


# ============================================================================
# Routers
# ============================================================================

# Mount routers with OpenAI-compatible paths (no prefix duplication)
app.include_router(chat.router, tags=["chat"])
app.include_router(files.router, tags=["files"])
app.include_router(models.router, tags=["models"])
app.include_router(responses.router, tags=["responses"])
app.include_router(search.router, tags=["search"])
app.include_router(admin.router, tags=["admin"])


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring.
    
    Used by load balancers and orchestration systems to verify
    service availability.
    
    Returns:
        dict: Status information including service name and version
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return {
        "status": "ok",
        "service": "chat-api",
        "version": "0.1.0",
    }


@app.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes.

    Indicates whether the service is ready to accept traffic.
    Performs a real database connectivity check (``SELECT 1``).

    Returns:
        dict: Readiness status with component health.
        JSONResponse 503 if the database is unreachable.

    Last Grunted: 02/05/2026 12:00:00 PM UTC
    """
    db_ok = await check_db_health()

    if not db_ok:
        logger.warning("readiness_check.database_unhealthy")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {"database": "unreachable"},
            },
        )

    return {
        "status": "ready",
        "checks": {
            "database": "ok",
        },
    }


@app.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes.
    
    Simple check that the service is running and responsive.
    
    Returns:
        dict: Liveness status
        
    Last Grunted: 02/04/2026 05:30:00 PM UTC
    """
    return {"status": "alive"}


@app.get("/internal/metrics")
async def internal_metrics() -> dict:
    """Internal SLO metrics snapshot."""
    return {"metrics": get_metric_snapshot()}


@app.get("/internal/audit")
async def internal_audit(limit: int = 100) -> dict:
    """Internal audit event buffer snapshot."""
    return {"events": get_audit_events(limit=limit)}
