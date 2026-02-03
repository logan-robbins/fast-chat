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
    
    Health:
        - GET /health - Health check

Last Grunted: 02/03/2026 11:45:00 AM UTC
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.db.engine import init_db
from src.routers import chat, files, models, responses
import logging
import time

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events.
    
    Initializes database on startup and performs cleanup on shutdown.
    
    Args:
        app: FastAPI application instance
        
    Yields:
        None - Control returns to the application
        
    Last Grunted: 02/03/2026 11:45:00 AM UTC
    """
    # Startup
    logger.info("Initializing Database...")
    await init_db()
    logger.info("Database initialized.")
    yield
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Chat API",
    description="OpenAI-compatible API for Fast-Chat",
    version="0.1.0",
    lifespan=lifespan
)


# OpenAI-style error handler
@app.exception_handler(Exception)
async def openai_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that returns OpenAI-style error responses.
    
    OpenAI error format:
    {
        "error": {
            "message": "Error description",
            "type": "error_type",
            "param": null,
            "code": "error_code"
        }
    }
    
    Args:
        request: The incoming HTTP request
        exc: The exception that was raised
        
    Returns:
        JSONResponse with OpenAI-style error format
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    error_type = type(exc).__name__
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": error_type,
                "param": None,
                "code": "internal_error"
            }
        }
    )


# Mount routers with OpenAI-compatible paths (no prefix duplication)
app.include_router(chat.router, tags=["chat"])
app.include_router(files.router, tags=["files"])
app.include_router(models.router, tags=["models"])
app.include_router(responses.router, tags=["responses"])


@app.get("/health")
async def health_check():
    """
    Health check endpoint for service monitoring.
    
    Returns:
        dict: Status information including service name
        
    Last Grunted: 02/03/2026 10:30:00 AM UTC
    """
    return {"status": "ok", "service": "chat-api"}
