# =============================================================================
# Dockerfile - Fast-Chat UI (Chainlit Frontend)
# =============================================================================
# Multi-stage production build for the Chainlit frontend.
# 
# Build: docker build -t fast-chat-ui .
# Run:   docker run -p 8080:8080 -e CHAT_API_URL=http://chat-api:8000 fast-chat-ui
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build stage with full dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache .


# -----------------------------------------------------------------------------
# Stage 2: Production runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd -r chainlit && useradd -r -g chainlit chainlit

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app.py ./
COPY .chainlit ./.chainlit
COPY public ./public

# Set ownership
RUN chown -R chainlit:chainlit /app

# Switch to non-root user
USER chainlit

# Environment variables with defaults
ENV CHAT_API_URL=http://chat-api:8000 \
    CHAT_API_TIMEOUT=120.0 \
    DEFAULT_MODEL=gpt-4o \
    LOG_LEVEL=INFO \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/healthz', timeout=5).raise_for_status()" || exit 1

# Run Chainlit
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8080"]
