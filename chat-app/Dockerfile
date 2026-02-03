# Build stage
FROM python:3.13-slim-bookworm AS builder

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy dependency definitions (uv.lock may not exist locally)
COPY pyproject.toml ./

# Install dependencies (without installing the project yet, for cache efficiency)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy actual source code
COPY src/tgpt_chat/ ./src/tgpt_chat/
COPY langgraph.json ./

# Now install the actual project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Runtime stage
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy everything from builder
COPY --from=builder /app /app

# Set Python path for imports
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Expose gRPC and health check ports
EXPOSE 50058 8081

# Entrypoint for production - runs gRPC server with embedded HTTP health server
CMD ["python", "-m", "tgpt_chat.server"]
