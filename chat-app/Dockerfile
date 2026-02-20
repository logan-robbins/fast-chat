# ============ Stage 1: Builder ============
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --no-dev --no-install-project

# ============ Stage 2: Runtime ============
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/artifacts && \
    chown -R appuser:appuser /home/appuser /app
USER appuser

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser prompts ./prompts

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "chat_app.api:app_api", "--host", "0.0.0.0", "--port", "8000"]
