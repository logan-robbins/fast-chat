# ============ Stage 1: Builder ============
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY chat-api/pyproject.toml chat-api/pyproject.lock* ./
RUN pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir -r <(grep -v '^dev' pyproject.toml | grep -v '^\\[project\\]' | grep -v '^name' | grep -v '^version' | grep -v '^description' | grep -v '^readme' | grep -v '^requires-python' | grep -v '^\\[build-system\\]' | grep -v '^requires' | grep -v '^backend' | grep -v '^\\[tool\\]' | grep -v '^\\[dependency-groups\\]' | grep -v '^\\[tool.pytest')

# ============ Stage 2: Runtime ============
FROM python:3.12-slim-bookworm AS runtime

# Install runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /home/appuser
USER appuser

# Copy application code
COPY --chown=appuser:appuser chat-api/src src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
