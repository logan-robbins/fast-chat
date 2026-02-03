# Fast-Chat Platform

A document-aware chat platform with RAG capabilities and OpenAI API compatibility.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client (Web/API)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            chat-api (BFF)                                    │
│  FastAPI service that handles:                                               │
│  - OpenAI-compatible API (chat/completions, responses, files, models)        │
│  - Thread management (SQLite)                                                │
│  - File uploads & processing (uses docproc)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                          ▼                       ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
│         chat-app                │  │              docproc                     │
│  LangGraph agent with:          │  │  Document processing library:            │
│  - Supervisor graph             │  │  - PDF/PPTX/DOCX/XLSX extraction         │
│  - RAG tool (uses docproc)      │  │  - GPT-4o vision for scanned docs        │
│  - Web search tool              │  │  - Summarization                         │
│  - Code interpreter             │  │  - ChromaDB vector storage               │
└─────────────────────────────────┘  └─────────────────────────────────────────┘
```

## Components

| Component | Purpose | Port |
|-----------|---------|------|
| `chat-api` | OpenAI-compatible API, threads, files | 8000 |
| `chat-app` | LangGraph multi-agent orchestrator | 8080 |
| `docproc` | Document processing library | (library) |
| `local-dev` | Docker Compose development setup | - |

## Quick Start

```bash
# 1. Set environment
export OPENAI_API_KEY=sk-...

# 2. Install docproc library
pip install -e ./docproc

# 3. Install and run chat-api
cd chat-api
pip install -e .
uvicorn src.main:app --port 8000

# 4. Install and run chat-app (separate terminal)
cd chat-app
pip install -e .
langgraph dev
```

## Key Design Decisions

### 1. Library over Service (docproc)

Document processing is a **library**, not a service. This means:
- No network latency for embeddings
- Simpler deployment (one less container)
- ChromaDB runs in-memory in each process

### 2. ChromaDB for Vector Storage

ChromaDB provides simple, embedded vector storage:
- No external server required
- In-memory by default (like SQLite)
- Optional persistence via `CHROMA_PERSIST_DIR`

### 3. Cloud-First

All services default to OpenAI APIs:
- `gpt-4o-mini` for chat and summarization
- `text-embedding-3-small` for embeddings
- Optional local models via `USE_LOCAL_MODELS=true`

### 4. Smart Document Routing

Documents are processed based on type:
- **Text files**: Direct read (no API calls)
- **XLSX/CSV**: Pandas extraction (no API calls)  
- **DOCX**: python-docx (no API calls)
- **PDF (text)**: pdfplumber first, vision fallback
- **PPTX**: GPT-4o vision (layout matters)

## Configuration

### Required

```bash
OPENAI_API_KEY=sk-...
```

### Optional

```bash
# Chat-api
DATABASE_URL=sqlite:///./chat.db

# Document processing
CHROMA_PERSIST_DIR=/path/to/persist    # Empty = in-memory
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Web search (for chat-app)
PERPLEXITY_API_KEY=pplx-...

# Local development
USE_LOCAL_MODELS=true
OLLAMA_BASE_URL=http://localhost:11434
```

## API Endpoints

### chat-api (OpenAI-compatible)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming) |
| `/v1/responses` | POST | Responses API (stateful) |
| `/v1/models` | GET | List models |
| `/v1/files` | POST/GET | File upload and management |
| `/health` | GET | Health check |

### chat-app (LangGraph)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat via LangGraph supervisor |
| `/health` | GET | Health check |

## Development

```bash
# Run tests
cd docproc && pytest
cd chat-api && pytest  
cd chat-app && pytest

# Docker Compose
cd local-dev
make up
```

## Storage

- **SQLite**: Thread and message persistence (chat-api)
- **ChromaDB**: Vector embeddings for RAG (docproc)
