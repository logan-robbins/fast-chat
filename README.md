# Fast-Chat Platform

A production-ready chat platform with multi-agent orchestration, RAG capabilities, and OpenAI API compatibility. Designed to match ChatGPT/Claude-quality user experience with real-time status streaming, context window visualization, and intelligent agent handoffs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              chat-ui (Chainlit)                              │
│  Production Chainlit Frontend:                                               │
│  - OAuth/SSO authentication (GitHub, Google, Azure AD, Okta)                 │
│  - Real-time SSE streaming with status events                                │
│  - File uploads with drag-and-drop                                           │
│  - Chat history persistence (SQLAlchemy/PostgreSQL)                          │
│  - Modern dark theme with professional styling                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            chat-api (BFF)                                    │
│  FastAPI Backend-for-Frontend:                                               │
│  - OpenAI-compatible API (chat/completions, responses, files, models)        │
│  - Thread/conversation persistence (SQLite)                                  │
│  - Token counting with tiktoken                                              │
│  - SSE streaming with status & usage events                                  │
│  - File uploads & processing (uses docproc)                                  │
│  - Vector search API (/v1/search) - owns ChromaDB                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                           ▼                       ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
│         chat-app                │  │              docproc                     │
│  LangGraph Multi-Agent:         │  │  Document processing library:            │
│  - Supervisor with handoffs     │  │  - PDF/PPTX/DOCX/XLSX extraction         │
│  - WebSearch (Perplexity)       │  │  - GPT-4o vision for scanned docs        │
│  - Knowledge Base (RAG)         │  │  - Summarization                         │
│  - Code Interpreter             │  │  - ChromaDB vector storage               │
│  - Context engineering          │  │  (imported by chat-api only)             │
│  - Calls chat-api for search    │  │                                          │
└─────────────────────────────────┘  └─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│      PostgreSQL / Redis         │
│  - Checkpoint persistence       │
│  - User memories (semantic)     │
│  - Request cancellation         │
└─────────────────────────────────┘
```

## Components

| Component | Purpose | Port |
|-----------|---------|------|
| `chat-ui` | Chainlit frontend with OAuth and SSE streaming | 8080 |
| `chat-api` | OpenAI-compatible BFF, threads, files, streaming | 8000 |
| `chat-app` | LangGraph multi-agent orchestrator | 8001 |
| `docproc` | Document processing library | (library) |
| `postgres` | Checkpoints, user memories | 5432 |
| `redis` | Request cancellation | 6379 |

## Streaming Protocol (SSE)

The API uses Server-Sent Events with multiple event types for rich client experiences:

### Event Types

| Event | Purpose | Payload |
|-------|---------|---------|
| `(default)` | Token streaming | OpenAI-compatible chunks |
| `status` | User-friendly status updates | `{type, message, agent?, tool?, details?}` |
| `usage` | Token usage & context stats | `{prompt_tokens, completion_tokens, context_utilization_pct}` |
| `agent_start` | Agent routing (legacy) | `{node: "websearch"}` |
| `complete` | Stream finished | `{type: "complete"}` |

### Status Event Types

```typescript
type StatusType = 
  | "thinking"        // Supervisor reasoning
  | "agent_handoff"   // Routing to agent: "Searching the web..."
  | "tool_start"      // Tool starting: "Looking through your documents..."
  | "tool_progress"   // Progress: "Searching collection 2/3..."
  | "tool_complete"   // Done: "Found 5 relevant documents"
  | "error";          // User-friendly error
```

### Example Stream

```
event: status
data: {"type":"thinking","message":"Thinking..."}

event: status
data: {"type":"agent_handoff","message":"Searching the web...","agent":"websearch"}

event: status
data: {"type":"tool_start","message":"Searching the web for: Python 3.12 features...","tool":"perplexity_search"}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"role":"assistant"}}]}
data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Python 3.12"}}]}
data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" introduces"}}]}

event: usage
data: {"prompt_tokens":150,"completion_tokens":200,"total_tokens":350,"context_window_limit":128000,"context_utilization_pct":0.3,"is_final":true}

data: [DONE]
```

### Stream Options

```json
{
  "stream": true,
  "stream_options": {
    "include_usage": true,   // Token usage in final chunk + periodic updates
    "include_status": true   // Status events (thinking, tool use, handoffs)
  }
}
```

## Context Engineering

### Message Trimming & Summarization

Long conversations are automatically managed:
- **Threshold**: 120k tokens triggers trimming
- **Target**: Trim to 100k tokens (leaving room for response)
- **Strategy**: Keep recent messages, summarize older ones
- **Summary**: Prepended to context as "Earlier conversation summary"

### Isolated Agent Context

Agents receive **only task descriptions**, not full conversation history:
- Prevents context bloat in multi-agent flows
- Predictable token usage regardless of conversation length
- Each agent gets focused, relevant context

### User Memory

Semantic search over stored user preferences and facts:
- Memories retrieved via PostgresStore
- Injected into supervisor prompt for personalization
- Namespace: `("users", user_id, "memories")`

## Quick Start

```bash
# 1. Set environment
export OPENAI_API_KEY=sk-...

# 2. Install docproc library
pip install -e ./docproc

# 3. Install and run chat-api
cd chat-api
uv venv --python 3.12 .venv
pip install -e .
uvicorn src.main:app --port 8000

# 4. Install and run chat-app (separate terminal)
cd chat-app
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
langgraph dev

# 5. Install and run chat-ui (separate terminal)
cd chat-ui
uv venv --python 3.12 .venv
uv pip install -e .
chainlit run app.py --port 8080
```

### Docker Compose (Recommended)

```bash
# Run all services with Docker Compose
docker-compose up

# Access the UI at http://localhost:8080
```

## Key Design Decisions

### 1. Three-Tier Architecture

```
Frontend  →  chat-api (BFF)  →  chat-app (Orchestrator)
```

- **chat-api**: Thin API layer, thread management, token counting, SSE forwarding
- **chat-app**: Business logic, agent orchestration, context engineering
- **Separation**: chat-api could route to different backends (graph versions) in the future

### 2. ChatGPT/Claude-Style UX

Real-time status streaming mirrors modern AI assistants:
- "Thinking..." during supervisor reasoning
- "Searching the web..." during tool use
- "Found 5 relevant documents" on completion
- Context window visualization with utilization percentage

### 3. Context Engineering

- **Token counting**: tiktoken for accurate counts
- **Trimming**: Auto-trim at 120k tokens, keep most recent
- **Summarization**: Older messages summarized to preserve context
- **Isolated handoffs**: Agents receive task descriptions, not full history
- **User memory**: Semantic search over stored preferences

### 4. Multi-Agent Architecture

chat-app uses a supervisor pattern with isolated context handoffs:
- **Supervisor**: Routes requests to specialized agents
- **WebSearch Agent**: Perplexity API for current information
- **Knowledge Base Agent**: RAG over uploaded documents
- **Code Interpreter Agent**: Python execution with file tools
- **Isolated Context**: Each agent receives only task description, not full conversation history

### 5. Vector Store Ownership

**chat-api owns the vector store** (ChromaDB):
- Document uploads go to chat-api → stored in ChromaDB via docproc
- RAG queries: chat-app calls chat-api's `/v1/search` endpoint
- Single source of truth for embeddings
- Easy migration path: swap ChromaDB for Pinecone/Weaviate in chat-api only

docproc is a library (not a service) imported by chat-api:
- No separate container for document processing
- Simpler deployment architecture

### 6. Smart Document Routing

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
# chat-ui (Chainlit Frontend)
CHAT_API_URL=http://localhost:8000     # URL for chat-api backend
CHAT_API_TIMEOUT=120.0                 # Request timeout in seconds
DEFAULT_MODEL=gpt-4o                   # Default model for completions
CHAINLIT_AUTH_SECRET=                  # Generate with: openssl rand -hex 32
CHAINLIT_URL=http://localhost:8080     # URL for OAuth callbacks

# OAuth providers (configure at least one for production)
OAUTH_GITHUB_CLIENT_ID=
OAUTH_GITHUB_CLIENT_SECRET=
OAUTH_GOOGLE_CLIENT_ID=
OAUTH_GOOGLE_CLIENT_SECRET=
OAUTH_AZURE_AD_CLIENT_ID=
OAUTH_AZURE_AD_CLIENT_SECRET=
OAUTH_AZURE_AD_TENANT_ID=

# chat-api (BFF) - owns vector store
DATABASE_URL=sqlite:///./chat.db
CHAT_APP_URL=http://localhost:8001/v1/chat/completions
CHROMA_PERSIST_DIR=/path/to/persist    # Empty = in-memory (dev)
LOG_LEVEL=INFO
LOG_FORMAT=json                        # json (production) or console (dev)

# chat-app (Orchestrator) - calls chat-api for search
CHAT_API_URL=http://localhost:8000     # URL for chat-api
CHECKPOINTER_TYPE=postgres             # postgres (production) or memory (dev)
DATABASE_URL=postgresql://user:pass@localhost:5432/chatapp

# Document processing (used by chat-api)
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Web search
PERPLEXITY_API_KEY=pplx-...

# Redis (for request cancellation)
REDIS_HOST=localhost
REDIS_PORT=6379

# HTTP client tuning
HTTP_MAX_CONNECTIONS=100
HTTP_TIMEOUT_READ=120.0

# Local development
USE_LOCAL_MODELS=true
OLLAMA_BASE_URL=http://localhost:11434
```

## API Endpoints

### chat-api (OpenAI-compatible BFF)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions with streaming, status, usage |
| `/v1/responses` | POST | Responses API (stateful conversations) |
| `/v1/responses/{id}` | GET/DELETE | Retrieve or delete stored response |
| `/v1/responses/{id}/cancel` | POST | Cancel background response |
| `/v1/responses/compact` | POST | Compact conversation history |
| `/v1/threads` | GET/POST | List or create conversation threads |
| `/v1/threads/{id}` | PATCH/DELETE | Update or delete thread |
| `/v1/threads/{id}/messages` | GET | Get thread message history |
| `/v1/threads/{id}/fork` | POST | Fork conversation at checkpoint |
| `/v1/models` | GET | List available models |
| `/v1/files` | POST/GET | File upload and management |
| `/v1/search` | POST | Semantic search over document collections |
| `/health` | GET | Health check |

### chat-app (LangGraph Orchestrator)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Multi-agent chat with streaming |
| `/v1/chat/resume` | POST | Resume interrupted conversation (HITL) |
| `/v1/chat/interrupt/{thread_id}` | GET | Check for pending interrupts |
| `/health` | GET | Health check |

## Frontend Integration

### TypeScript Example

```typescript
interface StatusEvent {
  type: 'thinking' | 'tool_start' | 'tool_progress' | 'tool_complete' | 'agent_handoff' | 'error';
  message: string;
  agent?: 'websearch' | 'knowledge_base' | 'code_interpreter';
  tool?: string;
  details?: { current?: number; total?: number; query?: string };
}

interface UsageEvent {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  context_window_limit: number;
  context_utilization_pct: number;
  is_final: boolean;
}

async function streamChat(messages: Message[]) {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages,
      stream: true,
      stream_options: { include_status: true, include_usage: true }
    })
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7);
      } else if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;
        
        if (currentEvent === 'status') {
          handleStatus(JSON.parse(data) as StatusEvent);
        } else if (currentEvent === 'usage') {
          handleUsage(JSON.parse(data) as UsageEvent);
        } else {
          handleToken(JSON.parse(data));
        }
        currentEvent = null;
      }
    }
  }
}

function handleStatus(status: StatusEvent) {
  switch (status.type) {
    case 'thinking': showSpinner('Thinking...'); break;
    case 'agent_handoff':
    case 'tool_start': showStatus(status.message); break;
    case 'tool_progress': updateProgress(status.details?.current, status.details?.total); break;
    case 'tool_complete': hideStatus(); break;
    case 'error': showError(status.message); break;
  }
}

function handleUsage(usage: UsageEvent) {
  updateContextBar(usage.context_utilization_pct);
  updateTokenCount(usage.total_tokens);
}
```

## Development

```bash
# Run tests
cd docproc && pytest
cd chat-api && pytest  
cd chat-app && pytest

# Docker Compose (from project root)
docker-compose up
```

## Storage

- **SQLite**: Thread and message persistence (chat-api BFF)
- **PostgreSQL**: LangGraph checkpoints and user memories (chat-app)
- **ChromaDB**: Vector embeddings for RAG (docproc)
- **Redis**: Request cancellation signaling

## Feature Comparison

| Feature | ChatGPT | Claude | Fast-Chat |
|---------|---------|--------|-----------|
| Token streaming | ✓ | ✓ | ✓ |
| Status updates ("Searching...") | ✓ | ✓ | ✓ |
| Context window visualization | ✓ | ✓ | ✓ |
| Multi-agent routing | ✓ | ✓ | ✓ |
| RAG over documents | ✓ | ✓ | ✓ |
| Code execution | ✓ | ✓ | ✓ |
| Web search | ✓ | ✓ | ✓ |
| Conversation memory | ✓ | ✓ | ✓ |
| Human-in-the-loop | - | - | ✓ |
| Open source | - | - | ✓ |

## Docs

Additional documentation in `/docs`:
- `memory.md` - Memory and persistence patterns
- `streaming.md` - SSE streaming implementation
- `thinking.md` - Extended thinking support
- `subgraphs.md` - Agent subgraph architecture
- `interrupt.md` - Human-in-the-loop interrupts
- `workflows_and_agents.md` - LangGraph patterns
