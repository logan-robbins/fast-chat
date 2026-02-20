# Fast-Chat

Fast-Chat is a multi-service conversational AI system with:
- `chat-ui`: Chainlit frontend
- `chat-api`: OpenAI-compatible BFF API
- `chat-app`: LangGraph multi-agent orchestrator
- `docproc`: document processing and vector-store utilities

## Architecture (Human View)

### Service Topology

```text
Browser
  |
  v
chat-ui (Chainlit, :8080)
  |
  v
chat-api (FastAPI BFF, :8000)
  |
  v
chat-app (LangGraph orchestrator, :8001)
  |
  +--> Redis (state + cancellation, :6379)
  +--> Postgres (checkpoints + memory, :5432)

chat-api also owns vector-search API and persists API-layer data.
```

### Request/Response Flow

```text
User message
  -> chat-ui
  -> chat-api `/v1/chat/completions` or `/v1/responses`
  -> chat-app supervisor graph
  -> tools/agents (web, RAG, code)
  -> SSE stream back through chat-api
  -> chat-ui renders tokens + status + usage events
```

## LLM-First Quickstart

### 1. Prerequisites
- Docker + Docker Compose plugin
- `uv` installed
- OpenAI API key

### 2. Configure environment

```bash
cp .env.example .env
# edit .env and set required keys
```

Minimum required for end-to-end chat:
- `OPENAI_API_KEY`

Common optional keys:
- `PERPLEXITY_API_KEY` (if enabled for web search provider)
- `LITELLM_BASE_URL`, `LITELLM_API_KEY` (if routing via LiteLLM)
- `CHAINLIT_AUTH_SECRET` (chat-ui auth/session hardening)

### 3. Run full stack (canonical path)

```bash
docker compose up --build -d
```

Access:
- UI: `http://localhost:8080`
- API docs: `http://localhost:8000/docs`
- Orchestrator health: `http://localhost:8001/health`

### 4. Stop

```bash
docker compose down
```

## Key Commands (Canonical)

### Compose lifecycle (project root)

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f chat-api
docker compose logs -f chat-app
docker compose logs -f chat-ui
docker compose down
```

### Local development without Docker (multi-terminal)

```bash
# sync environments
(cd docproc && uv sync --all-groups --frozen)
(cd chat-api && uv sync --all-groups --frozen)
(cd chat-app && uv sync --all-groups)
(cd chat-ui && uv sync --all-groups)

# terminal 1: chat-api
(cd chat-api && uv run uvicorn src.main:app --port 8000)

# terminal 2: chat-app
(cd chat-app && uv run langgraph dev)

# terminal 3: chat-ui
(cd chat-ui && uv run chainlit run app.py --port 8080)
```

### Tests

```bash
(cd docproc && uv run pytest -q)
(cd chat-api && uv run pytest -q)
(cd chat-app && uv run pytest -q)
(cd chat-ui && uv run pytest -q)
```

## Stable Paths (Prefer Referencing These)

These files are high-value anchors for agents because they define system contracts:

- `docker-compose.yml`
Reason: canonical runtime topology and container env wiring.

- `.env.example`
Reason: authoritative environment-variable schema and defaults.

- `chat-api/src/main.py`
Reason: API service bootstrap, middleware, and route registration.

- `chat-app/src/chat_app/graphs/app.py`
Reason: compiled LangGraph entrypoint used at runtime.

- `chat-app/src/chat_app/api.py`
Reason: OpenAI-compatible adapter and streaming interface.

- `chat-ui/app.py`
Reason: frontend runtime behavior, SSE handling, and model selection flow.

If file-level paths drift, use these folder-level anchors:
- `chat-api/src/routers/`: external API contracts and route behavior.
- `chat-app/src/chat_app/agents/`: agent role definitions.
- `chat-app/src/chat_app/tools/`: tool execution surface available to agents.
- `chat-app/prompts/`: system prompts and role instructions.
- `docproc/docproc/services/`: document extraction, chunking, summarization, and vector-store workflows.
- `tests/`: cross-service behavioral contracts and integration checks.

## Runtime Contracts (Important Endpoints)

### chat-api (`:8000`)
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`
- `POST /v1/search`
- `GET /health`

### chat-app (`:8001`)
- `POST /v1/chat/completions`
- `POST /v1/chat/resume`
- `GET /v1/chat/interrupt/{thread_id}`
- `GET /health`

## Data + State Ownership

- `chat-api`:
  - BFF request normalization
  - model policy / registry surfaces
  - search endpoint ownership
- `chat-app`:
  - supervisor routing
  - tool orchestration
  - memory/checkpoint interactions
- Redis:
  - cancellation and transient coordination state
- Postgres:
  - durable checkpoints/memory backing

## Operational Notes

- Single canonical Compose file: `docker-compose.yml`.
- Default service ports:
  - `8080` chat-ui
  - `8000` chat-api
  - `8001` chat-app (container internal maps to host `8001`)
  - `6379` Redis
  - `5432` Postgres
- Health checks are defined in Compose; use `docker compose ps` for status.

## Troubleshooting

- Service fails at startup:
  - run `docker compose logs -f <service>`
  - verify `.env` values against `.env.example`
- API timeout from UI:
  - verify `CHAT_API_URL` wiring in `docker-compose.yml`
- No model responses:
  - verify `OPENAI_API_KEY`
  - check `chat-api` and `chat-app` logs for provider/auth errors

## Repository Map

```text
chat-api/   FastAPI BFF and OpenAI-compatible API layer
chat-app/   LangGraph orchestration, agents, tools, prompts
chat-ui/    Chainlit frontend and user interaction layer
docproc/    Document extraction/summarization/vector utilities
tests/      Cross-service tests and contract checks
docs/       Supplemental design and behavior notes
```
