# Chat-App

LangGraph multi-agent orchestrator for the Fast-Chat platform.

## Overview

Chat-App implements LangGraph's supervisor pattern where a central orchestrator analyzes incoming requests and delegates tasks to specialized agents:

- **websearch**: Real-time web searches via Perplexity API
- **knowledge_base**: RAG queries over uploaded documents (ChromaDB)
- **code_interpreter**: Python REPL for calculations and data analysis

## Project Structure

```
src/chat_app/
├── api.py                 # FastAPI entry point
├── config.py              # Service configuration
├── agents/
│   └── agents.py          # Agent factory functions
├── graphs/
│   ├── app.py             # LangGraph application export
│   └── supervisor.py      # Supervisor graph creation
├── tools/
│   ├── tools.py           # Perplexity search tool
│   ├── rag_tool.py        # Document search tool
│   └── workspace_file_tools.py  # File management tools
└── utils/
    ├── messages.py        # Message conversion utilities
    ├── perplexity_client.py  # Perplexity API client
    └── file_context.py    # File context management
```

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- (Optional) Perplexity API key for web search

### Installation

```bash
cd chat-app
pip install -e .
```

### Running with LangGraph Studio

```bash
export OPENAI_API_KEY=sk-...
export PERPLEXITY_API_KEY=pplx-...  # Optional

langgraph dev
```

The LangGraph Studio will be available at http://localhost:2024

### Running as API Server

```bash
uvicorn src.chat_app.api:app_api --port 8080
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `PERPLEXITY_API_KEY` | Perplexity API key | Optional |
| `DEFAULT_MODEL` | Default LLM model | gpt-4o |
| `DEFAULT_TEMPERATURE` | LLM temperature | 0.0 |
| `REDIS_HOST` | Redis host | localhost |
| `REDIS_PORT` | Redis port | 6379 |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence | ./data/chroma |

## Agent Prompts

Agent system prompts are stored in `prompts/`:

- `websearch_agent.md` - Web search instructions
- `knowledge_base_agent.md` - RAG query instructions
- `code_interpreter_agent.md` - Code execution instructions

## Development

```bash
# Run tests
pytest

# Run with hot reload
langgraph dev --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat via supervisor graph |
| `/health` | GET | Health check |
