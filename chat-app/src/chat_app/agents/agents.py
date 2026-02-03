"""Agent creation functions with prompts loaded from external files.

This module provides factory functions for creating specialized ReAct agents:
- WebSearch Agent: Web searches via Perplexity API
- Knowledge Base Agent: RAG queries against vector store (ChromaDB)
- Code Interpreter Agent: Python REPL with file management

Each agent is created using LangGraph's create_react_agent which implements
the ReAct (Reasoning + Acting) pattern with tool-calling capabilities.
Agent prompts are loaded from markdown files in the prompts/ directory.

Dependencies:
- langchain_openai>=0.3.0 for ChatOpenAI
- langgraph>=0.2.0 for create_react_agent

Last Grunted: 02/03/2026 03:15:00 PM PST
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from chat_app.config import get_settings

settings = get_settings()

def load_prompt(filename: str) -> str:
    """Load agent system prompt from markdown file in prompts directory.

    Reads the prompt file from the prompts/ directory (relative to project root)
    and substitutes template variables like {workspace} with values from settings.

    Args:
        filename (str): Name of the markdown file in the prompts/ directory
            (e.g., "websearch_agent.md", "code_interpreter_agent.md")

    Returns:
        str: The prompt content with template variables substituted.
            Currently supports {workspace} substitution.

    Raises:
        FileNotFoundError: If the prompt file does not exist at the expected path

    Example:
        >>> prompt = load_prompt("websearch_agent.md")
        >>> "web search" in prompt.lower()
        True

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / filename
    with open(prompt_path, 'r') as f:
        content = f.read()
    # Replace any template variables
    return content.replace("{workspace}", settings.agent_workspace)

def _get_agent_model(model_name: str = "gpt-4o", tags: list = None) -> ChatOpenAI:
    """Get ChatOpenAI model configured for agents with streaming enabled.

    Creates a ChatOpenAI instance with settings optimized for ReAct agent use:
    - Streaming enabled for real-time token output
    - Temperature from settings (default 0.0 for deterministic responses)
    - Optional tags for Phoenix/OpenTelemetry tracing

    Args:
        model_name (str): OpenAI model identifier. Valid options include:
            - "gpt-4o" (default): GPT-4 Omni - most capable
            - "gpt-4o-mini": Faster, lower cost variant
            Defaults to "gpt-4o".
        tags (list, optional): List of string tags for observability/tracing.
            Used by Phoenix/OpenTelemetry to group traces by agent type.
            Example: ["websearch"], ["code_interpreter"]

    Returns:
        ChatOpenAI: Configured LangChain ChatOpenAI instance ready for use
            with create_react_agent. Has streaming=True set.

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    params = {
        "model": model_name,
        "temperature": settings.default_temperature,
        "streaming": True
    }
    
    # Add tags if provided (for observability/tracing)
    if tags:
        params["tags"] = tags

    return ChatOpenAI(**params)

def create_websearch_agent():
    """Create a ReAct agent for web searches using Perplexity API.

    The websearch agent specializes in retrieving current information from the web.
    It uses the Perplexity API (sonar model) which provides AI-synthesized
    search results with source citations.

    Returns:
        CompiledGraph: A LangGraph ReAct agent configured with:
            - perplexity_search tool for web queries with citations
            - System prompt from websearch_agent.md
            - Model specified by WEBSEARCH_MODEL env var (default: gpt-4o)
            - Streaming enabled for real-time token output

    Environment Variables:
        WEBSEARCH_MODEL: Override the default model for this agent
        PERPLEXITY_API_KEY: Required for perplexity_search tool

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    from chat_app.tools import perplexity_search

    return create_react_agent(
        model=_get_agent_model(os.getenv("WEBSEARCH_MODEL", settings.default_model), tags=["websearch"]),
        tools=[perplexity_search],
        name="websearch",
        prompt=load_prompt("websearch_agent.md")
    )


def create_knowledge_base_agent():
    """Create a ReAct agent for RAG queries against uploaded documents.

    The knowledge base agent specializes in semantic search over documents
    that have been uploaded and indexed in the vector store (ChromaDB).
    It respects thread-scoped vector collections for multi-tenant isolation.

    Returns:
        CompiledGraph: A LangGraph ReAct agent configured with:
            - search_document_content tool for vector store queries
            - System prompt from knowledge_base_agent.md
            - Model specified by KNOWLEDGE_BASE_MODEL env var (default: gpt-4o)

    Environment Variables:
        KNOWLEDGE_BASE_MODEL: Override the default model for this agent

    Note:
        Vector collections are passed via RunnableConfig at invocation time.
        The tool extracts collections from config["configurable"]["vector_collections"].

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    from chat_app.tools.rag_tool import search_document_content

    return create_react_agent(
        model=_get_agent_model(os.getenv("KNOWLEDGE_BASE_MODEL", settings.default_model), tags=["knowledge_base"]),
        tools=[search_document_content],
        name="knowledge_base",
        prompt=load_prompt("knowledge_base_agent.md")
    )


def create_code_agent():
    """Create a ReAct agent for Python code execution and file management.

    The code interpreter agent can execute Python code in a sandboxed REPL
    environment and manage files within the workspace directory. It supports:
    - Mathematical calculations and data analysis
    - File operations (read, write, copy, move, delete, list, search)
    - Matplotlib visualizations (headless with Agg backend)

    The workspace directory (default: "artifacts/") is created if it doesn't
    exist, and all file operations are restricted to this directory for security.

    Returns:
        CompiledGraph: A LangGraph ReAct agent configured with:
            - PythonREPLTool for code execution (uses langchain_experimental)
            - Workspace file tools: copy, delete, search, list, move, read, write
            - System prompt from code_interpreter_agent.md
            - Model specified by CODE_INTERPRETER_MODEL env var (default: gpt-4o)

    Environment Variables:
        CODE_INTERPRETER_MODEL: Override the default model for this agent
        MPLBACKEND: Matplotlib backend (set to "Agg" for headless operation)

    Security:
        - File operations are sandboxed to workspace directory only
        - Path traversal attempts are blocked by workspace_file_tools
        - Code execution is NOT sandboxed - use with trusted inputs only

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    workspace_path = Path(settings.agent_workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", settings.matplotlib_backend)

    from chat_app.tools.workspace_file_tools import create_workspace_file_tools
    
    file_tools = list(create_workspace_file_tools(workspace_path))

    python_tool = PythonREPLTool()
    python_tool.python_repl.run("import matplotlib; matplotlib.use('Agg')")

    return create_react_agent(
        model=_get_agent_model(os.getenv("CODE_INTERPRETER_MODEL", settings.default_model), tags=["code_interpreter"]),
        tools=file_tools + [python_tool],
        name="code_interpreter",
        prompt=load_prompt("code_interpreter_agent.md")
    )


def get_all_agents() -> Dict[str, Dict[str, Any]]:
    """Create and return all specialized agents with their descriptions.

    Factory function that instantiates all available agents for use by the
    supervisor. Each agent is returned with a description used for routing.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping agent names to configs:
            {
                "websearch": {"agent": CompiledGraph, "description": str},
                "knowledge_base": {"agent": CompiledGraph, "description": str},
                "code_interpreter": {"agent": CompiledGraph, "description": str}
            }

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    return {
        "websearch": {
            "agent": create_websearch_agent(),
            "description": "Search the web for current information using Perplexity"
        },
        "knowledge_base": {
            "agent": create_knowledge_base_agent(),
            "description": "Search uploaded documents, perform RAG queries, and access LangGraph/LangChain documentation"
        },
        "code_interpreter": {
            "agent": create_code_agent(),
            "description": "Execute Python code, solve mathematical problems, and manage files in artifacts directory"
        }
    }
