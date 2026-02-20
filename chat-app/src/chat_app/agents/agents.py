"""Agent creation functions with prompts loaded from external files.

This module provides factory functions for creating specialized ReAct agents:
- WebSearch Agent: Web searches via Perplexity API
- Knowledge Base Agent: RAG queries against vector store (ChromaDB)
- Code Interpreter Agent: Python REPL with file management

Each agent is created using LangGraph's create_react_agent which implements
the ReAct (Reasoning + Acting) pattern with tool-calling capabilities.
Agent prompts are loaded from markdown files in the prompts/ directory.
Those markdown templates follow the 2026 prompt engineering best practices:
clear tool expectations, explicit citation guidance, and transparent reporting.

Architecture:
    Agents are designed for isolated context invocation - they receive only
    a task description from the supervisor, not the full conversation history.
    This enables predictable token usage and clean separation of concerns.

Dependencies:
- langchain>=1.2.4 for init_chat_model
- langgraph>=0.2.0 for create_react_agent
- langchain_experimental>=0.3.0 for PythonREPLTool

Last Grunted: 02/04/2026 06:30:00 PM PST
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

from chat_app.config import get_settings
from chat_app.services.llm_factory import init_openai_chat_model

logger = logging.getLogger(__name__)
settings = get_settings()


def load_prompt(filename: str) -> str:
    """Load agent system prompt from markdown file in prompts directory.

    Reads the prompt file from the prompts/ directory (relative to project root)
    and substitutes template variables with values from settings.

    Supported template variables:
        - {workspace}: Agent workspace directory for file operations

    Args:
        filename: Name of the markdown file in the prompts/ directory
            (e.g., "websearch_agent.md", "code_interpreter_agent.md")

    Returns:
        str: The prompt content with template variables substituted.

    Raises:
        FileNotFoundError: If the prompt file does not exist at the expected path.

    Example:
        >>> prompt = load_prompt("websearch_agent.md")
        >>> "web search" in prompt.lower()
        True

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    prompt_path = Path(__file__).parent.parent.parent.parent / "prompts" / filename
    
    if not prompt_path.exists():
        logger.error(
            "Prompt file not found",
            extra={"path": str(prompt_path)}
        )
        raise FileNotFoundError(f"Agent prompt not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substitute template variables
    return content.replace("{workspace}", settings.agent_workspace)


def _get_agent_model(
    model_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    temperature: Optional[float] = None
) -> BaseChatModel:
    """Get a chat model configured for agent use via init_chat_model.

    Uses LangChain's init_chat_model for provider-agnostic model initialization.
    Supports automatic provider detection from model name prefix.

    Args:
        model_name: Model identifier with optional provider prefix.
            Examples: "gpt-4o", "openai:gpt-4o", "anthropic:claude-3-opus"
            Defaults to settings.default_model or "gpt-4o".
        tags: List of string tags for observability/tracing.
            Used by Phoenix/OpenTelemetry to group traces.
        temperature: Override temperature (default: settings.default_temperature).

    Returns:
        BaseChatModel: Configured chat model with streaming enabled.

    Example:
        >>> model = _get_agent_model("gpt-4o", tags=["websearch"])
        >>> # Model is ready for use with create_react_agent

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    model = model_name or settings.default_model or "gpt-4o"
    temp = temperature if temperature is not None else settings.default_temperature
    
    # Add provider prefix if not present
    return init_openai_chat_model(
        model,
        temperature=temp,
        streaming=True,
        tags=tags or []
    )

def create_websearch_agent():
    """Create a ReAct agent for web searches using MCP tools.

    The websearch agent specializes in retrieving current information from the web.
    It uses the configured MCP `web_search` capability (e.g., via Bright Data)
    to provide AI-synthesized search results with source citations.

    Context Engineering:
        This agent receives only a task description from the supervisor,
        not the full conversation history. This ensures predictable token
        usage and focused responses.

    Returns:
        CompiledGraph: A LangGraph ReAct agent configured with:
            - web_search tool for web queries with citations
            - System prompt from websearch_agent.md
            - Model specified by WEBSEARCH_MODEL env var (default: gpt-4o)
            - Streaming enabled for real-time token output

    Environment Variables:
        WEBSEARCH_MODEL: Override the default model for this agent
        MCP_SERVERS_JSON: Required configuration for MCP servers

    Last Grunted: 02/16/2026 05:10:00 PM PST
    """
    from chat_app.tools import web_search

    model_name = os.getenv("WEBSEARCH_MODEL", settings.default_model)
    
    logger.info(
        "Creating websearch agent",
        extra={"model": model_name}
    )

    return create_react_agent(
        model=_get_agent_model(model_name, tags=["websearch"]),
        tools=[web_search],
        name="websearch",
        prompt=load_prompt("websearch_agent.md")
    )


def create_knowledge_base_agent():
    """Create a ReAct agent for RAG queries against uploaded documents.

    The knowledge base agent specializes in semantic search over documents
    that have been uploaded and indexed in the vector store (ChromaDB).
    It respects thread-scoped vector collections for multi-tenant isolation.

    Context Engineering:
        This agent receives only a task description from the supervisor.
        Vector collections are passed via RunnableConfig at invocation time,
        extracted from config["configurable"]["vector_collections"].

    Returns:
        CompiledGraph: A LangGraph ReAct agent configured with:
            - search_document_content tool for vector store queries
            - System prompt from knowledge_base_agent.md
            - Model specified by KNOWLEDGE_BASE_MODEL env var (default: gpt-4o)

    Environment Variables:
        KNOWLEDGE_BASE_MODEL: Override the default model for this agent

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    from chat_app.tools.rag_tool import search_document_content

    model_name = os.getenv("KNOWLEDGE_BASE_MODEL", settings.default_model)
    
    logger.info(
        "Creating knowledge_base agent",
        extra={"model": model_name}
    )

    return create_react_agent(
        model=_get_agent_model(model_name, tags=["knowledge_base"]),
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

    Security Considerations:
        - File operations are sandboxed to workspace directory only
        - Path traversal attempts are blocked by workspace_file_tools
        - Code execution is NOT sandboxed - use with trusted inputs only

    Context Engineering:
        This agent receives only a task description from the supervisor.
        The workspace path is injected into the system prompt for context.

    Returns:
        CompiledGraph: A LangGraph ReAct agent configured with:
            - PythonREPLTool for code execution
            - Workspace file tools: copy, delete, search, list, move, read, write
            - System prompt from code_interpreter_agent.md
            - Model specified by CODE_INTERPRETER_MODEL env var (default: gpt-4o)

    Environment Variables:
        CODE_INTERPRETER_MODEL: Override the default model for this agent
        MPLBACKEND: Matplotlib backend (set to "Agg" for headless operation)

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    workspace_path = Path(settings.agent_workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Set matplotlib to headless mode
    os.environ.setdefault("MPLBACKEND", settings.matplotlib_backend)

    from chat_app.tools.workspace_file_tools import create_workspace_file_tools
    
    file_tools = list(create_workspace_file_tools(workspace_path))

    # Initialize Python REPL with matplotlib configured for headless operation
    python_tool = PythonREPLTool()
    python_tool.python_repl.run("import matplotlib; matplotlib.use('Agg')")

    model_name = os.getenv("CODE_INTERPRETER_MODEL", settings.default_model)
    
    logger.info(
        "Creating code_interpreter agent",
        extra={"model": model_name, "workspace": str(workspace_path)}
    )

    return create_react_agent(
        model=_get_agent_model(model_name, tags=["code_interpreter"]),
        tools=file_tools + [python_tool],
        name="code_interpreter",
        prompt=load_prompt("code_interpreter_agent.md")
    )


def get_all_agents() -> Dict[str, Dict[str, Any]]:
    """Create and return all specialized agents with their descriptions.

    Factory function that instantiates all available agents for use by the
    supervisor. Each agent is returned with a description used for routing
    decisions.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping agent names to configs:
            {
                "websearch": {"agent": CompiledGraph, "description": str},
                "knowledge_base": {"agent": CompiledGraph, "description": str},
                "code_interpreter": {"agent": CompiledGraph, "description": str}
            }

    Note:
        Agents are created synchronously during application startup.
        Each agent is a compiled LangGraph that can be invoked with
        isolated context (task description only).

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    logger.info("Initializing all agents")
    
    agents = {
        "websearch": {
            "agent": create_websearch_agent(),
            "description": "Search the web for current information, market data, and news using Bright Data"
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
    
    logger.info(
        "Initialized all agents",
        extra={"agent_count": len(agents), "agents": list(agents.keys())}
    )
    
    return agents
