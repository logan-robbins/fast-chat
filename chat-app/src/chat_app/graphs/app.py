"""
LangGraph Application Entry Point

This module exports the compiled supervisor graph for LangGraph Studio.
It creates and compiles the multi-agent supervisor system.
"""

import os
import logging
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Setup logger
logger = logging.getLogger(__name__)

from chat_app.graphs.supervisor import create_supervisor_graph
from chat_app.agents import get_all_agents
import chat_app.config  # loads .env

# Configure Phoenix tracing
phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
project_name = os.getenv("PHOENIX_PROJECT_NAME", "ai-orchestrator")

try:
    tracer_provider = register(
        project_name=project_name,
        endpoint=phoenix_endpoint,
        auto_instrument=False  # We'll manually instrument only what we need
    )

    # Only instrument LangChain to capture chain traces (not individual OpenAI calls)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    logger.info(
        "Phoenix tracing initialized",
        extra={"endpoint": phoenix_endpoint, "project": project_name}
    )
except Exception as e:
    logger.warning(
        "Phoenix instrumentation failed, continuing without tracing",
        extra={"error": str(e)}
    )

# Create and compile supervisor with agents
# Note: LangGraph Server handles persistence automatically, no checkpointer needed
agents = get_all_agents()
app = create_supervisor_graph(agents)
