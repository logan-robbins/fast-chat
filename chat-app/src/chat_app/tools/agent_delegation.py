"""Custom handoff tools for isolated subagent context.

This module provides custom handoff tools that create isolated contexts for subagents,
preventing context bloat by passing only relevant task descriptions instead of full
conversation history.

Context Engineering Pattern:
    The isolated handoff pattern ensures:
    - Subagents receive clean, focused task descriptions (no conversation history)
    - No conversation history pollution between agents
    - Predictable token usage regardless of conversation length
    - Clear separation of concerns between supervisor and workers

Status Streaming (ChatGPT/Claude 2025/2026 patterns):
    Agent handoffs emit user-friendly status events via get_stream_writer():
    - "Searching the web..." when handing off to websearch
    - "Looking through your documents..." when handing off to knowledge_base
    - "Running Python code..." when handing off to code_interpreter

Usage:
    The supervisor calls handoff_to_{agent_name} with a detailed task description.
    The tool creates an isolated state with only that description and invokes
    the agent asynchronously.

This pattern is recommended by LangGraph documentation for multi-agent systems
where context engineering is important for cost and quality control.

Dependencies:
- langchain_core.tools: For @tool decorator and InjectedToolCallId
- langgraph.types: For Command return type
- chat_app.status_streaming: For status event emission

Last Grunted: 02/04/2026 07:30:00 PM PST
"""
from __future__ import annotations

import logging
import traceback
from typing import Annotated, Any, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

logger = logging.getLogger(__name__)


def create_isolated_handoff_tool(
    *,
    agent_name: str,
    agent_instance: Any,
    description: str | None = None
):
    """Create a custom handoff tool that provides isolated context to subagents.

    Factory function that creates a LangChain tool for handing off tasks to
    a specific agent. The handoff provides isolated context containing only
    the task description, not the full conversation history.

    Context Engineering:
        This pattern ensures predictable token usage and prevents context bloat.
        The agent receives only the task description you provide, not the full
        conversation history.

    Args:
        agent_name: Name of the agent (e.g., "websearch", "code_interpreter").
            Used to generate tool name "handoff_to_{agent_name}".
        agent_instance: The compiled LangGraph agent to invoke. Must have
            an ainvoke() method accepting {"messages": [{"role": "user", "content": ...}]}.
        description: Human-readable description of the agent's capabilities.
            Used in the tool description for LLM routing decisions.
            Defaults to "Hand off task to {agent_name} specialist".

    Returns:
        Callable: An async LangChain tool that accepts:
            - task_description (str): Detailed task for the agent (required, non-empty)
            - tool_call_id (str): Injected by LangGraph for response routing
        
        The tool returns a Command with ToolMessage containing the agent's response.

    Example:
        >>> tool = create_isolated_handoff_tool(
        ...     agent_name="websearch",
        ...     agent_instance=websearch_agent,
        ...     description="Search the web for current information"
        ... )
        >>> tool.name
        "handoff_to_websearch"

    Error Handling:
        - Empty task_description: Returns error message in ToolMessage
        - Agent returns no messages: Returns error message
        - Agent returns empty response: Returns error message
        - Any exception: Returns error message with exception details

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    tool_name = f"handoff_to_{agent_name}"
    tool_description = description or f"Hand off task to {agent_name} specialist"

    @tool(tool_name, description=tool_description)
    async def isolated_handoff(
        task_description: Annotated[
            str,
            "Detailed description of what the agent should do, including all relevant context needed for the task"
        ],
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig = None,
    ) -> Command:
        """Execute handoff to agent with isolated context containing only the task description."""
        
        # Validate task_description is not empty or just whitespace
        if not task_description or not task_description.strip():
            error_msg = (
                f"Error: task_description cannot be empty. "
                f"Please provide a detailed description of what {agent_name} should do."
            )
            logger.error(
                "Empty task_description provided",
                extra={
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "task_description_length": len(task_description) if task_description else 0
                }
            )
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                    ]
                }
            )

        logger.info(
            "Handing off to agent",
            extra={
                "agent_name": agent_name,
                "tool_call_id": tool_call_id,
                "task_description_length": len(task_description),
                "task_description_preview": task_description[:200]
            }
        )
        
        # Emit user-friendly status event (ChatGPT/Claude-style)
        try:
            from chat_app.status_streaming import emit_agent_handoff
            emit_agent_handoff(agent_name, task_description)
        except Exception as status_err:
            # Status emission failures should not break execution
            logger.debug(
                "Failed to emit status event",
                extra={"error": str(status_err), "agent_name": agent_name}
            )

        try:
            # Create isolated context with only the task description
            # This is the key context engineering pattern - no conversation history
            isolated_state: Dict[str, Any] = {
                "messages": [{"role": "user", "content": task_description}]
            }

            logger.debug(
                "Invoking agent with isolated state",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                }
            )

            # Execute the agent ASYNCHRONOUSLY with isolated context
            # Pass the parent config so subagent tools can access configurable
            # (e.g., vector_collections for RAG scoping)
            result = await agent_instance.ainvoke(isolated_state, config=config)

            logger.debug(
                "Agent invocation completed",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                    "has_messages": bool(result.get("messages")) if isinstance(result, dict) else None
                }
            )

            # Extract the final response from the agent
            final_content = _extract_agent_response(result, agent_name, tool_call_id)

            logger.info(
                "Received response from agent",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "response_length": len(str(final_content))
                }
            )
            
            # Emit completion status event
            try:
                from chat_app.status_streaming import emit_tool_complete
                emit_tool_complete(
                    tool_name=f"handoff_to_{agent_name}",
                    message=f"{agent_name.replace('_', ' ').title()} analysis complete",
                    agent_name=agent_name,
                    details={"response_length": len(str(final_content))}
                )
            except Exception as status_err:
                logger.debug(
                    "Failed to emit completion status",
                    extra={"error": str(status_err), "agent_name": agent_name}
                )

            # Return the agent's response as a tool message
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=final_content,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                    ]
                }
            )

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(
                "Error during agent handoff",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": error_traceback
                },
                exc_info=True
            )
            
            # Emit error status event
            try:
                from chat_app.status_streaming import emit_error
                emit_error(
                    f"{agent_name} encountered an issue",
                    agent_name=agent_name,
                    details={"error_type": type(e).__name__}
                )
            except Exception:
                pass  # Never let status emission break error handling
            
            error_content = (
                f"Error executing task with {agent_name}: {type(e).__name__}: {str(e)}\n"
                f"Please check the logs for more details."
            )

            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=error_content,
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                    ]
                }
            )

    return isolated_handoff


def _extract_agent_response(
    result: Dict[str, Any],
    agent_name: str,
    tool_call_id: str
) -> str:
    """Extract the final response content from an agent's result.
    
    Args:
        result: The result dict from agent invocation.
        agent_name: Name of the agent for error messages.
        tool_call_id: Tool call ID for logging context.
        
    Returns:
        str: The agent's response content, or an error message if extraction fails.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    if not result.get("messages"):
        logger.error(
            "No messages returned from agent",
            extra={
                "agent_name": agent_name,
                "tool_call_id": tool_call_id,
                "result_type": type(result).__name__,
                "result_keys": list(result.keys()) if isinstance(result, dict) else None,
            }
        )
        return f"Error: {agent_name} did not return any response"
    
    message_count = len(result["messages"])
    logger.debug(
        "Agent returned messages",
        extra={
            "agent_name": agent_name,
            "tool_call_id": tool_call_id,
            "message_count": message_count,
        }
    )

    # Get the last message from the agent (should be the final response)
    final_message = result["messages"][-1]
    final_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    # Validate that we have actual content
    if not final_content or (isinstance(final_content, str) and not final_content.strip()):
        logger.error(
            "Empty response from agent",
            extra={
                "agent_name": agent_name,
                "tool_call_id": tool_call_id,
                "message_type": type(final_message).__name__
            }
        )
        return f"Error: {agent_name} returned an empty response"
    
    return final_content


def create_isolated_handoff_tools(agents: Dict[str, Dict[str, Any]]) -> List[Any]:
    """Create isolated handoff tools for all available agents.

    Bulk factory function that creates handoff tools for each agent in the
    provided dictionary. Used during supervisor graph creation to register
    all agent handoff tools.

    Args:
        agents: Dictionary from get_all_agents() mapping agent names to configs:
            {
                "agent_name": {
                    "agent": CompiledGraph,  # The agent instance
                    "description": str       # Agent capability description
                }
            }

    Returns:
        List: List of handoff tools, one per agent. Each tool is named
            "handoff_to_{agent_name}" and configured with the agent's
            description for LLM routing decisions.

    Example:
        >>> agents = get_all_agents()
        >>> tools = create_isolated_handoff_tools(agents)
        >>> [t.name for t in tools]
        ['handoff_to_websearch', 'handoff_to_knowledge_base', 'handoff_to_code_interpreter']

    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    handoff_tools: List[Any] = []

    for agent_name, agent_config in agents.items():
        agent_instance = agent_config["agent"]
        description = agent_config.get("description", f"Specialized {agent_name} agent")

        handoff_tool = create_isolated_handoff_tool(
            agent_name=agent_name,
            agent_instance=agent_instance,
            description=f"Hand off task to {description.lower()}"
        )

        handoff_tools.append(handoff_tool)
        logger.info(
            "Created isolated handoff tool",
            extra={"agent_name": agent_name, "tool_name": handoff_tool.name}
        )

    return handoff_tools


__all__ = [
    "create_isolated_handoff_tool",
    "create_isolated_handoff_tools",
]