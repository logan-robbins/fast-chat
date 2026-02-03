"""Custom handoff tools for isolated subagent context.

This module provides custom handoff tools that create isolated contexts for subagents,
preventing context bloat by passing only relevant task descriptions instead of full
conversation history.

The isolated handoff pattern ensures:
- Subagents receive clean, focused task descriptions (no conversation history)
- No conversation history pollution between agents
- Predictable token usage regardless of conversation length
- Clear separation of concerns between supervisor and workers

This pattern is recommended by LangGraph documentation for multi-agent systems
where context engineering is important for cost and quality control.

Dependencies:
- langchain_core.tools: For @tool decorator and InjectedToolCallId
- langgraph.types: For Command return type

Last Grunted: 02/03/2026 03:15:00 PM PST
"""

import asyncio
import logging
import traceback
from typing import Annotated, Dict, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

logger = logging.getLogger(__name__)


def create_isolated_handoff_tool(*, agent_name: str, agent_instance, description: str = None):
    """Create a custom handoff tool that provides isolated context to subagents.

    Factory function that creates a LangChain tool for handing off tasks to
    a specific agent. The handoff provides isolated context containing only
    the task description, not the full conversation history.

    Args:
        agent_name (str): Name of the agent (e.g., "websearch", "code_interpreter").
            Used to generate tool name "handoff_to_{agent_name}".
        agent_instance: The compiled LangGraph agent to invoke. Must have
            an invoke() method accepting {"messages": [{"role": "user", "content": ...}]}.
        description (str, optional): Human-readable description of the agent's
            capabilities. Used in the tool description for LLM routing decisions.
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

    Last Grunted: 02/03/2026 03:15:00 PM PST
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
    ) -> Command:
        """Execute handoff to agent with isolated context containing only the task description."""
        
        # Validate task_description is not empty or just whitespace
        if not task_description or not task_description.strip():
            error_msg = f"Error: task_description cannot be empty. Please provide a detailed description of what {agent_name} should do."
            logger.error(
                f"Empty task_description provided for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "task_description_length": len(task_description) if task_description else 0,
                    "task_description_preview": task_description[:50] if task_description else None
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
            f"Handing off to {agent_name}",
            extra={
                "agent_name": agent_name,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "task_description_length": len(task_description),
                "task_description_preview": task_description[:200]
            }
        )

        try:
            # Create isolated context with only the task description
            isolated_state = {
                "messages": [{"role": "user", "content": task_description}]
            }

            logger.debug(
                f"Invoking {agent_name} with isolated state",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "state_message_count": len(isolated_state["messages"])
                }
            )

            # Execute the agent with isolated context
            # Note: LangGraph automatically propagates parent graph's config to subagent tools
            result = agent_instance.invoke(isolated_state)

            logger.debug(
                f"Agent {agent_name} invocation completed",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "result_keys": list(result.keys()) if isinstance(result, dict) else None,
                    "has_messages": bool(result.get("messages")) if isinstance(result, dict) else None
                }
            )

            # Extract the final response from the agent
            if not result.get("messages"):
                logger.error(
                    f"No messages returned from {agent_name}",
                    extra={
                        "agent_name": agent_name,
                        "tool_call_id": tool_call_id,
                        "result_type": type(result).__name__,
                        "result_keys": list(result.keys()) if isinstance(result, dict) else None
                    }
                )
                final_content = f"Error: {agent_name} did not return any response"
            else:
                message_count = len(result["messages"])
                logger.debug(
                    f"Agent {agent_name} returned {message_count} message(s)",
                    extra={
                        "agent_name": agent_name,
                        "tool_call_id": tool_call_id,
                        "message_count": message_count
                    }
                )

                # Get the last message from the agent (should be the final response)
                final_message = result["messages"][-1]
                final_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                
                # Log message type for debugging
                message_type = type(final_message).__name__
                logger.debug(
                    f"Extracted final message from {agent_name}",
                    extra={
                        "agent_name": agent_name,
                        "tool_call_id": tool_call_id,
                        "message_type": message_type,
                        "content_type": type(final_content).__name__,
                        "content_length": len(str(final_content)) if final_content else 0
                    }
                )
                
                # Validate that we have actual content
                if not final_content or (isinstance(final_content, str) and not final_content.strip()):
                    logger.error(
                        f"Empty or whitespace-only response from {agent_name}",
                        extra={
                            "agent_name": agent_name,
                            "tool_call_id": tool_call_id,
                            "message_type": message_type,
                            "content_repr": repr(final_content) if final_content else None
                        }
                    )
                    final_content = f"Error: {agent_name} returned an empty response"

            logger.info(
                f"Received response from {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "response_length": len(str(final_content)),
                    "response_preview": str(final_content)[:200] if final_content else None
                }
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
                f"Error during handoff to {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": error_traceback
                },
                exc_info=True
            )
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


def create_isolated_handoff_tools(agents: Dict[str, Dict[str, Any]]) -> list:
    """Create isolated handoff tools for all available agents.

    Bulk factory function that creates handoff tools for each agent in the
    provided dictionary. Used during supervisor graph creation to register
    all agent handoff tools.

    Args:
        agents (Dict[str, Dict[str, Any]]): Dictionary from get_all_agents()
            mapping agent names to configs:
            {
                "agent_name": {
                    "agent": CompiledGraph,  # The agent instance
                    "description": str       # Agent capability description
                }
            }

    Returns:
        list: List of handoff tools, one per agent. Each tool is named
            "handoff_to_{agent_name}" and configured with the agent's
            description for LLM routing decisions.

    Example:
        >>> agents = await get_all_agents()
        >>> tools = create_isolated_handoff_tools(agents)
        >>> [t.name for t in tools]
        ['handoff_to_websearch', 'handoff_to_knowledge_base', ...]

    Note:
        Logs "Created isolated handoff tool for {agent_name}" for each tool
        created, useful for debugging agent initialization.

    Last Grunted: 02/03/2026 03:15:00 PM PST
    """
    handoff_tools = []

    for agent_name, agent_config in agents.items():
        agent_instance = agent_config["agent"]
        description = agent_config.get("description", f"Specialized {agent_name} agent")

        tool = create_isolated_handoff_tool(
            agent_name=agent_name,
            agent_instance=agent_instance,
            description=f"Hand off task to {description.lower()}"
        )

        handoff_tools.append(tool)
        logger.info(f"Created isolated handoff tool for {agent_name}")

    return handoff_tools