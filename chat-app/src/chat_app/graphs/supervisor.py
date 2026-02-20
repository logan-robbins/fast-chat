"""Supervisor graph creation for multi-agent orchestration with LangGraph best practices.

This module implements the supervisor pattern using LangGraph's create_supervisor
with improvements following the LangGraph 2026 documentation:
- Explicit TypedDict state schema (ChatAppState)
- pre_model_hook for context engineering (trim, summarize, inject memories)
- Custom isolated handoff tools (agents receive only task descriptions)
- Checkpointer for durable execution (PostgreSQL or in-memory)

Key components:
- SUPERVISOR_TEMPLATE: Comprehensive system prompt for routing and synthesis
- pre_model_hook: Dynamic prompt injection for date/time, file context, memories
- create_supervisor_graph: Factory function for the compiled supervisor graph

Architecture:
    The supervisor uses custom handoff tools (create_isolated_handoff_tools) that
    pass only the task description to subagents, preventing context bloat and
    ensuring predictable token usage regardless of conversation length.

Context Engineering:
    - Messages are trimmed before each LLM call using pre_model_hook
    - Older messages are summarized to preserve context while reducing tokens
    - User memories are retrieved via semantic search for personalization
    - File context is injected when documents are uploaded

Graph Topology (after compile):
    __start__ → supervisor → {websearch, knowledge_base, code_interpreter, __end__}
    Each agent → supervisor (return to supervisor after execution)

Last Grunted: 02/05/2026 12:00:00 PM UTC
"""
from datetime import datetime
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field

from chat_app.config import NIM_MODEL_NAME, get_settings
from chat_app.memory import format_memories_for_prompt, search_user_memories
from chat_app.services.llm_factory import init_openai_chat_model
from chat_app.state import ChatAppState
from chat_app.summarization import count_message_tokens, maybe_summarize
from chat_app.tools.agent_delegation import create_isolated_handoff_tools

logger = logging.getLogger(__name__)


async def get_relevant_memories(user_id: str, query: str, limit: int = 3) -> str:
    """Retrieve user memories relevant to the current query via semantic search.
    
    Uses the PostgresStore semantic search to find memories that are contextually
    relevant to the user's current message. These memories are then formatted
    for inclusion in the system prompt.
    
    Args:
        user_id: The user's unique identifier for memory namespace.
        query: The current user message to match against stored memories.
        limit: Maximum number of memories to return (default: 3).
        
    Returns:
        str: Formatted memories string for prompt injection, or empty string
            if no memories found or user_id is not provided.
            
    Note:
        Silently handles exceptions to ensure memory lookup failures
        don't break the main conversation flow.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    if not user_id:
        return ""
    
    try:
        memories = await search_user_memories(user_id, query, limit=limit)
        if memories:
            return format_memories_for_prompt(memories)
    except Exception as e:
        logger.warning(
            "Memory lookup failed",
            extra={"user_id": user_id, "error": str(e)}
        )
    
    return ""


# Context window management constants
MAX_CONTEXT_TOKENS: int = 116000  # 128k - 4k system - 8k response buffer
TRIM_THRESHOLD: int = 120000  # Trigger trimming above this threshold

# The system prompt assumes the downstream agent prompts follow 2026 best practices for transparency and citations.
SUPERVISOR_TEMPLATE = """# Role and Identity
You are an intelligent supervisor agent responsible for orchestrating a team of specialized agents to help users accomplish their goals. You serve as the primary interface between the user and your agent team, providing a seamless, helpful experience.

Your core responsibilities:
- Understand user intent deeply, even when requests are ambiguous
- Route tasks to the most appropriate specialized agent(s)
- Synthesize agent responses into coherent, valuable answers
- Maintain conversation context and continuity
- Ensure all claims are properly attributed and cited

# Personality and Communication Style
- Be warm, professional, and genuinely helpful
- Use clear, natural language—avoid jargon unless the user demonstrates technical fluency
- Match your tone to the user's style (formal with formal, casual with casual)
- Be concise by default, but thorough when complexity warrants it
- Ask clarifying questions when user intent is genuinely ambiguous, but make reasonable assumptions when possible rather than asking excessive questions
- Acknowledge uncertainty honestly rather than fabricating information

# Available Agents

<agents>
<agent name="websearch">
<description>Searches the web for current information, market data, and news using Bright Data</description>
<capabilities>
- Real-time web searches for current events, news, and recent information
- Finding documentation, articles, and online resources
- Fact-checking and verification of current information
- Retrieving up-to-date pricing, availability, or status information
</capabilities>
<best_for>
- Questions about current events or recent developments
- Requests requiring information that may have changed recently
- Finding external documentation or resources
- Verifying facts against current sources
</best_for>
</agent>

<agent name="code_interpreter">
<description>Executes Python code for computation and analysis</description>
<capabilities>
- Mathematical calculations (simple arithmetic to complex analysis)
- Data processing, transformation, and analysis
- Generating visualizations and charts
- Statistical computations
- Algorithm implementation and testing
</capabilities>
<best_for>
- Any mathematical computation beyond basic arithmetic
- Data analysis or manipulation tasks
- Creating visualizations from data
- Testing code logic or algorithms
- Processing structured data (CSV, JSON, etc.)
</best_for>
<requires_confirmation>true</requires_confirmation>
</agent>

{rag_agent_def}
</agents>

# Guidelines
1. Route requests to the appropriate agent by providing clear, detailed task descriptions
2. Write complete task descriptions that include all relevant context the agent needs
3. Always provide a non-empty task_description when calling agent tools - empty descriptions will cause errors
4. Assign work to one agent at a time, do not call agents in parallel
5. Do not do any work yourself
6. Each agent will receive only the task description you provide - no conversation history
7. Agents will provide their complete response in a single message

<routing_logic>
## Task Type Matching

| Request Type | Primary Agent | Fallback |
|--------------|---------------|----------|
| Current events, news, web content | websearch | — |
| Math, calculations, data analysis | code_interpreter | — |
| Uploaded documents, internal knowledge | knowledge_base | websearch |
| Code execution, Python scripts | code_interpreter | — |

## Confidence Levels
- HIGH: Clear intent, standard request type
- MEDIUM: Some ambiguity but best guess is clear
- LOW: Multiple valid interpretations, needs clarification

## Ambiguous Requests
When routing is unclear (LOW confidence):
- Use knowledge_base for document-related queries
- Use websearch for external/public information
- Route to "needs_clarification" when genuinely uncertain
</routing_logic>

# Task Delegation

<delegation_guidelines>
## Writing Effective Task Descriptions
Each agent receives ONLY the task description you provide—they have no access to conversation history. Therefore, your task descriptions must be:

1. **Self-contained**: Include all necessary context within the task description
2. **Specific**: Clearly state what information or action is needed
3. **Scoped**: Define boundaries so the agent knows when the task is complete
4. **Structured**: For complex requests, break down into clear components

## Task Description Template
When delegating, include relevant elements:
- **Objective**: What specifically needs to be done
- **Context**: Any background information the agent needs
- **Constraints**: Limitations, filters, or requirements
- **Output Format**: How results should be structured (when relevant)

## Sequential Processing
- Assign work to ONE agent at a time
- Wait for results before deciding on next steps
- Use information from one agent to inform queries to another when needed
</delegation_guidelines>

# Response Synthesis

<synthesis_guidelines>
## Your Role as Synthesizer
Agents gather raw information; YOUR job is to transform it into what the user actually needs. Think of yourself as a skilled analyst who receives data and delivers insight.

## The Synthesis Spectrum
Depending on the user's request, apply the appropriate level of synthesis:

### Level 1: Direct Relay (Minimal Synthesis)
When the user asks for specific data points or factual answers:
- Present the information clearly and directly
- Organize for readability
- Preserve precision and accuracy
- Add brief context only if it aids understanding

### Level 2: Structured Summary (Moderate Synthesis)
When the user asks to "list", "show", or "find" multiple items:
- Group related items logically
- Highlight key patterns or notable entries
- Provide counts or categorizations where helpful
- Maintain connection to source details

### Level 3: Analytical Narrative (Deep Synthesis)
When the user asks to "describe", "summarize", "report on", or "explain" for an audience:
- **Extract themes and patterns**: Don't enumerate items; identify what they collectively represent
- **Write for the audience**: Match detail level to the user's needs
- **Use narrative prose**: Write paragraphs that tell a story, not bullet lists that recite facts
- **Abstract upward**: Identify patterns and themes rather than listing individual items
- **Add insight**: What does this mean? What's the trajectory? What should the audience understand?

## Synthesis Checklist
Before responding, ask yourself:
- [ ] Did I answer what the user actually asked (not just what the agent returned)?
- [ ] Is my response at the right level of abstraction for the user's needs?
- [ ] Have I added value beyond what the raw data provides?
- [ ] Would this response be useful to the intended audience?
</synthesis_guidelines>

# Citation and Attribution

<citation_rules>
## Core Principles
All factual claims derived from agent responses MUST be properly attributed. This ensures transparency, enables verification, and builds trust.

## Citation Format
Use inline numbered citations: [n]

### Inline Citations
- Place citation immediately after the relevant claim
- Use sequential numbering starting from [1]
- Multiple citations for one claim: [1][2] or [1, 2]
- Same source referenced multiple times uses the same number

### Reference Section
Always end your response with a "## References" section containing all cited sources:
```
## References
1. document-name.pdf
2. https://example.com/full-url-preserved
```

## Source Type Formatting
| Source Type | Reference Format |
|-------------|------------------|
| Uploaded documents | Filename only: `report.pdf` |
| Web URLs | Full URL: `https://example.com/path` |

## Citation Rules
1. **PRESERVE** all citations from agent responses—never drop or modify source URLs/filenames
2. **RENUMBER** sequentially when combining responses from multiple agents
3. **CONSOLIDATE** duplicate sources (same URL/document = same citation number)
4. **INDICATE** when information lacks a source: "(source unavailable)" or "(based on general knowledge)"

## Critical Reminders
- NEVER truncate URLs or filenames
- NEVER fabricate citations—if unsure of source, indicate uncertainty
- ALWAYS include the References section, even with a single source
- When no citations are available from agents, clearly indicate information source or uncertainty
</citation_rules>

# Response Formatting

<formatting_guidelines>
## General Principles
- Use the minimum formatting necessary for clarity
- Match format to content type and user expectations
- Prefer prose for explanations; use structure for complex multi-part information

## When to Use Different Formats

### Prose (Default)
- Explanations and analysis
- Summaries and narratives  
- Simple answers and responses
- Recommendations with reasoning

### Bullet Points
- Multiple distinct items with no hierarchy
- Quick reference lists
- Action items or next steps
- Comparison points

### Tables
- Data with multiple attributes per item
- Comparisons across consistent dimensions
- Status reports with uniform fields

### Headers
- Long responses with distinct sections
- Multi-topic responses
- Reference documentation

## Formatting Rules
- Use **bold** sparingly for genuine emphasis only
- Include a blank line before lists and after headers (CommonMark standard)
- Keep bullet points substantive (1-2 sentences minimum)
- Don't use formatting as a substitute for clear thinking
</formatting_guidelines>

# Error Handling

<error_handling>
## Agent Failures
If an agent returns an error or empty result:
1. Acknowledge the issue briefly to the user
2. Explain what was attempted
3. Offer alternatives:
   - Try a different search query or approach
   - Use an alternative agent if appropriate
   - Ask the user for additional information that might help
4. Never pretend an error didn't occur or fabricate results

## Insufficient Information
If agent results don't fully answer the user's question:
1. Provide what information is available
2. Clearly indicate what couldn't be found
3. Suggest next steps or alternative approaches

## Ambiguous Results
If results are unclear or contradictory:
1. Present the information with appropriate uncertainty
2. Note the ambiguity or contradiction
3. Recommend verification if the information is critical
</error_handling>

{rag_instructions}

# Final Checklist
Before sending any response, verify:
- [ ] Did I understand and address the user's actual question?
- [ ] Are all factual claims properly cited?
- [ ] Is the response at the appropriate level of detail?
- [ ] Have I synthesized (not just reorganized) agent outputs?
- [ ] Is the formatting appropriate and not excessive?
- [ ] Is the References section complete and accurate?
"""

RAG_AGENT_DEF = """<agent name="knowledge_base">
<description>Searches uploaded documents and performs RAG queries</description>
<capabilities>
- Semantic search over uploaded documents
- Answering questions based on document content
- Finding specific information within files
</capabilities>
<best_for>
- Questions about uploaded documents
- Finding information in PDFs, documents, or files
- RAG (Retrieval-Augmented Generation) queries
</best_for>
</agent>"""

RAG_INSTRUCTIONS = """# IMPORTANT: File and Knowledge Source Handling Instructions

The user has uploaded documents to this conversation. You MUST use the handoff_to_knowledge_base tool 
to search these documents before answering any question that could be answered by the uploaded files.

Rules:
- ANY question referencing "documents", "uploaded", "files", "based on", or specific document names 
  MUST be delegated to the knowledge_base agent using handoff_to_knowledge_base
- DO NOT try to answer document-related questions from your own knowledge
- DO NOT say "the documents don't contain..." without FIRST calling handoff_to_knowledge_base to search
- If the user asks about uploaded content, call handoff_to_knowledge_base with a clear search task description
- If knowledge_base returns no results, THEN you may say the information wasn't found"""


async def pre_model_hook(
    state: ChatAppState,
    config: RunnableConfig
) -> Dict[str, Any]:
    """Construct dynamic system prompt and trim messages before each LLM call.
    
    This hook implements LangGraph's recommended pattern for managing message
    history. It is called before every supervisor LLM invocation to:
    
    1. Trim messages to fit within the context window
    2. Summarize older messages that were trimmed
    3. Inject runtime context (date/time, file context, memories)
    4. Build the complete system prompt with relevant agent definitions
    
    Context Engineering Strategy:
        - Messages exceeding TRIM_THRESHOLD trigger summarization
        - Summary is prepended to provide context for trimmed history
        - User memories are retrieved via semantic search
        - File context enables RAG agent when documents are present
    
    Args:
        state: Current graph state containing messages and metadata.
        config: RunnableConfig with configurable options including:
            - file_context: Markdown-formatted file summaries
            - user_id: User identifier for memory retrieval
        
    Returns:
        dict: State update containing:
            - llm_input_messages: Messages to send to the LLM
            - conversation_summary: Updated summary (if changed)
            
    Note:
        This function never mutates state directly - it returns updates
        that LangGraph applies immutably.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract current state
    messages: List[BaseMessage] = state.get("messages", [])
    existing_summary: str = state.get("conversation_summary", "")
    new_summary: str = existing_summary
    
    # Trim messages if they exceed the context threshold
    if messages:
        total_tokens = count_message_tokens(messages)
        
        if total_tokens > TRIM_THRESHOLD:
            logger.info(
                "Trimming messages",
                extra={
                    "original_tokens": total_tokens,
                    "threshold": TRIM_THRESHOLD,
                    "message_count": len(messages)
                }
            )
            messages, new_summary = await maybe_summarize(
                messages,
                existing_summary=existing_summary,
                max_tokens=MAX_CONTEXT_TOKENS,
                summarize_threshold=TRIM_THRESHOLD
            )
    
    # Extract configurable values
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    file_context_md: Optional[str] = configurable.get("file_context")
    vector_collections: list = configurable.get("vector_collections", [])
    user_id: Optional[str] = configurable.get("user_id")
    
    # Build RAG agent definition and pre-fetch relevant documents
    rag_agent_def = ""
    rag_instructions = ""
    
    if file_context_md or vector_collections:
        rag_agent_def = RAG_AGENT_DEF
        rag_instructions = RAG_INSTRUCTIONS
        if file_context_md:
            rag_instructions += f"\n\n---\n\n# File Context\n{file_context_md}"
        
        # Pre-fetch relevant document chunks for the user's question.
        # This injects RAG context directly so the supervisor can answer
        # without needing to delegate -- more reliable than tool routing.
        if vector_collections and messages:
            last_user_msg = _extract_last_human_message(messages)
            if last_user_msg:
                try:
                    from chat_app.services.bff_client import search_documents
                    search_results = await search_documents(
                        query=last_user_msg,
                        collections=vector_collections,
                        limit=8,
                    )
                    if search_results:
                        doc_context = "\n\n# Retrieved Document Context\n"
                        doc_context += "The following excerpts were retrieved from the user's uploaded documents:\n\n"
                        for i, result in enumerate(search_results, 1):
                            doc_context += f"[{i}] (from {result.filename}, relevance: {result.score:.0%})\n"
                            doc_context += f"{result.text}\n\n"
                        rag_instructions += doc_context
                        logger.info(
                            "pre_model_hook.rag_prefetch",
                            results=len(search_results),
                            top_score=search_results[0].score if search_results else 0,
                        )
                except Exception as e:
                    logger.warning("pre_model_hook.rag_prefetch_failed: %s", str(e))
    
    # Build intent context if classification is available
    intent_context = ""
    user_intent = state.get("user_intent")
    if user_intent:
        intent_context = (
            f"\n\n# Current Intent Classification\n"
            f"Intent: {user_intent.get('primary_intent', 'unknown')}\n"
            f"Confidence: {user_intent.get('confidence', 'unknown')}\n"
            f"Description: {user_intent.get('description', '')}"
        )
    
    # Build summary context for older messages
    summary_context = ""
    if new_summary:
        summary_context = f"\n\n# Conversation Summary (Earlier Messages)\n{new_summary}\n"
    
    # Retrieve relevant user memories via semantic search
    memory_context = ""
    if user_id and messages:
        last_user_msg = _extract_last_human_message(messages)
        if last_user_msg:
            memory_context_str = await get_relevant_memories(user_id, last_user_msg, limit=3)
            if memory_context_str:
                memory_context = f"\n\n{memory_context_str}\n"
    
    # Assemble the complete system prompt
    system_prompt = SUPERVISOR_TEMPLATE.format(
        rag_agent_def=rag_agent_def,
        rag_instructions=rag_instructions
    )
    
    full_system_content = (
        f"Current Date/Time: {dt_string}"
        f"{intent_context}{memory_context}{summary_context}"
        f"\n\n{system_prompt}"
    )
    
    output_messages = [SystemMessage(content=full_system_content)]
    output_messages.extend(messages)
    
    # Return state update (never mutate state directly)
    return {
        "llm_input_messages": output_messages,
        "conversation_summary": new_summary
    }


def _extract_last_human_message(messages: List[BaseMessage]) -> Optional[str]:
    """Extract the content of the last human message from a message list.
    
    Args:
        messages: List of LangChain message objects.
        
    Returns:
        str | None: Content of the last human message, or None if not found.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            return msg.content
    return None


def get_checkpointer():
    """Get the configured checkpointer based on settings.
    
    Returns either a PostgresSaver or InMemorySaver based on
    the CHECKPOINTER_TYPE setting. PostgreSQL is recommended for
    production as it supports persistence, time-travel debugging,
    and up to 1GB state per checkpoint.
    
    Returns:
        BaseCheckpointSaver: Configured checkpointer instance.
            - PostgresSaver: For production with DATABASE_URL
            - InMemorySaver: For development/testing
            
    Raises:
        ValueError: If CHECKPOINTER_TYPE=postgres but DATABASE_URL is not set.
        
    Last Grunted: 02/04/2026 06:30:00 PM PST
    """
    settings = get_settings()
    
    if settings.checkpointer_type == "postgres":
        if not settings.database_url:
            raise ValueError(
                "DATABASE_URL must be set when CHECKPOINTER_TYPE=postgres. "
                "Set DATABASE_URL=postgresql://user:pass@host:5432/db"
            )
        
        try:
            checkpointer = PostgresSaver.from_conn_string(settings.database_url)
            # Setup the checkpointer tables on first use
            checkpointer.setup()
            logger.info("PostgreSQL checkpointer initialized")
            return checkpointer
        except Exception as e:
            logger.error(
                "Failed to initialize PostgreSQL checkpointer",
                extra={"error": str(e), "database_url_host": settings.database_url.split("@")[-1] if "@" in settings.database_url else "unknown"}
            )
            raise
    else:
        # Default to in-memory for development
        logger.info("Using in-memory checkpointer (development mode)")
        return InMemorySaver()


def create_supervisor_graph(agents: Dict[str, Dict[str, Any]]):
    """Create and compile the supervisor graph for multi-agent orchestration.
    
    Manually constructs the supervisor graph to ensure robust context engineering
    and system prompt injection, replacing the library helper which had
    issues with dynamic prompts.
    
    Architecture:
        __start__ -> supervisor -> {agents (wrapped)} -> supervisor
        
    Args:
        agents: Dictionary mapping agent names to configs.
            
    Returns:
        CompiledGraph: Compiled supervisor graph.
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command
    from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
    from chat_app.utils.messages import content_to_text
    
    # Create custom handoff tools for isolated context
    handoff_tools = create_isolated_handoff_tools(agents)
    settings = get_settings()
    
    # Configure model
    model_name = NIM_MODEL_NAME or settings.default_model or "gpt-4o"
    supervisor_model = init_openai_chat_model(
        model_name,
        temperature=0.0,
        streaming=True,
        tags=["supervisor"],
    ).bind_tools(handoff_tools) # Bind routing tools
    
    logger.info(
        "Creating supervisor graph (manual)",
        extra={
            "agent_count": len(agents),
            "agents": list(agents.keys()),
            "model": model_name,
        }
    )

    # Define the supervisor node logic
    async def supervisor_node(state: ChatAppState, config: RunnableConfig) -> Command[Literal[tuple(list(agents.keys()) + ["__end__"])]]:
        # 1. Run context engineering to get dynamic system prompt
        context_update = await pre_model_hook(state, config)
        
        # Use the injected messages if available, otherwise fall back to state messages
        # This fixes the issue where system prompt was ignored
        messages_to_send = context_update.get("llm_input_messages", state["messages"])
        
        # 2. Invoke the model
        response = await supervisor_model.ainvoke(messages_to_send, config)
        
        # 3. Handle routing based on tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Map tool calls to agent nodes
            # The handoff tools are named like "handoff_to_websearch"
            # We assume the tool name maps to the agent name via registry
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            
            # Extract agent name from tool name (e.g. handoff_to_websearch -> websearch)
            # This relies on the convention in create_isolated_handoff_tools
            if tool_name.startswith("handoff_to_"):
                agent_name = tool_name.replace("handoff_to_", "")
                
                # Get the task description from arguments
                task = tool_call["args"].get("task_description", "")
                
                # Log the routing
                logger.info(f"Routing to {agent_name}", extra={"task": task})
                
                # Emit "Agent Handoff" status update for the routing decision
                from chat_app.status_streaming import emit_agent_handoff
                emit_agent_handoff(agent_name, task)
                
                # Return Command to go to agent node
                # The agent node will receive the current state.
                # Since we want isolated context, the sub-agents must be wrapped 
                # or configured to ignore history. 
                # In this architecture, we update the state with the *intent* 
                # and let the agent node handle it?
                # LangGraph `Command(goto=agent_name, update={"...": ...})`
                
                return Command(
                    goto=agent_name,
                    update={
                        "metadata": {
                            **state.get("metadata", {}), 
                            "last_agent_called": agent_name
                        },
                        # Append the assistant message (tool call) to history
                        "messages": [response]
                    }
                )
        
        # 4. No tool call -> Final response
        return Command(
            goto=END,
            update={"messages": [response]}
        )

    # Helper to create isolated agent wrapper nodes
    def create_agent_wrapper(agent_name, agent_graph):
        """Build a LangGraph node that invokes an isolated agent with a single task.

        The returned coroutine extracts the task description from the supervisor's
        most recent tool call, sends it as a standalone message to the agent, and
        wraps the agent's final response as a `ToolMessage` so the supervisor can
        consume it while maintaining the handoff metadata.
        """
        async def wrapped_agent_node(state: ChatAppState, config: RunnableConfig) -> Command[Literal["supervisor"]]:
            # Extract task from the correct tool call in the last message
            last_message = state["messages"][-1]
            task = ""
            tool_call_id = None
            
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    if tc["name"] == f"handoff_to_{agent_name}":
                        task = tc["args"].get("task_description", "")
                        tool_call_id = tc["id"]
                        break
            
            if not task:
                task = "Continue previous task" # Fallback
                
            # Invoke agent with ISOLATED context (just the task)
            # We treat the task as a user message to the agent
            agent_input = {"messages": [HumanMessage(content=task)]}
            
            # Stream status from agent if needed, but agent handles its own tool status updates
            agent_response = await agent_graph.ainvoke(agent_input, config)
            
            # Extract final response content
            final_content = "No response from agent"
            if "messages" in agent_response and agent_response["messages"]:
                raw_content = agent_response["messages"][-1].content
                final_content = content_to_text(raw_content)
                
            # Return result as a ToolMessage to satisfy the supervisor's tool call
            return Command(
                goto="supervisor",
                update={
                    "messages": [
                        ToolMessage(
                            content=str(final_content),
                            tool_call_id=tool_call_id,
                            name=f"handoff_to_{agent_name}"
                        )
                    ]
                }
            )
        return wrapped_agent_node

    # Build the graph
    workflow = StateGraph(ChatAppState)
    
    # Add supervisor node
    workflow.add_node("supervisor", supervisor_node)
    
    # Add wrapped agent nodes
    for name, config_data in agents.items():
        # The agent graph itself is the node
        workflow.add_node(name, create_agent_wrapper(name, config_data["agent"]))
        # Ensure agents return to supervisor
        # (This is usually handled by the agent returning a Command or edge)
        # But here we might need to enforce it. 
        # For simplicity, we assume agents finish and we want to route back to supervisor?
        # Or does the supervisor act as a single-turn router? 
        # "ReAct" style: Supervisor -> Agent -> Supervisor
        
        
    # Start at supervisor
    workflow.add_edge(START, "supervisor")
    
    # Compile
    checkpointer = get_checkpointer()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Supervisor graph compiled successfully")
    return compiled_graph
