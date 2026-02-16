# Fast-Chat Backend Platform SPEC (Enterprise, Extensible, Frontend-Agnostic)

## 1) Product Intent

Fast-Chat should evolve into an enterprise-ready backend platform for building ChatGPT/OpenWebUI-class assistants with open-source extensibility, strict API contracts, and pluggable runtime components. The backend is the product. UI implementations (Chainlit or any other frontend) are reference clients rather than the core value layer.

This specification translates README-defined capabilities into explicit functional requirements, then extends them into a modern, modular architecture centered around:

- **LiteLLM as the only model gateway** (no local model backends in product scope).
- **Pluggable retrieval and memory** (multiple vector DBs and retrieval backends).
- **Optional offloading of tools/workflows to MCP servers and A2A-capable remote agents**.
- **Composable agent graphs** with clear extension points and runtime policy control.
- **Strict streaming/event contracts** for frontend interoperability.

---

## 2) Scope and Principles

### 2.1 In Scope

1. OpenAI-compatible and Responses-style APIs for chat and agent workflows.
2. Streaming-first architecture with token, status, and usage events.
3. Multi-agent orchestration with supervisor + specialist agents.
4. RAG and document ingestion with pluggable storage/retrieval.
5. Conversation persistence, checkpointing, and memory.
6. Enterprise features: authn/authz, observability, auditing, rate limiting, policy governance.
7. Extensibility model for MCP, A2A, vector DB adapters, and agent plugins.

### 2.2 Out of Scope

1. Building a custom model gateway (LiteLLM is mandatory).
2. Local model serving (Ollama/vLLM/local GPU as first-class platform options).
3. Frontend-specific UI requirements beyond protocol compatibility.

### 2.3 Design Principles

- **Backend-first contracts**: every behavior exposed via stable APIs/events.
- **Pluggability over hardcoding**: adapters and capabilities discovered by registry.
- **Cloud-neutral deployment**: supports self-hosted enterprise environments.
- **Policy-driven operations**: governance and safety enforced centrally.
- **Graceful degradation**: missing optional dependencies do not break core chat.

---

## 3) Functional Requirements Derived from README

This section separates and formalizes functional requirements already implied in `README.md`.

### 3.1 Platform and Architecture Requirements

- **FR-001**: System SHALL implement a multi-component architecture including frontend, BFF API, orchestration service, and supporting data services.
- **FR-002**: BFF layer SHALL provide OpenAI-compatible endpoints for chat, responses, files, and models.
- **FR-003**: Orchestrator SHALL support multi-agent routing with supervisor handoffs.
- **FR-004**: Backend SHALL support RAG over user-uploaded documents.
- **FR-005**: Backend SHALL support code-interpreter style tool execution.
- **FR-006**: Backend SHALL support web search via external provider integration.

### 3.2 Streaming and Interaction Requirements

- **FR-007**: Chat endpoints SHALL support SSE token streaming.
- **FR-008**: Streaming SHALL include optional `status` events with user-friendly progress semantics.
- **FR-009**: Streaming SHALL include optional `usage` events with prompt/completion/total token metrics and context utilization.
- **FR-010**: Streaming protocol SHALL terminate with `[DONE]` sentinel semantics.
- **FR-011**: Event schema SHALL support status subtypes (`thinking`, `agent_handoff`, `tool_start`, `tool_progress`, `tool_complete`, `error`).

### 3.3 Context Management Requirements

- **FR-012**: Backend SHALL implement token counting for request context management.
- **FR-013**: Backend SHALL enforce conversation trimming above threshold, with target reduction.
- **FR-014**: Backend SHALL summarize older history when trimming.
- **FR-015**: Agent handoffs SHALL provide isolated task-centric context rather than full history.
- **FR-016**: Backend SHALL provide personalized memory retrieval and injection into orchestration context.

### 3.4 Persistence and Data Requirements

- **FR-017**: BFF SHALL persist threads and messages.
- **FR-018**: Orchestrator SHALL persist checkpoints/state for resumability.
- **FR-019**: Backend SHALL persist embeddings/vector data for retrieval.
- **FR-020**: Backend SHALL support cancellation signaling for in-flight/background operations.
- **FR-021**: API SHALL support thread listing, update, deletion, message retrieval, and forking.

### 3.5 Document and Retrieval Requirements

- **FR-022**: Backend SHALL support file upload and retrieval-linked processing.
- **FR-023**: Document processor SHALL support text extraction from PDF/PPTX/DOCX/XLSX and text files.
- **FR-024**: Document processing SHALL include fallback for scanned or layout-sensitive files (vision extraction path).
- **FR-025**: Backend SHALL expose semantic search endpoint over collections.

### 3.6 API Surface Requirements

- **FR-026**: Backend SHALL expose `/v1/chat/completions` and `/v1/responses` style APIs.
- **FR-027**: Backend SHALL expose response lifecycle endpoints (retrieve, delete, cancel, compact).
- **FR-028**: Backend SHALL expose `/v1/models` model discovery endpoint.
- **FR-029**: Backend SHALL expose `/v1/files` management endpoints.
- **FR-030**: Backend SHALL expose health endpoints per service.
- **FR-031**: Orchestrator SHALL expose interrupt/resume endpoints for HITL flows.

### 3.7 Operational Requirements

- **FR-032**: Platform SHALL be deployable by Docker Compose for local/dev parity.
- **FR-033**: Platform SHALL support structured logging modes for production/dev.
- **FR-034**: Platform SHALL support OAuth/SSO integration in reference frontend without coupling backend contracts to one UI.

---

## 4) Future-State Product Requirements (Enterprise + Extensibility)

### 4.1 Model Gateway Standardization (LiteLLM-Only)

- **PR-001**: All model invocations SHALL route through LiteLLM.
- **PR-002**: Backend SHALL remove/disable local model toggles from product defaults.
- **PR-003**: Model configuration SHALL be centralized via `ModelRegistry` backed by LiteLLM providers/models.
- **PR-004**: Per-tenant policy SHALL control allowed model families, max context, tool-calling support, and budget limits.
- **PR-005**: Runtime SHALL support model fallback chains and failover policies defined declaratively.

### 4.2 Extensible Retrieval and Knowledge Architecture

- **PR-006**: Retrieval layer SHALL expose a provider interface with interchangeable vector backends (e.g., Chroma, PGVector, Pinecone, Weaviate, Qdrant, Milvus, Azure AI Search, Elasticsearch).
- **PR-007**: Indexing pipeline SHALL support per-collection chunking/embedding/reranking policies.
- **PR-008**: Retrieval pipeline SHALL support hybrid search (vector + lexical) and optional rerank.
- **PR-009**: RAG execution SHALL support local adapter mode and remote offload mode (MCP/A2A retriever agents).
- **PR-010**: Retrieval traces SHALL be observable (query, latency, hit counts, rerank decisions, source citations).

### 4.3 MCP Integration

- **PR-011**: Platform SHALL support registering multiple MCP servers as tool/retrieval providers.
- **PR-012**: MCP capabilities SHALL be introspected and mapped into tool catalog entries.
- **PR-013**: Tool execution policy SHALL restrict MCP server access by tenant, workspace, and role.
- **PR-014**: MCP failures SHALL degrade gracefully with explicit status/error events.
- **PR-015**: MCP tool outputs SHALL be normalized into canonical tool-result schema.

### 4.4 A2A (Agent-to-Agent) Interoperability

- **PR-016**: Platform SHALL support remote specialist agents via A2A contracts.
- **PR-017**: Supervisor SHALL route tasks to local graph nodes or remote A2A agents based on policy/capability.
- **PR-018**: A2A exchanges SHALL include capability discovery, task envelope, trace context, and result schema.
- **PR-019**: A2A execution SHALL support sync and async modes with callback/webhook or polling completion.
- **PR-020**: A2A trust policy SHALL enforce identity verification and signed agent manifests.

### 4.5 Extensible Agent Graph Runtime

- **PR-021**: Agent graphs SHALL be declarative (YAML/JSON/Python config) and hot-loadable by version.
- **PR-022**: Node types SHALL include LLM, tool, retriever, planner, evaluator, guardrail, and router nodes.
- **PR-023**: Runtime SHALL support graph version pinning per tenant/workspace.
- **PR-024**: Runtime SHALL support interruption and resume with deterministic checkpoint recovery.
- **PR-025**: Runtime SHALL expose graph execution events for debugging and analytics.

### 4.6 Enterprise Controls

- **PR-026**: Authentication SHALL support OIDC/SAML-compatible enterprise IdPs.
- **PR-027**: Authorization SHALL provide RBAC plus optional ABAC policy hooks.
- **PR-028**: Sensitive data controls SHALL include PII redaction options, encryption at rest, and key management integration.
- **PR-029**: Audit logs SHALL capture model/tool/retrieval invocations and admin config changes.
- **PR-030**: Multi-tenancy SHALL enforce data isolation across threads, files, indexes, and memories.

### 4.7 Reliability and SRE Requirements

- **PR-031**: Platform SHALL define SLOs for latency, availability, and stream reliability.
- **PR-032**: Distributed tracing SHALL span BFF, orchestrator, retrieval, LiteLLM calls, MCP, and A2A interactions.
- **PR-033**: Circuit breakers and retry policies SHALL be configurable per external dependency.
- **PR-034**: Background jobs SHALL be idempotent and resumable.
- **PR-035**: Release process SHALL support zero/low-downtime migration for schema and graph versions.

---

## 5) Target Architecture (Backend-Centric)

## 5.1 Logical Planes

1. **Experience Plane (API/BFF)**
   - OpenAI-compatible APIs.
   - Session/thread abstraction.
   - Streaming event multiplexer.

2. **Reasoning Plane (Orchestration Runtime)**
   - Supervisor and graph runtime.
   - Agent selection, policy checks, execution control.
   - HITL interrupt/resume.

3. **Knowledge Plane**
   - Ingestion pipeline.
   - Retrieval adapters.
   - Memory service.

4. **Integration Plane**
   - LiteLLM client.
   - MCP bridge.
   - A2A bridge.

5. **Governance Plane**
   - Authn/authz.
   - Policy engine.
   - Auditing and compliance.

6. **Operations Plane**
   - Telemetry, tracing, metrics.
   - Control APIs for runtime configuration.

### 5.2 Service Boundaries

- `gateway-api`: public API, streaming, auth, tenancy context, request shaping.
- `orchestrator`: graph runtime, supervisor policies, execution/cancellation.
- `knowledge-service`: ingestion + retrieval abstractions + citation assembly.
- `memory-service`: long-term semantic memory and profile facts.
- `integration-service` (optional split): MCP/A2A connectors and normalization.
- `control-plane`: config, model catalog, policy bundles, graph registry.

---

## 6) Canonical Interfaces

### 6.1 Model Interface (LiteLLM)

```yaml
ModelCallRequest:
  tenant_id: string
  model_alias: string
  mode: [chat, responses, embedding]
  messages_or_input: object
  tools: []
  response_format: object?
  max_tokens: int?
  temperature: float?
  metadata: object?

ModelCallResponse:
  output: object
  usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
  finish_reason: string
  provider: string
  model_resolved: string
  latency_ms: int
```

### 6.2 Retriever Interface

```yaml
RetrieveRequest:
  tenant_id: string
  workspace_id: string
  query: string
  top_k: int
  filters: object?
  strategy: [vector, hybrid, lexical]
  rerank: bool

RetrieveResponse:
  hits:
    - doc_id: string
      chunk_id: string
      score: float
      text: string
      metadata: object
      source_uri: string?
  diagnostics:
    vector_db: string
    latency_ms: int
    total_candidates: int
```

### 6.3 Tool Interface (Local/MCP/A2A-normalized)

```yaml
ToolInvokeRequest:
  tool_id: string
  execution_mode: [local, mcp, a2a]
  args: object
  trace_id: string

ToolInvokeResponse:
  status: [ok, partial, error]
  output: object
  artifacts: []
  error: object?
  telemetry: object
```

---

## 7) API and Event Contract Evolution

### 7.1 OpenAI-Compatible Endpoints (Must Keep)

- `/v1/chat/completions`
- `/v1/responses`
- `/v1/models`
- `/v1/files`
- Thread and message endpoints
- Search endpoints

### 7.2 New Control and Registry Endpoints

- `/v1/admin/model-registry`
- `/v1/admin/retrieval-providers`
- `/v1/admin/mcp/servers`
- `/v1/admin/a2a/agents`
- `/v1/admin/graph-versions`
- `/v1/admin/policies`

### 7.3 Event Envelope Standard

All SSE events SHOULD conform to:

```json
{
  "event_id": "uuid",
  "event_type": "token|status|usage|tool|agent|error|complete",
  "timestamp": "RFC3339",
  "trace_id": "...",
  "thread_id": "...",
  "payload": {}
}
```

Back-compat mode SHALL still emit legacy default token chunks and named `status`/`usage` events for existing clients.

---

## 8) Retrieval and RAG Blueprint

### 8.1 Ingestion Pipeline Stages

1. File receive and malware/type validation.
2. Content extraction (format-specific strategy).
3. Optional OCR/vision extraction.
4. Chunking policy application.
5. Embedding via LiteLLM-compatible embedding model routing.
6. Index write via selected vector provider adapter.
7. Metadata indexing and provenance capture.
8. Optional summarization and synthetic QA generation for recall quality.

### 8.2 Query Pipeline Stages

1. Query rewrite and intent classification.
2. Retriever selection (local adapter or MCP/A2A offload).
3. Primary retrieval (vector/hybrid).
4. Optional rerank/citation filtering.
5. Context assembly with budget-aware packing.
6. Response generation with citation binding.

### 8.3 Vector Provider Adapter Contract

Each adapter MUST implement:

- `create_collection`
- `upsert_chunks`
- `query`
- `delete_by_doc`
- `delete_collection`
- `health`
- `capabilities` (hybrid support, metadata filter support, max batch)

---

## 9) MCP and A2A Execution Model

### 9.1 Capability Registry

- Unified registry entry for tools/retrievers/agents:
  - `id`, `type`, `transport`, `auth`, `tenant_scope`, `capabilities`, `cost_profile`, `sla_profile`.

### 9.2 Routing Policy

Supervisor routing decisions should consider:

- Capability match score.
- Tenant policy allow/deny.
- Cost and latency budget.
- Data residency constraints.
- Reliability score and recent error rates.

### 9.3 Execution Safety

- Sandboxed execution contexts where applicable.
- Request/response schema validation.
- Timeout + cancellation propagation.
- Deterministic retry semantics for idempotent tasks.

---

## 10) State, Memory, and Conversation Semantics

### 10.1 Thread State

- Thread metadata: user/tenant/workspace/model policy.
- Message lineage: parent references enabling branch/fork semantics.
- Checkpoint IDs for graph resume.

### 10.2 Memory Types

1. **Profile memory**: stable user preferences.
2. **Episodic memory**: session/task outcomes.
3. **Semantic memory**: extracted factual embeddings.

Memory writes SHOULD pass through confidence and sensitivity filters; all memory retrievals MUST include source and confidence metadata.

### 10.3 Context Budgeting Policy

- Hard max prompt budget per model alias.
- Reserved completion budget.
- Priority tiers: system > policy > recent turns > memory > retrieval.
- Automatic summarization and drop strategy when over budget.

---

## 11) Security, Compliance, and Governance

### 11.1 Identity and Access

- JWT/OIDC trust with tenant claims.
- Service-to-service mTLS or signed tokens.
- Scoped API keys for machine clients.

### 11.2 Data Governance

- Encryption at rest for relational/vector/object stores.
- KMS integration for key rotation.
- Tenant-scoped deletion workflows (right to erasure).
- Configurable retention policies by data class.

### 11.3 Auditability

Audit events MUST include:

- Who invoked what (principal + tenant).
- Which model/tool/retriever/agent was used.
- Input/output hashes (or redacted summaries).
- Cost and latency metrics.
- Policy decisions and overrides.

---

## 12) Observability and Reliability

### 12.1 Metrics

- Request throughput and p95/p99 latency per endpoint.
- Stream-start latency and stream interruption rates.
- Token and cost metrics per model/provider/tenant.
- Tool/retriever success rates.
- Queue depth and retry counts.

### 12.2 Tracing

- End-to-end trace propagation across BFF → orchestrator → LiteLLM/retrieval/MCP/A2A.
- Span attributes for model, tool, retrieval provider, policy rule IDs.

### 12.3 Reliability Patterns

- Bulkheads for provider classes.
- Exponential backoff with jitter.
- Circuit breakers for unstable integrations.
- Dead-letter queue for failed async tasks.

---

## 13) Deployment and Configuration Model

### 13.1 Configuration Layers

1. Global defaults.
2. Environment override.
3. Tenant policy overlay.
4. Workspace/project overrides.

### 13.2 Required Configuration Domains

- LiteLLM base configuration and provider credentials.
- Model alias registry and fallback chains.
- Retrieval provider definitions and credentials.
- MCP server registrations.
- A2A agent registrations.
- Security policies and retention settings.

### 13.3 Removal/Deprecation Policy

- Local model environment toggles are deprecated for enterprise profile.
- Backward compatibility maintained for one minor cycle with explicit warnings.

---

## 14) Reference Implementation Plan

### Phase 1: Contract Stabilization

- Lock SSE and API schemas.
- Introduce canonical event envelope (back-compat mode).
- Add model registry abstraction in front of LiteLLM.

### Phase 2: Retrieval Abstraction

- Create vector adapter interface.
- Migrate existing Chroma integration behind adapter.
- Add one additional provider (PGVector or Qdrant) as proof.

### Phase 3: Integration Fabric

- Add MCP registry and executor.
- Add A2A agent registry + invocation client.
- Integrate routing policy engine in supervisor.

### Phase 4: Enterprise Hardening

- Multi-tenant isolation checks.
- RBAC/ABAC policy enforcement.
- Full audit trail and cost analytics.

### Phase 5: Scale and Ops

- SLO dashboards.
- Chaos/failure injection tests for provider outages.
- Versioned graph deployment rollout strategy.

---

## 15) Acceptance Criteria

### 15.1 Compatibility

- Existing OpenAI-compatible clients continue working without changes.
- Existing status/usage stream consumers continue working.

### 15.2 Extensibility

- New vector backend can be added by implementing adapter contract only.
- New MCP server can be registered without code modifications.
- New A2A remote specialist can be integrated via registry and policy config.

### 15.3 Enterprise Readiness

- Tenant data isolation tests pass.
- Audit log completeness passes required control checks.
- Role-based restrictions verified for tools and model access.

### 15.4 Operational Confidence

- SLOs defined and monitored.
- Recovery drill demonstrates resume after orchestrator interruption.
- Provider failover works for at least one model fallback chain.

---

## 16) Risks and Mitigations

1. **Risk**: Over-complex plugin architecture slows delivery.
   - **Mitigation**: Define thin, stable interfaces first; ship one reference implementation per extension point.

2. **Risk**: Event contract fragmentation across frontends.
   - **Mitigation**: Publish versioned SSE schema and conformance tests.

3. **Risk**: Enterprise policy requirements conflict with open-source defaults.
   - **Mitigation**: Profile-based defaults (`oss`, `enterprise`) with clear behavior differences.

4. **Risk**: External integration reliability (MCP/A2A/provider outages).
   - **Mitigation**: Circuit breakers, fallback agents, and transparent user-facing status events.

---

## 17) Definition of Done for this SPEC

This SPEC is considered complete when:

1. README-derived functionality has been converted into explicit requirement IDs.
2. LiteLLM-only model architecture is formalized.
3. Vector DB and retrieval abstraction is fully specified.
4. MCP and A2A integration patterns are defined with policy and safety controls.
5. Extensible graph runtime requirements and rollout phases are documented.
6. Acceptance criteria provide a concrete validation target for engineering and product teams.

