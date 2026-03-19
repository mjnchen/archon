# Archon — Enterprise Agent Framework Requirements

This document defines the 12 core requirements for Archon, organized into three priority tiers.
Archon is built on top of [LiteLLM](https://github.com/BerriAI/litellm) to provide a unified,
provider-agnostic agent framework designed for enterprise production workloads.

---

## Tier 1 — Must-Have

These are non-negotiable for the initial release.

### 1. Observability & Raw Request Logging

**Rationale:** Enterprise deployments require full visibility into every LLM interaction for
debugging, cost attribution, and compliance.

**Key sub-features:**

- **Raw HTTP JSON capture** — Log the exact JSON body, URL, and headers sent to each LLM
  provider. LiteLLM exposes this via `litellm.log_raw_request_response = True` and the
  `model_call_details["raw_request_typed_dict"]` callback field, which provides:
  - `raw_request_api_base` — the provider URL
  - `raw_request_body` — the full JSON request body (post-transformation, provider-specific)
  - `raw_request_headers` — HTTP headers (auth masked)
- **Step-by-step agent trace** — Record each iteration of the agent loop: LLM call, tool
  invocations, tool results, and the LLM's next decision. Each step tagged with a trace ID
  and step index.
- **Latency tracking** — Wall-clock time per step (LLM call latency vs. tool execution latency)
  and total agent run duration.
- **Token usage & cost tracking** — Per-step and per-run token counts and estimated cost,
  leveraging LiteLLM's built-in cost calculation (`response._hidden_params`, callback
  `response_cost`).

**Implementation notes:**

- Register a custom `CustomLogger` subclass with LiteLLM's callback system.
- Store traces in a structured format (JSON lines or a database) for later querying.
- Expose traces via a Python API and optionally a REST endpoint.

---

### 2. Tool / Function Calling (OpenAI + MCP)

**Rationale:** Agents act on the world through tools. Enterprise agents need to support both the
OpenAI function-calling standard and MCP (Model Context Protocol) for interoperability.

**Key sub-features:**

- **OpenAI-style tools** — Define tools as `{"type": "function", "function": {"name", "parameters", "description"}}`.
  LiteLLM normalizes all provider responses to this format, so tool parsing is provider-agnostic.
- **MCP tools** — Connect to MCP servers (Zapier, Jira, Linear, custom). LiteLLM's
  `litellm/experimental_mcp_client/` handles the MCP transport.
- **Plain Python callables** — Register any Python function as a tool with automatic JSON Schema
  generation from type hints.
- **Tool permission scoping** — Define which agents or tenants can invoke which tools. Enforced
  before execution.
- **Execution timeouts** — Configurable per-tool timeout with automatic cancellation.
- **Retry logic** — Configurable retries for transient tool failures.
- **Sandboxed execution** — Option to run untrusted tool code in an isolated subprocess or
  container.

**Provider differences (handled by LiteLLM):**

| Aspect          | OpenAI                     | Anthropic                        | Gemini                            |
|-----------------|----------------------------|----------------------------------|-----------------------------------|
| Tool definition | `function.parameters`      | `input_schema`                   | `function_declarations`           |
| Args format     | JSON string                | Parsed object                    | Parsed object                     |
| Tool call field | `tool_calls` array         | `tool_use` content blocks        | `functionCall` in parts           |
| Result role     | `role: "tool"`             | `tool_result` inside user msg    | `functionResponse` in user parts  |
| ID linkage      | `tool_call_id`             | `tool_use_id`                    | By name (no IDs)                  |

Archon operates on the unified OpenAI format; LiteLLM handles all transformations.

---

### 3. Agent Loop Patterns

**Rationale:** Different tasks require different execution strategies. The framework must support
the most common agent loop patterns out of the box.

**Key sub-features:**

- **ReAct loop** (Reason → Act → Observe → Repeat) — The core loop: the LLM reasons about
  what to do, calls a tool, observes the result, and repeats until it produces a final answer
  or hits a stop condition.
- **Agent handover** — One agent can delegate control to another agent mid-conversation. The
  receiving agent inherits the conversation context (or a subset of it). Useful for
  specialized sub-agents (e.g., a "research agent" hands off to a "writing agent").
- **Max iterations cap** — Hard limit on loop iterations to prevent runaway agents. Configurable
  per agent.
- **Budget cap** — Hard limit on total token spend or dollar cost per agent run. The agent is
  stopped if the budget is exceeded.
- **Graceful termination** — When a cap is hit, the agent produces a summary of what it
  accomplished and what remains, rather than silently dying.

**Implementation notes:**

- The ReAct loop is the default. Each iteration: call LLM → check for tool calls → execute
  tools → append results → repeat.
- Handover is implemented as a special tool (`handover_to_agent`) that the LLM can invoke.
  The framework intercepts this and switches the active agent.

---

### 4. Structured State / Conversation Management

**Rationale:** Agents need to track conversation history correctly across turns, including tool
call/result pairs. State must be serializable for persistence and replay.

**Key sub-features:**

- **Conversation history** — Maintains the full message list (system, user, assistant, tool)
  in OpenAI format. Correctly pairs tool calls with their results.
- **Serialization / deserialization** — Save the entire agent state (conversation, config,
  tool registry) to a JSON file. Restore an agent from a saved file to continue or replay.
- **Init from raw HTTP JSON** — Create a new agent pre-loaded with the conversation state
  from a logged raw request JSON file. Enables replay, debugging, and regression testing.
- **Context window management** — When conversation history exceeds the model's context
  window, apply a configurable strategy: truncation (drop oldest messages), summarization
  (LLM-generated summary of older context), or sliding window.

**Implementation notes:**

- State is a simple dataclass/Pydantic model: `AgentState(messages, tools, config, metadata)`.
- JSON serialization uses `model_dump()` / `model_validate()`.
- Raw HTTP JSON replay: parse the logged `raw_request_body`, extract `messages` and `tools`,
  and hydrate an `AgentState`.

---

## Tier 2 — Enterprise Differentiators

These features distinguish Archon for enterprise procurement and compliance.

### 5. Audit Trail

**Rationale:** Regulated industries require an immutable record of every action an AI agent
takes, who triggered it, and what data it accessed.

**Key sub-features:**

- **Immutable event log** — Every agent action (LLM call, tool invocation, tool result,
  handover, human approval, termination) is recorded as an append-only event with a
  timestamp, trace ID, step index, and event type.
- **Tenant / user attribution** — Each event is tagged with the user ID and tenant ID that
  initiated the agent run.
- **Data access tracking** — Record which tools accessed which external resources (APIs,
  databases, files).
- **Exportable formats** — Export audit trails as JSON, CSV, or PDF for compliance reviews.
- **Compliance targets** — SOC2 (access logs, change tracking), GDPR (data processing records),
  EU AI Act Article 14 (human oversight documentation).

**Implementation notes:**

- Built on top of the observability trace (Requirement 1) with additional metadata fields.
- Storage backend: pluggable (local JSON lines for dev, PostgreSQL/S3 for production).

---

### 6. Human-in-the-Loop (HITL)

**Rationale:** Enterprise agents must not autonomously execute high-risk actions without human
approval. This is both a safety requirement and a compliance requirement (EU AI Act).

**Key sub-features:**

- **Pause and approve** — The agent pauses before executing a tool call and waits for human
  approval. Configurable per tool: some tools always require approval, others never.
- **Conditional approval policies** — Rules like "require approval if transfer amount > $1000"
  or "require approval for any external API call in production".
- **Approval channels** — Integrations for routing approval requests to Slack, email, webhook,
  or a custom UI. The agent resumes when approval is received.
- **Timeout and fallback** — If no approval is received within a configurable timeout, the
  agent can skip the action, retry, or terminate gracefully.
- **Approval audit** — Every approval/rejection is recorded in the audit trail with the
  approver's identity and timestamp.

**Implementation notes:**

- HITL is implemented as middleware in the tool execution pipeline. Before a tool runs,
  the HITL middleware checks the policy and blocks if approval is required.
- Approval state is persisted (see Requirement 10) so the agent can resume after a process
  restart.

---

### 7. Guardrails

**Rationale:** Prevent agents from producing harmful output, leaking sensitive data, or
executing dangerous tool calls.

**Key sub-features:**

- **Input guardrails** — Validate and sanitize user input before it reaches the LLM. Detect
  prompt injection, jailbreak attempts, and off-topic queries.
- **Output guardrails** — Check LLM output before it's returned to the user or acted upon.
  Detect PII, toxic content, and policy violations.
- **Tool call guardrails** — Validate tool arguments before execution. E.g., reject SQL
  queries containing `DROP TABLE`, reject file paths outside an allowed directory.
- **LiteLLM proxy guardrails** — Leverage LiteLLM's built-in guardrail hooks
  (`litellm/proxy/guardrails/`) for server-side enforcement when using the proxy.
- **Pluggable guardrail interface** — Define custom guardrails as Python functions that
  receive the input/output and return allow/deny with a reason.

**Implementation notes:**

- Guardrails run as a pipeline: `input_guardrails → LLM call → output_guardrails`.
- Tool call guardrails run inside the tool execution pipeline, before the actual tool code.
- Each guardrail returns `GuardrailResult(allowed: bool, reason: str, modified_content: Optional[str])`.

---

### 8. Multi-Tenancy & Access Control

**Rationale:** Enterprise deployments serve multiple teams or customers from a single
deployment. Data isolation and access control are critical.

**Key sub-features:**

- **Tenant isolation** — Each agent run is scoped to a tenant. Conversation history, audit
  logs, and tool access are isolated per tenant.
- **Role-based access control (RBAC)** — Roles: admin (full access), operator (run agents,
  view traces), viewer (read-only traces). Configurable per tenant.
- **API key scoping** — Different API keys per tenant or team. Keys can be restricted to
  specific agents, tools, or models.
- **LiteLLM integration** — Leverage LiteLLM proxy's built-in key management, team management,
  and budget controls (`litellm/proxy/auth/`, `litellm/proxy/management_endpoints/`).

**Implementation notes:**

- Tenant context is passed through every layer (agent, tools, logging, audit).
- Access control is enforced at the API layer, not inside the agent loop.

---

## Tier 3 — Advanced

### 9. Multi-Agent Orchestration

**Rationale:** Complex enterprise workflows require multiple specialized agents working
together. This is a required capability.

**Key sub-features:**

- **Sequential pipelines** — Agent A's output feeds into Agent B as input. Useful for
  multi-stage processing (research → analysis → report writing).
- **Parallel fan-out** — Multiple agents work on sub-tasks simultaneously, results are
  merged by a coordinator. Useful for gathering information from multiple sources.
- **Supervisor pattern** — A coordinator agent receives a task, decomposes it into sub-tasks,
  delegates to worker agents, and synthesizes results.
- **Agent registry** — Named agents with defined capabilities, tools, and system prompts.
  The supervisor can select agents by capability.
- **Shared context** — Agents in a pipeline can share a context object (key-value store)
  for passing intermediate results without stuffing everything into the conversation.

**Implementation notes:**

- Orchestration is defined as a DAG (directed acyclic graph) of agent nodes.
- Each node is an agent with input/output contracts.
- The supervisor pattern is a special case where the DAG is dynamically constructed by the
  supervisor agent at runtime.

---

### 10. State Persistence & Checkpointing

**Rationale:** Enterprise workflows can be long-running (hours or days) and must survive
process restarts, deployments, and infrastructure failures.

**Key sub-features:**

- **Durable execution** — If the process crashes mid-agent-run, resume from the last
  checkpoint rather than restarting from scratch.
- **Checkpoint storage** — After each agent loop iteration, persist the current state
  (conversation, pending tool calls, metadata) to durable storage.
- **Long-running workflows** — Support workflows that span hours or days with async tool
  results (e.g., waiting for human approval, waiting for an external process to complete).
- **Resume from raw HTTP JSON** — Since we capture the full raw request JSON at each step,
  a failed run can be reconstructed by replaying the logged requests.

**Implementation approach — Netflix Conductor (or similar):**

- [Netflix Conductor](https://conductor.netflix.com/) is a workflow orchestration engine
  designed for durable, long-running workflows with built-in state persistence and retry.
- Each agent loop iteration maps to a Conductor task. The agent's state is the workflow's
  state.
- Conductor provides: task queuing, retry policies, timeout handling, workflow visualization,
  and a REST API for workflow management.
- Alternative options: Temporal.io (similar to Conductor, stronger typing), or a simpler
  custom solution using PostgreSQL + a task queue (Redis/Celery).
- The raw HTTP JSON logs serve as a secondary recovery mechanism: even without Conductor
  state, the full conversation can be reconstructed from the logged requests.

**Implementation notes:**

- Define a `WorkflowBackend` interface with implementations for:
  - `InMemoryBackend` — for dev/testing (no persistence)
  - `ConductorBackend` — Netflix Conductor integration
  - `TemporalBackend` — Temporal.io integration (alternative)
  - `PostgresBackend` — Simple custom persistence for lightweight deployments

---

### 11. Evaluation & Testing

**Rationale:** Enterprise teams need to validate agent behavior before and after deployment,
catch regressions, and compare model/prompt variants.

**Key sub-features:**

- **Replay from logged JSON** — Re-run an agent using a saved raw HTTP JSON file as the
  starting state. Compare the new output with the original. Essential for regression testing
  after prompt or model changes.
- **A/B testing** — Run the same input through two agent configurations (different models,
  prompts, or tool sets) and compare outputs side by side.
- **Drift detection** — Monitor agent behavior metrics (tool call frequency, response length,
  error rate) over time. Alert when metrics deviate from baseline.
- **Evaluation metrics** — Pluggable evaluators: correctness (vs. ground truth), latency,
  cost, tool usage efficiency, user satisfaction scores.
- **Test harness** — A pytest-compatible test runner that loads test cases (input + expected
  behavior) and runs them against the agent.

**Implementation notes:**

- Replay leverages the serialization from Requirement 4 and raw JSON from Requirement 1.
- A/B testing uses LiteLLM's router to direct traffic to different model deployments.

---

### 12. Rate Limiting & Budget Controls

**Rationale:** Prevent runaway costs, enforce fair usage across tenants, and respect provider
rate limits.

**Key sub-features:**

- **Per-agent spend limits** — Maximum dollar cost per agent run. Agent is terminated if
  exceeded (see Requirement 3, budget cap).
- **Per-tenant spend limits** — Monthly or daily spend caps per tenant. New agent runs are
  rejected when the limit is reached.
- **Concurrency throttling** — Limit the number of concurrent agent runs per tenant or
  globally. Queued runs wait until a slot is available.
- **Provider rate limit awareness** — LiteLLM's router handles per-model rate limiting,
  retries, and fallback to alternative deployments. Archon surfaces rate limit events in
  the agent trace.
- **Alerting** — Configurable alerts (webhook, email, Slack) when spend approaches a
  threshold (e.g., 80% of budget consumed).

**Implementation notes:**

- Leverage LiteLLM's router (`litellm/router.py`) for provider-level rate limiting and
  fallback.
- Implement tenant-level controls in the Archon API layer using a token bucket or sliding
  window counter backed by Redis.
- Budget tracking uses the cost data from LiteLLM callbacks (Requirement 1).

---

## Implementation Status (v0.1)

| # | Requirement | Status | Module(s) |
|---|-------------|--------|-----------|
| 1 | Observability & Raw HTTP JSON Logging | **Implemented** | `observer.py` |
| 2 | Tool/Function Calling (OpenAI + MCP) | **Implemented** | `tools.py` |
| 3 | Agent Loop Patterns (ReAct + Handover) | **Implemented** | `agent.py`, `orchestrator.py` |
| 4 | Structured State / Conversation Mgmt | **Implemented** | `state.py` |
| 5 | Audit Trail | **Implemented** | `audit.py` |
| 6 | Human-in-the-Loop | **Implemented** | `hitl.py` |
| 7 | Guardrails | **Implemented** | `guardrails.py` |
| 8 | Multi-Tenancy & Access Control | **Implemented** | `access.py`, `types.py` |
| 9 | Multi-Agent Orchestration | **Implemented** | `orchestrator.py` |
| 10 | State Persistence & Checkpointing | Planned | — (Netflix Conductor / Temporal) |
| 11 | Evaluation & Testing | Planned | — (replay from logged JSON) |
| 12 | Rate Limiting & Budget Controls | Planned | — (LiteLLM router covers most of this) |

---

## Cross-Cutting Concerns

These apply across all requirements:

- **Provider agnosticism** — All features work with any LLM provider supported by LiteLLM
  (100+ providers: OpenAI, Anthropic, Gemini, Azure, Bedrock, Ollama, etc.).
- **Async-first** — All core APIs support both sync and async execution patterns.
- **Minimal dependencies** — Core framework depends only on LiteLLM and standard library.
  Optional backends (Conductor, Temporal, Redis) are extras.
- **Configuration-driven** — Agent behavior, guardrails, HITL policies, and budget controls
  are defined in YAML/JSON config files, not hardcoded.
- **Extensibility** — Every major component (tools, guardrails, logging, state backend) uses
  a pluggable interface pattern.
