# Archon

A minimal enterprise agent framework built on [LiteLLM](https://github.com/BerriAI/litellm).

Archon provides a provider-agnostic agent runtime with enterprise-grade observability,
audit trails, guardrails, and durable workflow execution — all backed by LiteLLM's
unified interface to 100+ LLM providers.

## Why Archon?

- **Full HTTP request transparency** — Every raw JSON request/response sent to any LLM
  provider is captured and queryable. Replay any agent run from its logged HTTP traces.
- **Durable workflows** — Long-running agent tasks survive process restarts via Netflix
  Conductor (or Temporal) integration for state persistence and checkpointing.
- **Multi-provider by default** — Runs on OpenAI, Anthropic, Gemini, Azure, Bedrock,
  Ollama, and 100+ other providers through LiteLLM with zero code changes.
- **Enterprise-ready** — Multi-tenancy, RBAC, audit trails, human-in-the-loop approvals,
  input/output guardrails, and budget controls built in.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Archon API                     │
│          (Agent runs, traces, config)            │
├──────────┬──────────┬──────────┬────────────────┤
│  Agent   │ Guardrails│  HITL   │  Audit Trail   │
│  Loops   │ Pipeline  │ Middleware│  Event Log    │
├──────────┴──────────┴──────────┴────────────────┤
│              Tool Execution Layer                 │
│       (OpenAI tools + MCP + Python callables)    │
├─────────────────────────────────────────────────┤
│                   LiteLLM                        │
│   (Unified LLM interface, routing, callbacks)    │
├─────────────────────────────────────────────────┤
│        State / Workflow Backend                   │
│  (In-memory | Conductor | Temporal | Postgres)   │
└─────────────────────────────────────────────────┘
```

## Key Capabilities

| Capability                  | Status   |
|-----------------------------|----------|
| Raw HTTP JSON logging       | Planned  |
| ReAct loop                  | Planned  |
| Agent handover              | Planned  |
| OpenAI + MCP tool calling   | Planned  |
| Multi-agent orchestration   | Planned  |
| Audit trail                 | Planned  |
| Human-in-the-loop           | Planned  |
| Guardrails                  | Planned  |
| State persistence           | Planned  |
| Multi-tenancy & RBAC        | Planned  |
| Evaluation & replay         | Planned  |
| Rate limiting & budgets     | Planned  |

## Requirements

See [REQUIREMENTS.md](REQUIREMENTS.md) for the full specification of all 12 enterprise
requirements organized by priority tier.

## Getting Started

```bash
pip install archon
# or
poetry add archon
```

*Archon is under active development. API is not yet stable.*

## Development

```bash
git clone https://github.com/your-org/archon.git
cd archon
poetry install
poetry run pytest
```

## License

TBD
