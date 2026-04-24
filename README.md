# Archon

A minimal enterprise agent framework with a provider-agnostic LLM layer,
built-in observability, audit trails, guardrails, and multi-agent orchestration.

## Why Archon?

- **No vendor lock-in** — First-class adapters for OpenAI (Chat Completions,
  Reasoning, Responses API), Anthropic, and Gemini. Adding a new provider means
  one file and one registry entry, nothing else changes.
- **Full request transparency** — Every LLM call is captured as a structured
  trace step: messages sent, response received, tokens used, cost estimated.
- **Enterprise-ready out of the box** — Multi-tenancy, RBAC, audit trails,
  human-in-the-loop approvals, input/output guardrails, and budget controls
  built in, not bolted on.
- **Extensible by design** — Provider adapters, guardrails, approval handlers,
  and audit backends are all open protocols you implement and register.

## Architecture

```
┌─────────────────────────────────────────────────┐
│               archon (public API)                │
│     Agent · AgentState · ToolRegistry            │
│     Pipeline · FanOut · Supervisor               │
├──────────────┬──────────────┬───────────────────┤
│  observability│    safety    │       llm          │
│  ArchonLogger │ Guardrails  │  OpenAI (chat /   │
│  AuditTrail  │ HITL        │  reasoning /       │
│              │ RBAC        │  responses API)    │
│              │             │  Anthropic         │
│              │             │  Gemini            │
└──────────────┴──────────────┴───────────────────┘
```

## Key Capabilities

| Capability                   | Status        |
|------------------------------|---------------|
| ReAct agent loop             | ✅ Implemented |
| Agent handover               | ✅ Implemented |
| Multi-agent orchestration    | ✅ Implemented |
| Tool calling (OpenAI format) | ✅ Implemented |
| LLM trace & cost tracking    | ✅ Implemented |
| Audit trail                  | ✅ Implemented |
| Human-in-the-loop            | ✅ Implemented |
| Guardrails (input/output)    | ✅ Implemented |
| Multi-tenancy & RBAC         | ✅ Implemented |
| Budget & iteration caps      | ✅ Implemented |
| Context window management    | ✅ Implemented |
| State save/load/replay       | ✅ Implemented |
| OpenAI Chat Completions      | ✅ Implemented |
| OpenAI Reasoning (o1/o3)     | ✅ Implemented |
| OpenAI Responses API         | ✅ Implemented |
| Anthropic (Claude)           | ✅ Implemented |
| Gemini                       | ✅ Implemented |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

Provider API keys in environment variables:

| Provider  | Environment variable |
|-----------|----------------------|
| OpenAI    | `OPENAI_API_KEY`     |
| Anthropic | `ANTHROPIC_API_KEY`  |
| Gemini    | `GEMINI_API_KEY`     |

## Getting Started

```bash
# with uv (recommended)
uv add archon

# with pip
pip install archon
```

```python
from archon import Agent, AgentConfig, ToolRegistry

registry = ToolRegistry()

@registry.register
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Sunny, 22°C in {location}"

agent = Agent(
    config=AgentConfig(
        model="gpt-4o-mini",
        system_prompt="You are a helpful weather assistant.",
    ),
    tools=registry,
)

result = agent.run("What's the weather in Tokyo?")
print(result.output)
```

Switch provider by changing the model name — no other code changes:

```python
AgentConfig(model="claude-sonnet-4-6")   # Anthropic
AgentConfig(model="gemini-2.0-flash")    # Gemini
AgentConfig(model="o3-mini")             # OpenAI reasoning
```

## Development

### Setup

```bash
git clone https://github.com/your-org/archon.git
cd archon
uv sync          # installs runtime + dev dependencies
```

### Common tasks

```bash
uv run pytest                  # run tests
uv run ruff check archon/      # lint
uv run ruff format archon/     # format

uv add <package>               # add a runtime dependency
uv add --dev <package>         # add a dev dependency
uv lock --upgrade              # upgrade all deps and regenerate lockfile
```

### Project layout

```
archon/
  agent.py            core ReAct loop
  state.py            conversation state & serialization
  tools.py            tool registry
  types.py            canonical types (ArchonMessage, AgentConfig, …)
  exceptions.py
  config.py           YAML/JSON config loader
  orchestrator.py     Pipeline, FanOut, Supervisor, AgentRegistry

  llm/                provider adapters
    _base.py          LLMAdapter ABC, LLMResponse, cost estimation
    openai.py         Chat Completions, Reasoning (o1/o3), Responses API
    anthropic.py      Claude
    gemini.py         Gemini via OpenAI-compat endpoint

  observability/
    observer.py       ArchonLogger — per-run trace collection
    audit.py          AuditTrail — immutable event log

  safety/
    guardrails.py     input/output/tool-call validation pipeline
    hitl.py           human approval middleware
    access.py         RBAC permission helpers
```

## License

TBD
