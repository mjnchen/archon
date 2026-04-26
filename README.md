# Archon

A minimal, provider-agnostic agent framework. Think OpenAI Agents SDK's
shape — streaming, handoffs, structured output, MCP, lifecycle hooks — but
not locked to OpenAI. Memory, durable storage, and infrastructure are
seams you wire up; the framework stays out of your way.

## Why Archon?

- **No vendor lock-in.** First-class adapters for OpenAI (Chat, Reasoning,
  Responses API), Anthropic, and Gemini. Switch providers by changing the
  model name — nothing else.
- **Streaming, parallel tools, structured output, MCP** — the primitives
  modern agents are built from, all in the box.
- **Safety in the box, infra out of it.** Guardrails, HITL approvals, RBAC,
  audit, budget caps ship with the framework. Memory, vector stores, and
  durable execution don't — bring your own.
- **Readable.** ~3,500 LOC of Python you can hold in your head. No
  subclasses six layers deep.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  archon (public API)                 │
│       Agent · Session · ToolRegistry · AgentHooks    │
│       Pipeline · FanOut · Supervisor                 │
├──────────────┬──────────────┬───────────────────────┤
│ observability│    safety    │         llm            │
│ ArchonLogger │ Guardrails   │  OpenAI · Anthropic    │
│ AuditTrail   │ HITL · RBAC  │  Gemini · streaming    │
└──────────────┴──────────────┴───────────────────────┘
                       │
                       └──── archon.mcp (optional)
```

## Capabilities

| Capability                     | Status        |
|--------------------------------|---------------|
| ReAct agent loop               | ✅ |
| Streaming (`Agent.astream`)    | ✅ |
| Parallel tool calls            | ✅ |
| Structured output (JSON Schema)| ✅ |
| Lifecycle hooks (`AgentHooks`) | ✅ |
| Sessions (auto-thread state)   | ✅ |
| Handoffs                       | ✅ |
| Multi-agent orchestration      | ✅ |
| Tool calling                   | ✅ |
| MCP client (stdio + SSE)       | ✅ (optional extra) |
| Prompt caching (Anthropic)     | ✅ |
| Retries on transient errors    | ✅ |
| LLM trace & cost tracking      | ✅ |
| Audit trail                    | ✅ |
| Human-in-the-loop              | ✅ |
| Guardrails (input/output)      | ✅ |
| Multi-tenancy & RBAC           | ✅ |
| Budget & iteration caps        | ✅ |
| State save / load / replay     | ✅ |
| Long-term / semantic memory    | ❌ — bring your own |
| Durable checkpointing          | ❌ — wrap in Temporal/Restate/DBOS |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

| Provider  | Environment variable |
|-----------|----------------------|
| OpenAI    | `OPENAI_API_KEY`     |
| Anthropic | `ANTHROPIC_API_KEY`  |
| Gemini    | `GEMINI_API_KEY`     |

## Getting Started

```bash
uv add archon                  # or: pip install archon
uv add 'archon[mcp]'           # MCP client support (optional)
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

Switch providers by changing only the model name:

```python
AgentConfig(model="claude-sonnet-4-6")   # Anthropic
AgentConfig(model="gemini-2.0-flash")    # Gemini
AgentConfig(model="o3-mini")             # OpenAI reasoning
```

## Streaming

```python
from archon import Agent, AgentConfig, TextDeltaEvent, ToolStartEvent, CompleteEvent

agent = Agent(config=AgentConfig(model="gpt-4o-mini"))

async for event in agent.astream("Tell me a joke."):
    if isinstance(event, TextDeltaEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolStartEvent):
        print(f"\n[calling {event.tool_name}]")
    elif isinstance(event, CompleteEvent):
        print(f"\n\n(cost: ${event.result.total_cost:.4f})")
```

`Agent.arun` is a thin wrapper that drains `astream` for callers who just
want the final `AgentResult`.

## Structured Output

```python
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}

agent = Agent(
    config=AgentConfig(
        model="gpt-4o-mini",
        output_schema=schema,
    ),
)

result = await agent.arun("This product is amazing!")
print(result.final_output)   # {'sentiment': 'positive', 'confidence': 0.95}
```

Provider-enforced via `response_format` on OpenAI/Gemini and tool-forcing
on Anthropic. Want typed objects? One line on each side:

```python
schema = MyPydanticModel.model_json_schema()
# ... agent run ...
parsed = MyPydanticModel(**result.final_output)
```

## Sessions

Thread conversation state across `.run()` calls without managing it yourself:

```python
from archon import Session

session = Session()
await agent.arun("My name is Alice.", session=session)
await agent.arun("What's my name?", session=session)   # remembers Alice
```

Each call gets a fresh `run_id` while the message history persists in the
session.

## Lifecycle Hooks

```python
from archon import AgentHooks

class MetricsHooks(AgentHooks):
    async def on_llm_end(self, agent, response):
        print(f"LLM call: ${response.cost:.4f}, {response.usage.total_tokens} tokens")

    async def on_tool_start(self, agent, tool_call):
        print(f"Tool: {tool_call.name}({tool_call.arguments})")

agent = Agent(config=..., hooks=MetricsHooks())
```

Hooks fire alongside (not instead of) `astream` events. Use hooks for side
effects (logging, metrics); use `astream` for UI.

## MCP

Mount any MCP server's tools into a `ToolRegistry`:

```python
from archon import Agent, AgentConfig, ToolRegistry
from archon.mcp import MCPClient

async with MCPClient(command=["python", "-m", "weather_mcp"]) as mcp:
    registry = ToolRegistry()
    mcp.mount(registry)

    agent = Agent(config=AgentConfig(model="gpt-4o-mini"), tools=registry)
    result = await agent.arun("What's the weather?")
```

Both stdio and SSE transports are supported. Install with
`pip install 'archon[mcp]'`.

## Safety, Audit, HITL

```python
from archon import (
    Agent, AgentConfig, ToolRegistry,
    GuardrailPipeline, PIIDetector,
    HumanApprovalManager, ConsoleApprovalHandler, ApprovalPolicy,
    AuditTrail, JsonLinesAuditBackend,
    TenantContext, Role,
)

agent = Agent(
    config=AgentConfig(model="gpt-4o-mini"),
    tools=registry,
    guardrails=GuardrailPipeline(input_guardrails=[PIIDetector()]),
    hitl=HumanApprovalManager(
        policies=[ApprovalPolicy(tool_name_patterns=["send_*", "delete_*"])],
        handler=ConsoleApprovalHandler(),
    ),
    audit=AuditTrail(backend=JsonLinesAuditBackend("audit.jsonl")),
    tenant=TenantContext(tenant_id="acme", user_id="alice", role=Role.OPERATOR),
)
```

## Multi-Agent Orchestration

```python
from archon import AgentRegistry, Pipeline, FanOut, Supervisor

reg = AgentRegistry()
reg.register("researcher", AgentConfig(model="gpt-4o-mini", system_prompt="..."))
reg.register("writer",     AgentConfig(model="gpt-4o-mini", system_prompt="..."))

# Sequential
result = await Pipeline(reg, ["researcher", "writer"]).arun("Climate report")

# Parallel
result = await FanOut(reg, ["researcher", "writer"]).arun("Q3 earnings")

# Coordinator delegates to workers via a built-in delegate_to tool
result = await Supervisor(
    reg, coordinator="manager", workers=["researcher", "writer"],
).arun("Write a climate report")
```

## Development

```bash
git clone https://github.com/mjnchen/archon.git
cd archon
uv sync                         # runtime + dev deps
uv run pytest                   # run tests
uv run ruff check archon/       # lint
uv run ruff format archon/      # format
```

## Project Layout

```
archon/
  agent.py            core ReAct loop, astream
  hooks.py            AgentHooks lifecycle protocol
  session.py          Session — auto-thread state across runs
  retry.py            narrow retry helper
  state.py            AgentState, conversation history, save/load
  tools.py            ToolRegistry — register, execute, JSON Schema
  types.py            canonical types + streaming events
  exceptions.py
  config.py           YAML/JSON config loader
  orchestrator.py     Pipeline, FanOut, Supervisor, AgentRegistry

  llm/                provider adapters
    _base.py          LLMAdapter, LLMResponse, LLMStreamEvent, cost model
    openai.py         Chat, Reasoning, Responses (with streaming)
    anthropic.py      Claude (with streaming + prompt caching)
    gemini.py         Gemini via OpenAI-compat endpoint

  observability/
    observer.py       ArchonLogger — per-run trace collection
    audit.py          AuditTrail — immutable event log

  safety/
    guardrails.py     input/output/tool-call validation
    hitl.py           human-in-the-loop approval middleware
    access.py         RBAC

  mcp/                optional — pip install archon[mcp]
    client.py         MCPClient — mount MCP server tools
```

## License

MIT — see [LICENSE](LICENSE) for details.
