"""Minimal example — a weather agent with raw HTTP JSON logging.

Run with:  poetry run python examples/weather_agent.py

Requires OPENAI_API_KEY (or any LiteLLM-supported provider key) in the environment.
"""

import asyncio
import json

from archon.agent import Agent
from archon.audit import AuditTrail, InMemoryAuditBackend
from archon.guardrails import GuardrailPipeline, PIIDetector
from archon.observer import ArchonLogger
from archon.tools import ToolRegistry
from archon.types import AgentConfig, TenantContext


# ---------------------------------------------------------------------------
# 1. Define tools
# ---------------------------------------------------------------------------

tools = ToolRegistry()


@tools.register
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    # Stub — in production this would call a real weather API.
    temps = {"new york": 22, "london": 15, "tokyo": 28, "boston": 18}
    temp = temps.get(location.lower(), 20)
    if unit == "fahrenheit":
        temp = temp * 9 // 5 + 32
    return json.dumps({"location": location, "temperature": temp, "unit": unit})


@tools.register
def get_forecast(location: str, days: int = 3) -> str:
    """Get a weather forecast for the next N days."""
    return json.dumps({
        "location": location,
        "forecast": [{"day": i + 1, "high": 20 + i, "low": 12 + i} for i in range(days)],
    })


# ---------------------------------------------------------------------------
# 2. Set up observability, audit, guardrails
# ---------------------------------------------------------------------------

observer = ArchonLogger()
observer.install()

audit = AuditTrail(backend=InMemoryAuditBackend())
guardrails = GuardrailPipeline(input_guardrails=[PIIDetector()])

tenant = TenantContext(tenant_id="acme-corp", user_id="alice", role="operator")


# ---------------------------------------------------------------------------
# 3. Create and run the agent
# ---------------------------------------------------------------------------

async def main() -> None:
    agent = Agent(
        config=AgentConfig(
            name="weather_bot",
            model="gpt-4o-mini",
            system_prompt="You are a helpful weather assistant. Use tools to look up weather data.",
            max_iterations=5,
        ),
        tools=tools,
        observer=observer,
        audit=audit,
        guardrails=guardrails,
        tenant=tenant,
    )

    result = await agent.arun("What's the weather like in Boston and Tokyo?")

    print("\n=== Agent Output ===")
    print(result.output)
    print(f"\nIterations: {result.iterations}")
    print(f"Total cost: ${result.total_cost:.6f}")
    print(f"Total tokens: {result.total_tokens.total_tokens}")
    print(f"Stop reason: {result.stop_reason}")

    print("\n=== Trace Steps ===")
    for step in result.trace:
        print(f"  [{step.step_type.value}] duration={step.duration_ms:.0f}ms cost=${step.cost:.6f}")
        if step.raw_http:
            print(f"    → API: {step.raw_http.api_base}")
            print(f"    → Body keys: {list(step.raw_http.request_body.keys())}")

    print("\n=== Audit Events ===")
    events = await audit.backend.query(trace_id=result.run_id)
    for ev in events:
        print(f"  [{ev.event_type.value}] {ev.data}")


if __name__ == "__main__":
    asyncio.run(main())
