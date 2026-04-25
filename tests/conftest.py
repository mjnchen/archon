"""Shared test fixtures — build LLMResponse objects and sample tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from archon.llm import LLMChoice, LLMResponse, LLMStreamEvent, LLMUsage
from archon.types import ArchonMessage, ArchonToolCall
from archon.observability import ArchonLogger
from archon.tools import ToolRegistry
from archon.types import AgentConfig


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def make_completion_response(
    content: Optional[str] = "Hello!",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMResponse:
    """Build a real LLMResponse for use in tests.

    tool_calls items: {"id": str, "name": str, "arguments": str (JSON)}
    """
    import json as _json

    archon_tool_calls = None
    if tool_calls:
        archon_tool_calls = [
            ArchonToolCall(
                id=tc["id"],
                name=tc["name"],
                arguments=_json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"],
            )
            for tc in tool_calls
        ]

    usage = LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return LLMResponse(
        choices=[LLMChoice(message=ArchonMessage(
            role="assistant",
            content=content,
            tool_calls=archon_tool_calls,
        ))],
        usage=usage,
        cost=0.001,
    )


def make_stream_patch(*responses: LLMResponse):
    """Build a replacement for ``archon.llm.astream`` that yields events for
    each of the given LLMResponses, in order.

    Usage::

        with patch("archon.llm.astream", make_stream_patch(resp1, resp2)):
            result = await agent.arun(...)
    """
    it = iter(responses)

    async def fake_astream(*_args, **_kwargs):
        resp = next(it)
        msg = resp.choices[0].message
        if msg.content:
            yield LLMStreamEvent(kind="text_delta", text=msg.content)
        for tc in msg.tool_calls or []:
            yield LLMStreamEvent(kind="tool_call_complete", tool_call=tc)
        yield LLMStreamEvent(kind="done", usage=resp.usage, cost=resp.cost)

    return fake_astream


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tools() -> ToolRegistry:
    """A registry with two simple tools."""
    registry = ToolRegistry()

    @registry.register
    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get the current weather for a location."""
        return json.dumps({"location": location, "temp": 22, "unit": unit})

    @registry.register
    async def search_web(query: str) -> str:
        """Search the web for a query."""
        return json.dumps({"results": [f"Result for: {query}"]})

    return registry


@pytest.fixture
def observer() -> ArchonLogger:
    return ArchonLogger()


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        name="test_agent",
        model="gpt-4o-mini",
        system_prompt="You are a test agent.",
        max_iterations=5,
    )
