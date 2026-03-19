"""Shared test fixtures — mock LiteLLM responses and sample tools."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from archon.observer import ArchonLogger
from archon.tools import ToolRegistry
from archon.types import AgentConfig


# ---------------------------------------------------------------------------
# Mock LiteLLM response objects
# ---------------------------------------------------------------------------

def make_assistant_message(
    content: Optional[str] = "Hello!",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> MagicMock:
    """Build a mock assistant message (mirrors litellm's response.choices[0].message)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        msg.model_dump.return_value["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]
        mock_tcs = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc["id"]
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = tc["arguments"]
            mock_tcs.append(mock_tc)
        msg.tool_calls = mock_tcs
    return msg


def make_completion_response(
    content: Optional[str] = "Hello!",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a mock litellm.acompletion() return value."""
    msg = make_assistant_message(content, tool_calls)
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = msg
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    response._hidden_params = {"response_cost": 0.001}
    return response


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
