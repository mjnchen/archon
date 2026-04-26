"""Tests for MCP client mounting tools into a ToolRegistry."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from archon.mcp.client import MCPClient
from archon.tools import ToolRegistry


@pytest.mark.asyncio
async def test_mount_registers_mcp_tools_into_registry():
    """A connected MCPClient with discovered tools should mount them."""

    fake_tool = SimpleNamespace(
        name="say_hello",
        description="Say hello to someone.",
        inputSchema={
            "type": "object",
            "properties": {"who": {"type": "string"}},
            "required": ["who"],
        },
    )

    fake_session = MagicMock()
    fake_session.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(text="hello, Alice")]
        )
    )

    client = MCPClient(command=["dummy"])
    client._session = fake_session
    client._tools_cache = [fake_tool]

    registry = ToolRegistry()
    client.mount(registry)

    assert registry.has("say_hello")
    tool_def = registry.get("say_hello")
    assert tool_def.description == "Say hello to someone."
    assert tool_def.parameters["properties"]["who"]["type"] == "string"

    out = await registry.execute("say_hello", {"who": "Alice"})
    assert out == "hello, Alice"
    fake_session.call_tool.assert_awaited_once_with("say_hello", {"who": "Alice"})


def test_constructor_rejects_both_command_and_url():
    with pytest.raises(ValueError):
        MCPClient(command=["x"], url="http://x")


def test_constructor_rejects_neither_command_nor_url():
    with pytest.raises(ValueError):
        MCPClient()
