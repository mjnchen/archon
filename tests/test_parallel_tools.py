"""Tests for parallel tool execution in the agent loop."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch

import pytest

from archon.agent import Agent
from archon.tools import ToolRegistry
from archon.types import AgentConfig

from tests.conftest import make_completion_response, make_stream_patch


class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_should_run_multiple_tool_calls_concurrently(self):
        """Two slow tools called in one turn should finish in roughly the
        time of the slowest, not the sum."""
        registry = ToolRegistry()

        @registry.register
        async def slow_a() -> str:
            """Slow tool A."""
            await asyncio.sleep(0.15)
            return "a-done"

        @registry.register
        async def slow_b() -> str:
            """Slow tool B."""
            await asyncio.sleep(0.15)
            return "b-done"

        tool_call_resp = make_completion_response(
            content=None,
            tool_calls=[
                {"id": "tc_a", "name": "slow_a", "arguments": "{}"},
                {"id": "tc_b", "name": "slow_b", "arguments": "{}"},
            ],
        )
        final_resp = make_completion_response(content="ok")

        config = AgentConfig(name="par", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config, tools=registry)

        t0 = time.monotonic()
        with patch(
            "archon.llm.astream",
            make_stream_patch(tool_call_resp, final_resp),
        ):
            await agent.arun("run both")
        elapsed = time.monotonic() - t0

        # Sequential would be ~0.30s; parallel should be ~0.15s. Allow slack.
        assert elapsed < 0.25, f"tools ran sequentially (took {elapsed:.2f}s)"

    @pytest.mark.asyncio
    async def test_should_commit_results_in_tool_calls_order(self):
        """Even when fast tools complete first, tool messages must be added
        to state in the original tool_calls order — wire format demands it."""
        registry = ToolRegistry()

        @registry.register
        async def fast_one() -> str:
            """Fast tool — returns immediately."""
            return "FAST"

        @registry.register
        async def slow_two() -> str:
            """Slow tool — completes second."""
            await asyncio.sleep(0.05)
            return "SLOW"

        tool_call_resp = make_completion_response(
            content=None,
            tool_calls=[
                {"id": "tc_slow", "name": "slow_two", "arguments": "{}"},
                {"id": "tc_fast", "name": "fast_one", "arguments": "{}"},
            ],
        )
        final_resp = make_completion_response(content="done")

        config = AgentConfig(name="ordered", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config, tools=registry)

        with patch(
            "archon.llm.astream",
            make_stream_patch(tool_call_resp, final_resp),
        ):
            result = await agent.arun("run")

        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert tool_msgs[0].tool_call_id == "tc_slow"  # ordered by tool_calls, not completion
        assert tool_msgs[0].content == "SLOW"
        assert tool_msgs[1].tool_call_id == "tc_fast"
        assert tool_msgs[1].content == "FAST"
