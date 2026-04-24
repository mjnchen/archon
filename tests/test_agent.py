"""Tests for Agent — ReAct loop, tool execution, handover, budget/iteration caps."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from archon.agent import Agent
from archon.exceptions import BudgetExceeded, HandoverRequest, MaxIterationsExceeded
from archon.observability import ArchonLogger
from archon.tools import ToolRegistry
from archon.types import AgentConfig, StepType

from tests.conftest import make_completion_response


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_should_complete_without_tool_calls(self):
        config = AgentConfig(name="simple", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config)

        mock_response = make_completion_response(content="The answer is 42.")
        with patch("archon.llm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await agent.arun("What is the meaning of life?")

        assert result.output == "The answer is 42."
        assert result.stop_reason == "completed"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_should_execute_tool_and_continue(self, sample_tools: ToolRegistry):
        config = AgentConfig(name="tool_user", model="gpt-4o-mini", max_iterations=5)
        agent = Agent(config=config, tools=sample_tools)

        # First call: LLM requests a tool call
        tool_call_response = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            }],
        )
        # Second call: LLM responds with final answer
        final_response = make_completion_response(content="It's 22°C in Boston.")

        with patch("archon.llm.acompletion", new_callable=AsyncMock, side_effect=[tool_call_response, final_response]):
            result = await agent.arun("What's the weather in Boston?")

        assert result.output == "It's 22°C in Boston."
        assert result.iterations == 2
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_should_hit_max_iterations(self):
        config = AgentConfig(name="looper", model="gpt-4o-mini", max_iterations=2)
        tools = ToolRegistry()

        @tools.register
        def dummy_tool() -> str:
            """Always called."""
            return "ok"

        tool_call_response = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "dummy_tool",
                "arguments": "{}",
            }],
        )
        agent = Agent(config=config, tools=tools)

        with patch("archon.llm.acompletion", new_callable=AsyncMock, return_value=tool_call_response):
            result = await agent.arun("Keep going")

        assert result.stop_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_should_hit_budget_cap(self):
        config = AgentConfig(name="expensive", model="gpt-4o-mini", max_iterations=10, max_cost=0.0001)
        agent = Agent(config=config)

        expensive_resp = make_completion_response(content="Intermediate")
        expensive_resp.cost = 1.0  # exceeds the 0.0001 budget

        with patch("archon.llm.acompletion", new_callable=AsyncMock, return_value=expensive_resp):
            result = await agent.arun("Expensive call")

        assert result.stop_reason == "budget_exceeded"


class TestHandover:
    @pytest.mark.asyncio
    async def test_should_raise_handover_request(self):
        config = AgentConfig(name="delegator", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config)

        response = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "handover_to_agent",
                "arguments": json.dumps({"target_agent": "specialist", "summary": "Need help"}),
            }],
        )

        with patch("archon.llm.acompletion", new_callable=AsyncMock, return_value=response):
            with pytest.raises(HandoverRequest) as exc_info:
                await agent.arun("I need a specialist")

        assert exc_info.value.target_agent == "specialist"
        assert exc_info.value.summary == "Need help"


class TestObserver:
    @pytest.mark.asyncio
    async def test_should_record_trace_steps(self, observer: ArchonLogger):
        config = AgentConfig(name="traced", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config, observer=observer)

        response = make_completion_response(content="Done.")
        with patch("archon.llm.acompletion", new_callable=AsyncMock, return_value=response):
            result = await agent.arun("Hello")

        trace = observer.get_trace(result.run_id)
        assert len(trace) == 1
        assert trace[0].step_type == StepType.LLM_CALL
        assert result.run_id is not None
