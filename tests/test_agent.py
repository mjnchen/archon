"""Tests for Agent — ReAct loop, tool execution, handover, budget/iteration caps."""

import json
from unittest.mock import patch

import pytest

from archon.agent import Agent
from archon.exceptions import BudgetExceeded, HandoverRequest, MaxIterationsExceeded
from archon.observability import ArchonLogger
from archon.tools import ToolRegistry
from archon.types import AgentConfig, StepType

from tests.conftest import make_completion_response, make_stream_patch


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_should_complete_without_tool_calls(self):
        config = AgentConfig(name="simple", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config)

        mock_response = make_completion_response(content="The answer is 42.")
        with patch("archon.llm.astream", make_stream_patch(mock_response)):
            result = await agent.arun("What is the meaning of life?")

        assert result.output == "The answer is 42."
        assert result.stop_reason == "completed"
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_should_execute_tool_and_continue(self, sample_tools: ToolRegistry):
        config = AgentConfig(name="tool_user", model="gpt-4o-mini", max_iterations=5)
        agent = Agent(config=config, tools=sample_tools)

        tool_call_response = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            }],
        )
        final_response = make_completion_response(content="It's 22°C in Boston.")

        with patch("archon.llm.astream", make_stream_patch(tool_call_response, final_response)):
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

        with patch(
            "archon.llm.astream",
            make_stream_patch(tool_call_response, tool_call_response),
        ):
            result = await agent.arun("Keep going")

        assert result.stop_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_should_hit_budget_cap(self):
        config = AgentConfig(name="expensive", model="gpt-4o-mini", max_iterations=10, max_cost=0.0001)
        agent = Agent(config=config)

        expensive_resp = make_completion_response(content="Intermediate")
        expensive_resp.cost = 1.0  # exceeds the 0.0001 budget

        with patch("archon.llm.astream", make_stream_patch(expensive_resp)):
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

        with patch("archon.llm.astream", make_stream_patch(response)):
            with pytest.raises(HandoverRequest) as exc_info:
                await agent.arun("I need a specialist")

        assert exc_info.value.target_agent == "specialist"
        assert exc_info.value.summary == "Need help"


class TestStreaming:
    @pytest.mark.asyncio
    async def test_should_yield_text_delta_and_complete_events(self):
        from archon.types import (
            CompleteEvent,
            IterationEvent,
            TextDeltaEvent,
        )

        config = AgentConfig(name="streamer", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config)

        response = make_completion_response(content="streamed answer")
        events = []
        with patch("archon.llm.astream", make_stream_patch(response)):
            async for ev in agent.astream("hi"):
                events.append(ev)

        assert any(isinstance(e, IterationEvent) for e in events)
        assert any(isinstance(e, TextDeltaEvent) and e.text == "streamed answer" for e in events)
        assert isinstance(events[-1], CompleteEvent)
        assert events[-1].result.output == "streamed answer"

    @pytest.mark.asyncio
    async def test_should_emit_tool_start_and_end_events(self, sample_tools: ToolRegistry):
        from archon.types import ToolEndEvent, ToolStartEvent

        config = AgentConfig(name="tool_streamer", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config, tools=sample_tools)

        tool_call_resp = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            }],
        )
        final_resp = make_completion_response(content="It's nice.")

        starts, ends = [], []
        with patch("archon.llm.astream", make_stream_patch(tool_call_resp, final_resp)):
            async for ev in agent.astream("weather?"):
                if isinstance(ev, ToolStartEvent):
                    starts.append(ev)
                elif isinstance(ev, ToolEndEvent):
                    ends.append(ev)

        assert len(starts) == 1
        assert starts[0].tool_name == "get_weather"
        assert len(ends) == 1
        assert ends[0].tool_call_id == "call_001"


class TestHooks:
    @pytest.mark.asyncio
    async def test_should_fire_lifecycle_hooks(self, sample_tools: ToolRegistry):
        from archon.hooks import AgentHooks

        events: list[str] = []

        class RecordingHooks(AgentHooks):
            async def on_agent_start(self, agent, state):
                events.append("agent_start")

            async def on_llm_start(self, agent, messages):
                events.append("llm_start")

            async def on_llm_end(self, agent, response):
                events.append("llm_end")

            async def on_tool_start(self, agent, tool_call):
                events.append(f"tool_start:{tool_call.name}")

            async def on_tool_end(self, agent, tool_call, output):
                events.append(f"tool_end:{tool_call.name}")

            async def on_agent_end(self, agent, result):
                events.append("agent_end")

        config = AgentConfig(name="hooked", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config, tools=sample_tools, hooks=RecordingHooks())

        tool_call_resp = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            }],
        )
        final_resp = make_completion_response(content="Done.")

        with patch("archon.llm.astream", make_stream_patch(tool_call_resp, final_resp)):
            await agent.arun("weather?")

        assert events[0] == "agent_start"
        assert events[-1] == "agent_end"
        assert "tool_start:get_weather" in events
        assert "tool_end:get_weather" in events
        assert events.count("llm_start") == 2  # one per iteration
        assert events.count("llm_end") == 2


class TestSession:
    @pytest.mark.asyncio
    async def test_should_thread_history_across_runs(self):
        from archon.session import Session

        config = AgentConfig(name="chatter", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config)
        session = Session()

        first = make_completion_response(content="Hello Alice.")
        second = make_completion_response(content="Your name is Alice.")

        with patch("archon.llm.astream", make_stream_patch(first)):
            await agent.arun("My name is Alice.", session=session)
        with patch("archon.llm.astream", make_stream_patch(second)):
            result = await agent.arun("What's my name?", session=session)

        # Session.state should accumulate user/assistant turns across runs.
        roles = [m.role for m in session.state.messages]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2
        assert result.output == "Your name is Alice."


class TestObserver:
    @pytest.mark.asyncio
    async def test_should_record_trace_steps(self, observer: ArchonLogger):
        config = AgentConfig(name="traced", model="gpt-4o-mini", max_iterations=3)
        agent = Agent(config=config, observer=observer)

        response = make_completion_response(content="Done.")
        with patch("archon.llm.astream", make_stream_patch(response)):
            result = await agent.arun("Hello")

        trace = observer.get_trace(result.run_id)
        assert len(trace) == 1
        assert trace[0].step_type == StepType.LLM_CALL
        assert result.run_id is not None
