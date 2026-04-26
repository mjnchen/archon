"""Tests for the streaming layer — adapter default fallback + agent events."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from archon.agent import Agent
from archon.llm._base import LLMAdapter, LLMResponse, LLMChoice, LLMUsage
from archon.tools import ToolRegistry
from archon.types import (
    AgentConfig,
    ArchonMessage,
    ArchonToolCall,
    CompleteEvent,
    IterationEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolEndEvent,
    ToolStartEvent,
)

from tests.conftest import make_completion_response, make_stream_patch


class _FakeAdapter(LLMAdapter):
    """Adapter whose ``complete`` returns a canned response. Used to verify
    the base-class ``astream`` fallback yields equivalent events."""

    def __init__(self, response: LLMResponse):
        self._response = response

    async def complete(self, *args, **kwargs):
        return self._response


class TestAdapterStreamFallback:
    @pytest.mark.asyncio
    async def test_default_astream_emits_text_then_done(self):
        resp = LLMResponse(
            choices=[LLMChoice(message=ArchonMessage(role="assistant", content="hi"))],
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost=0.0,
        )
        adapter = _FakeAdapter(resp)

        events = []
        async for ev in adapter.astream(
            model="x", messages=[], tools=None, temperature=None, top_p=None,
        ):
            events.append(ev)

        kinds = [e.kind for e in events]
        assert kinds == ["text_delta", "done"]
        assert events[0].text == "hi"
        assert events[-1].usage.total_tokens == 2

    @pytest.mark.asyncio
    async def test_default_astream_emits_tool_calls_before_done(self):
        resp = LLMResponse(
            choices=[LLMChoice(message=ArchonMessage(
                role="assistant",
                content=None,
                tool_calls=[ArchonToolCall(id="t1", name="foo", arguments={"a": 1})],
            ))],
            usage=LLMUsage(),
            cost=0.0,
        )
        adapter = _FakeAdapter(resp)

        events = []
        async for ev in adapter.astream(
            model="x", messages=[], tools=None, temperature=None, top_p=None,
        ):
            events.append(ev)

        kinds = [e.kind for e in events]
        assert kinds == ["tool_call_complete", "done"]
        assert events[0].tool_call.name == "foo"
        assert events[0].tool_call.arguments == {"a": 1}


class TestAgentEventOrder:
    @pytest.mark.asyncio
    async def test_iteration_precedes_text_delta_precedes_complete(self):
        config = AgentConfig(name="ord", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config)

        resp = make_completion_response(content="answer")
        events = []
        with patch("archon.llm.astream", make_stream_patch(resp)):
            async for ev in agent.astream("?"):
                events.append(ev)

        i_idx = next(i for i, e in enumerate(events) if isinstance(e, IterationEvent))
        t_idx = next(i for i, e in enumerate(events) if isinstance(e, TextDeltaEvent))
        c_idx = next(i for i, e in enumerate(events) if isinstance(e, CompleteEvent))
        assert i_idx < t_idx < c_idx

    @pytest.mark.asyncio
    async def test_tool_start_emitted_before_tool_end(self, sample_tools: ToolRegistry):
        config = AgentConfig(name="t", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config, tools=sample_tools)

        tc_resp = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "tc1",
                "name": "get_weather",
                "arguments": json.dumps({"location": "NYC"}),
            }],
        )
        final = make_completion_response(content="done")

        with patch("archon.llm.astream", make_stream_patch(tc_resp, final)):
            events = [e async for e in agent.astream("weather?")]

        start_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolStartEvent))
        end_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolEndEvent))
        call_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolCallEvent))
        # The model decides on the call (ToolCallEvent), then the agent
        # signals start, runs it, signals end.
        assert call_idx < start_idx < end_idx

    @pytest.mark.asyncio
    async def test_complete_event_is_terminal(self):
        config = AgentConfig(name="t", model="gpt-4o-mini", max_iterations=2)
        agent = Agent(config=config)

        resp = make_completion_response(content="done")
        with patch("archon.llm.astream", make_stream_patch(resp)):
            events = [e async for e in agent.astream("?")]

        assert isinstance(events[-1], CompleteEvent)
        assert events[-1].result.output == "done"
