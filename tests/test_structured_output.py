"""Tests for structured output — both the JSON-content path and the
synthetic-tool path used on Anthropic.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from archon.agent import Agent
from archon.llm._base import FINAL_OUTPUT_TOOL_NAME
from archon.types import AgentConfig

from tests.conftest import make_completion_response, make_stream_patch


SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}


class TestJsonContentPath:
    """OpenAI / Gemini path: model emits JSON as plain content; agent parses it."""

    @pytest.mark.asyncio
    async def test_should_parse_valid_json_into_final_output(self):
        config = AgentConfig(
            name="extractor",
            model="gpt-4o-mini",
            output_schema=SCHEMA,
            max_iterations=2,
        )
        agent = Agent(config=config)

        resp = make_completion_response(
            content='{"sentiment": "positive", "confidence": 0.95}'
        )
        with patch("archon.llm.astream", make_stream_patch(resp)):
            result = await agent.arun("This product is amazing!")

        assert result.final_output == {"sentiment": "positive", "confidence": 0.95}
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_should_set_final_output_none_on_invalid_json(self):
        config = AgentConfig(
            name="extractor",
            model="gpt-4o-mini",
            output_schema=SCHEMA,
            max_iterations=2,
        )
        agent = Agent(config=config)

        resp = make_completion_response(content="not actually json {")
        with patch("archon.llm.astream", make_stream_patch(resp)):
            result = await agent.arun("?")

        assert result.final_output is None


class TestSyntheticToolPath:
    """Anthropic path: model calls __archon_final_output__ with structured args."""

    @pytest.mark.asyncio
    async def test_should_extract_arguments_from_synthetic_tool_call(self):
        config = AgentConfig(
            name="extractor",
            model="claude-sonnet-4-6",
            output_schema=SCHEMA,
            max_iterations=2,
        )
        agent = Agent(config=config)

        resp = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "toolu_001",
                "name": FINAL_OUTPUT_TOOL_NAME,
                "arguments": json.dumps(
                    {"sentiment": "negative", "confidence": 0.8}
                ),
            }],
        )
        with patch("archon.llm.astream", make_stream_patch(resp)):
            result = await agent.arun("Disappointing.")

        assert result.final_output == {"sentiment": "negative", "confidence": 0.8}
        assert result.stop_reason == "completed"

    @pytest.mark.asyncio
    async def test_should_take_synthetic_tool_call_over_other_tool_calls(self):
        """If the model returns both a real tool call and the final-output
        tool call in one turn, the final-output wins and we stop."""
        from archon.tools import ToolRegistry

        registry = ToolRegistry()

        @registry.register
        def real_tool() -> str:
            """A real tool the model could call."""
            return "ran"

        config = AgentConfig(
            name="multi",
            model="claude-sonnet-4-6",
            output_schema=SCHEMA,
            max_iterations=2,
        )
        agent = Agent(config=config, tools=registry)

        resp = make_completion_response(
            content=None,
            tool_calls=[
                {
                    "id": "tc_1",
                    "name": "real_tool",
                    "arguments": "{}",
                },
                {
                    "id": "tc_2",
                    "name": FINAL_OUTPUT_TOOL_NAME,
                    "arguments": json.dumps(
                        {"sentiment": "positive", "confidence": 0.9}
                    ),
                },
            ],
        )
        with patch("archon.llm.astream", make_stream_patch(resp)):
            result = await agent.arun("?")

        assert result.final_output == {"sentiment": "positive", "confidence": 0.9}
        assert result.iterations == 1
