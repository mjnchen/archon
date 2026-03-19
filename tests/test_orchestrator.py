"""Tests for orchestration — Pipeline, FanOut, Supervisor."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from archon.observer import ArchonLogger
from archon.orchestrator import AgentRegistry, FanOut, Pipeline, Supervisor, run_with_handover
from archon.tools import ToolRegistry
from archon.types import AgentConfig

from tests.conftest import make_completion_response


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()

    tools_a = ToolRegistry()

    @tools_a.register
    def research(query: str) -> str:
        """Research a topic."""
        return json.dumps({"findings": f"Researched: {query}"})

    tools_b = ToolRegistry()

    @tools_b.register
    def write_report(content: str) -> str:
        """Write a report."""
        return json.dumps({"report": f"Report on: {content}"})

    reg.register("researcher", AgentConfig(model="gpt-4o-mini", system_prompt="You research things."), tools_a)
    reg.register("writer", AgentConfig(model="gpt-4o-mini", system_prompt="You write reports."), tools_b)
    reg.register("manager", AgentConfig(model="gpt-4o-mini", system_prompt="You coordinate work."))

    return reg


class TestAgentRegistry:
    def test_should_register_and_list_agents(self, registry: AgentRegistry):
        names = registry.list_agents()
        assert "researcher" in names
        assert "writer" in names
        assert "manager" in names

    def test_should_build_agent(self, registry: AgentRegistry):
        agent = registry.build("researcher")
        assert agent.config.name == "researcher"

    def test_should_raise_on_unknown_agent(self, registry: AgentRegistry):
        with pytest.raises(KeyError):
            registry.build("nonexistent")


class TestPipeline:
    @pytest.mark.asyncio
    async def test_should_chain_agents_sequentially(self, registry: AgentRegistry):
        pipeline = Pipeline(registry, ["researcher", "writer"])

        resp1 = make_completion_response(content="Research findings about climate.")
        resp2 = make_completion_response(content="Final report on climate change.")

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[resp1, resp2]):
            result = await pipeline.arun("Write about climate change")

        assert result.final_output == "Final report on climate change."
        assert len(result.agent_results) == 2
        assert result.agent_results[0].agent_name == "researcher"
        assert result.agent_results[1].agent_name == "writer"


class TestFanOut:
    @pytest.mark.asyncio
    async def test_should_run_agents_in_parallel(self, registry: AgentRegistry):
        fanout = FanOut(registry, ["researcher", "writer"])

        resp1 = make_completion_response(content="Analysis from researcher.")
        resp2 = make_completion_response(content="Analysis from writer.")

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[resp1, resp2]):
            result = await fanout.arun("Analyze Q3 earnings")

        assert len(result.agent_results) == 2
        assert "researcher" in result.final_output or "writer" in result.final_output

    @pytest.mark.asyncio
    async def test_should_use_custom_merge(self, registry: AgentRegistry):
        def custom_merge(results):
            return " | ".join(r.output for r in results)

        fanout = FanOut(registry, ["researcher", "writer"], merge_fn=custom_merge)

        resp1 = make_completion_response(content="A")
        resp2 = make_completion_response(content="B")

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[resp1, resp2]):
            result = await fanout.arun("Test")

        assert result.final_output == "A | B"


class TestSupervisor:
    @pytest.mark.asyncio
    async def test_should_coordinate_via_delegate_tool(self, registry: AgentRegistry):
        supervisor = Supervisor(
            registry,
            coordinator="manager",
            workers=["researcher", "writer"],
        )

        # Manager calls delegate_to("researcher", "research climate")
        delegate_response = make_completion_response(
            content=None,
            tool_calls=[{
                "id": "call_001",
                "name": "delegate_to",
                "arguments": json.dumps({"agent_name": "researcher", "task": "research climate"}),
            }],
        )
        # Worker (researcher) responds
        worker_response = make_completion_response(content="Climate findings.")
        # Manager synthesizes final answer
        final_response = make_completion_response(content="Final synthesis of climate research.")

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[
            delegate_response, worker_response, final_response
        ]):
            result = await supervisor.arun("Write a climate report")

        assert result.final_output == "Final synthesis of climate research."
        assert len(result.agent_results) >= 2
