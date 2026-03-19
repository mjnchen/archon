"""Tests for ToolRegistry — schema generation, execution, timeouts."""

import asyncio
import json
from typing import List

import pytest

from archon.exceptions import ToolExecutionError, ToolNotFoundError
from archon.tools import ToolRegistry


class TestSchemaGeneration:
    def test_should_generate_schema_from_type_hints(self, sample_tools: ToolRegistry):
        tool = sample_tools.get("get_weather")
        assert tool.name == "get_weather"
        assert "properties" in tool.parameters
        assert "location" in tool.parameters["properties"]
        assert tool.parameters["properties"]["location"]["type"] == "string"
        assert "required" in tool.parameters
        assert "location" in tool.parameters["required"]

    def test_should_include_default_values(self, sample_tools: ToolRegistry):
        tool = sample_tools.get("get_weather")
        unit_prop = tool.parameters["properties"]["unit"]
        assert unit_prop["default"] == "celsius"

    def test_should_extract_description_from_docstring(self, sample_tools: ToolRegistry):
        tool = sample_tools.get("get_weather")
        assert "weather" in tool.description.lower()

    def test_should_handle_list_type_hint(self):
        registry = ToolRegistry()

        @registry.register
        def process_items(items: List[str], count: int) -> str:
            """Process a list of items."""
            return ""

        tool = registry.get("process_items")
        assert tool.parameters["properties"]["items"]["type"] == "array"
        assert tool.parameters["properties"]["count"]["type"] == "integer"

    def test_should_generate_openai_format(self, sample_tools: ToolRegistry):
        tools = sample_tools.to_openai_tools()
        assert len(tools) == 2
        assert tools[0]["type"] == "function"
        assert "name" in tools[0]["function"]
        assert "parameters" in tools[0]["function"]

    def test_should_filter_by_name(self, sample_tools: ToolRegistry):
        tools = sample_tools.to_openai_tools(names=["get_weather"])
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"


class TestExecution:
    @pytest.mark.asyncio
    async def test_should_execute_sync_tool(self, sample_tools: ToolRegistry):
        result = await sample_tools.execute("get_weather", {"location": "Boston"})
        data = json.loads(result)
        assert data["location"] == "Boston"
        assert data["temp"] == 22

    @pytest.mark.asyncio
    async def test_should_execute_async_tool(self, sample_tools: ToolRegistry):
        result = await sample_tools.execute("search_web", {"query": "test"})
        data = json.loads(result)
        assert "results" in data

    @pytest.mark.asyncio
    async def test_should_raise_on_missing_tool(self, sample_tools: ToolRegistry):
        with pytest.raises(ToolNotFoundError):
            await sample_tools.execute("nonexistent", {})

    @pytest.mark.asyncio
    async def test_should_raise_on_timeout(self):
        registry = ToolRegistry()

        @registry.register(timeout=0.1)
        async def slow_tool() -> str:
            """A slow tool."""
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(ToolExecutionError):
            await registry.execute("slow_tool", {})


class TestRegistration:
    def test_should_register_with_decorator(self):
        registry = ToolRegistry()

        @registry.register
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert registry.has("my_tool")

    def test_should_register_with_custom_name(self):
        registry = ToolRegistry()

        @registry.register(name="custom_name", requires_approval=True)
        def my_tool(x: str) -> str:
            """My tool."""
            return x

        assert registry.has("custom_name")
        assert not registry.has("my_tool")
        assert registry.get("custom_name").requires_approval is True

    def test_should_list_all_tools(self, sample_tools: ToolRegistry):
        tools = sample_tools.list_tools()
        names = {t.name for t in tools}
        assert "get_weather" in names
        assert "search_web" in names

    def test_should_merge_registries(self):
        r1 = ToolRegistry()
        r2 = ToolRegistry()

        @r1.register
        def tool_a() -> str:
            """A."""
            return "a"

        @r2.register
        def tool_b() -> str:
            """B."""
            return "b"

        r1.merge(r2)
        assert r1.has("tool_a")
        assert r1.has("tool_b")
