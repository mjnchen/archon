"""Tests for AgentState — serialization, from_raw_request, truncation."""

import json
import tempfile
from pathlib import Path

import pytest

from archon.state import AgentState
from archon.types import AgentConfig, ArchonMessage


class TestSerialization:
    def test_should_roundtrip_json(self, tmp_path: Path):
        state = AgentState(agent_name="test")
        state.add_system("You are helpful.")
        state.add_user("Hello")
        state.add_assistant(ArchonMessage(role="assistant", content="Hi there!"))

        path = tmp_path / "state.json"
        state.save(path)
        loaded = AgentState.load(path)

        assert loaded.agent_name == "test"
        assert len(loaded.messages) == 3
        assert loaded.messages[0].role == "system"
        assert loaded.messages[1].content == "Hello"
        assert loaded.messages[2].content == "Hi there!"

    def test_should_preserve_run_id(self, tmp_path: Path):
        state = AgentState(run_id="abc123")
        path = tmp_path / "state.json"
        state.save(path)
        loaded = AgentState.load(path)
        assert loaded.run_id == "abc123"


class TestFromRawRequest:
    def test_should_hydrate_from_openai_request_body(self):
        raw = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "temperature": 0.7,
        }
        state = AgentState.from_raw_request(raw)
        assert state.config.model == "gpt-4o"
        assert len(state.messages) == 2
        assert state.config.temperature == 0.7
        assert "get_weather" in state.config.tool_names

    def test_should_accept_json_string(self):
        raw = json.dumps({"model": "claude-3", "messages": [{"role": "user", "content": "Hi"}]})
        state = AgentState.from_raw_request(raw)
        assert state.config.model == "claude-3"
        assert len(state.messages) == 1

    def test_should_handle_empty_request(self):
        state = AgentState.from_raw_request({})
        assert state.config.model == "unknown"
        assert len(state.messages) == 0


class TestMessageHelpers:
    def test_should_add_system_at_front(self):
        state = AgentState()
        state.add_user("Hello")
        state.add_system("System prompt")
        assert state.messages[0].role == "system"
        assert state.messages[1].role == "user"

    def test_should_replace_existing_system(self):
        state = AgentState()
        state.add_system("V1")
        state.add_system("V2")
        assert len([m for m in state.messages if m.role == "system"]) == 1
        assert state.messages[0].content == "V2"

    def test_should_add_tool_result(self):
        state = AgentState()
        state.add_tool_result("call_123", "result data")
        assert state.messages[-1].role == "tool"
        assert state.messages[-1].tool_call_id == "call_123"


class TestTruncation:
    def test_should_keep_within_limit(self):
        state = AgentState()
        state.add_system("sys")
        for i in range(10):
            state.add_user(f"msg {i}")
        state.truncate(max_messages=5)
        assert len(state.messages) == 5
        assert state.messages[0].role == "system"

    def test_should_noop_when_under_limit(self):
        state = AgentState()
        state.add_user("hello")
        state.truncate(max_messages=10)
        assert len(state.messages) == 1

    def test_sliding_should_not_start_with_tool_result(self):
        state = AgentState()
        state.add_system("sys")
        state.add_user("q1")
        state.add_assistant(ArchonMessage(role="assistant", content="a1"))
        state.add_tool_result("c1", "r1")
        state.add_user("q2")
        state.add_assistant(ArchonMessage(role="assistant", content="a2"))

        state.truncate(max_messages=4, strategy="sliding")
        assert state.messages[0].role == "system"
        # Should not start the non-system part with a tool result
        assert state.messages[1].role != "tool"
