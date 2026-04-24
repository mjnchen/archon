"""Agent state — conversation history, serialization, context window management."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from archon.types import AgentConfig, ArchonMessage, ArchonToolCall, TraceStep

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Serializable state of an agent run.

    Holds the conversation history as ``ArchonMessage`` objects, a configuration
    snapshot, accumulated trace, and arbitrary metadata.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = "default"
    messages: List[ArchonMessage] = Field(default_factory=list)
    config: AgentConfig = Field(default_factory=AgentConfig)
    trace: List[TraceStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    total_cost: float = 0.0

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_system(self, content: str) -> None:
        if not self.messages or self.messages[0].role != "system":
            self.messages.insert(0, ArchonMessage(role="system", content=content))
        else:
            self.messages[0].content = content

    def add_user(self, content: str) -> None:
        self.messages.append(ArchonMessage(role="user", content=content))

    def add_assistant(self, msg: ArchonMessage) -> None:
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append(ArchonMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        ))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> AgentState:
        p = Path(path)
        return cls.model_validate_json(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Init from raw HTTP JSON (OpenAI Chat Completions wire format)
    # ------------------------------------------------------------------

    @classmethod
    def from_raw_request(
        cls,
        raw_json: Union[str, Dict[str, Any]],
        agent_name: str = "replayed",
    ) -> AgentState:
        """Create an ``AgentState`` from a logged OpenAI-format ``raw_request_body``."""
        if isinstance(raw_json, str):
            raw_json = json.loads(raw_json)

        messages = [
            _archon_message_from_wire(m)
            for m in raw_json.get("messages", [])
        ]
        model = raw_json.get("model", "unknown")
        tools = raw_json.get("tools", [])
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in tools
            if t.get("type") == "function"
        ]

        config = AgentConfig(
            name=agent_name,
            model=model,
            tool_names=tool_names,
            temperature=raw_json.get("temperature"),
            top_p=raw_json.get("top_p"),
        )

        return cls(
            agent_name=agent_name,
            messages=messages,
            config=config,
            metadata={"restored_from": "raw_request", "original_tools": tools},
        )

    # ------------------------------------------------------------------
    # Context window management
    # ------------------------------------------------------------------

    def truncate(
        self,
        max_messages: int,
        strategy: Literal["oldest", "sliding"] = "oldest",
    ) -> None:
        """Trim conversation history to fit within *max_messages*."""
        if len(self.messages) <= max_messages:
            return

        system: Optional[ArchonMessage] = None
        rest = self.messages
        if rest and rest[0].role == "system":
            system = rest[0]
            rest = rest[1:]

        budget = max_messages - (1 if system else 0)

        if strategy == "sliding":
            rest = self._sliding_trim(rest, budget)
        else:
            rest = rest[-budget:]

        self.messages = ([system] if system else []) + rest

    @staticmethod
    def _sliding_trim(messages: List[ArchonMessage], budget: int) -> List[ArchonMessage]:
        if len(messages) <= budget:
            return messages

        tail = messages[-budget:]

        # Never start with an orphaned tool result that has no preceding assistant turn.
        while tail and tail[0].role == "tool":
            tail = tail[1:]

        return tail


# ---------------------------------------------------------------------------
# Helper: parse a single OpenAI-wire message dict into ArchonMessage
# ---------------------------------------------------------------------------

def _archon_message_from_wire(d: Dict[str, Any]) -> ArchonMessage:
    role = d.get("role", "user")
    content = d.get("content")
    tool_call_id = d.get("tool_call_id")

    tool_calls = None
    if d.get("tool_calls"):
        tool_calls = [
            ArchonToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"].get("arguments", "{}")),
            )
            for tc in d["tool_calls"]
        ]

    return ArchonMessage(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
