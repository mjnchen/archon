"""Agent state — conversation history, serialization, init from raw JSON, context window management."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from archon.types import AgentConfig, TraceStep

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Serializable state of an agent run.

    Holds the conversation history, configuration snapshot, accumulated trace,
    and arbitrary metadata.  Can be saved/loaded as JSON and restored from a
    raw HTTP request body captured by the observer.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = "default"
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    config: AgentConfig = Field(default_factory=AgentConfig)
    trace: List[TraceStep] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    total_cost: float = 0.0

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def add_system(self, content: str) -> None:
        if not self.messages or self.messages[0].get("role") != "system":
            self.messages.insert(0, {"role": "system", "content": content})
        else:
            self.messages[0]["content"] = content

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Persist the state to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> AgentState:
        """Restore state from a previously saved JSON file."""
        p = Path(path)
        return cls.model_validate_json(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Init from raw HTTP JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_raw_request(
        cls,
        raw_json: Union[str, Dict[str, Any]],
        agent_name: str = "replayed",
    ) -> AgentState:
        """Create an ``AgentState`` from a logged ``raw_request_body``.

        The raw body is the provider-specific JSON that was sent over the wire.
        For OpenAI-compatible providers it contains ``messages``, ``model``,
        ``tools``, etc.  This method extracts those fields to hydrate a state
        object suitable for replay or debugging.
        """
        if isinstance(raw_json, str):
            raw_json = json.loads(raw_json)

        messages = raw_json.get("messages", [])
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
        """Trim conversation history to fit within *max_messages*.

        Strategies:
        - ``oldest`` — keep the system message (if any) and the newest
          *max_messages - 1* messages, dropping the oldest.
        - ``sliding`` — same as oldest but preserves the last user message
          and all tool call / result pairs that follow it.
        """
        if len(self.messages) <= max_messages:
            return

        system: Optional[Dict[str, Any]] = None
        rest = self.messages
        if rest and rest[0].get("role") == "system":
            system = rest[0]
            rest = rest[1:]

        budget = max_messages - (1 if system else 0)

        if strategy == "sliding":
            rest = self._sliding_trim(rest, budget)
        else:
            rest = rest[-budget:]

        self.messages = ([system] if system else []) + rest

    @staticmethod
    def _sliding_trim(messages: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        """Keep the tail, but never split a tool_calls/tool result pair."""
        if len(messages) <= budget:
            return messages

        tail = messages[-budget:]

        # If the first message in tail is a tool result without its preceding
        # assistant+tool_calls, we need to drop it to avoid an orphaned result.
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]

        return tail
