"""Lifecycle hooks — async callbacks fired at key points in an agent run.

Subclass :class:`AgentHooks` and override the events you care about. All
methods are async and default to no-ops, so subclasses only implement what
they need. Pass an instance to ``Agent(hooks=...)``.

Hooks are for side effects (telemetry, custom logging, metrics). For
real-time streaming UI use ``Agent.astream`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from archon.exceptions import HandoverRequest
from archon.llm._base import LLMResponse
from archon.state import AgentState
from archon.types import AgentResult, ArchonMessage, ArchonToolCall

if TYPE_CHECKING:
    from archon.agent import Agent


class AgentHooks:
    """Default no-op hook implementation. Subclass to react to events."""

    async def on_agent_start(self, agent: "Agent", state: AgentState) -> None:
        pass

    async def on_agent_end(self, agent: "Agent", result: AgentResult) -> None:
        pass

    async def on_llm_start(
        self,
        agent: "Agent",
        messages: List[ArchonMessage],
    ) -> None:
        pass

    async def on_llm_end(self, agent: "Agent", response: LLMResponse) -> None:
        pass

    async def on_tool_start(
        self,
        agent: "Agent",
        tool_call: ArchonToolCall,
    ) -> None:
        pass

    async def on_tool_end(
        self,
        agent: "Agent",
        tool_call: ArchonToolCall,
        output: str,
    ) -> None:
        pass

    async def on_handoff(self, agent: "Agent", request: HandoverRequest) -> None:
        pass
