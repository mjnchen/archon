"""Anthropic adapter — Claude model family."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from archon.types import ArchonMessage, ArchonToolCall
from archon.llm._base import LLMAdapter, LLMChoice, LLMResponse, LLMUsage, estimate_cost


# ---------------------------------------------------------------------------
# Wire format helpers
# ---------------------------------------------------------------------------

def to_anthropic_wire(
    messages: List[ArchonMessage],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """ArchonMessage list → (system_str, Anthropic messages list).

    Differences from OpenAI wire format:
      • System message → separate ``system`` parameter
      • Tool calls     → ``tool_use`` content blocks inside the assistant turn
      • Tool results   → ``tool_result`` content blocks merged into one user turn
    """
    system: Optional[str] = None
    out: List[Dict[str, Any]] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "system":
            system = msg.content
            i += 1

        elif msg.role == "user":
            out.append({"role": "user", "content": msg.content})
            i += 1

        elif msg.role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls or []:
                blocks.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            out.append({"role": "assistant", "content": blocks})
            i += 1

        elif msg.role == "tool":
            # Merge consecutive tool results into one user turn (Anthropic requirement).
            results: List[Dict[str, Any]] = []
            while i < len(messages) and messages[i].role == "tool":
                tr = messages[i]
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                })
                i += 1
            out.append({"role": "user", "content": results})

        else:
            i += 1

    return system, out


def to_anthropic_tools(oai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI-format tool schemas → Anthropic tool schemas."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in oai_tools
    ]


def from_anthropic_wire(resp: Any) -> ArchonMessage:
    """Anthropic SDK response → ArchonMessage.

    Handles ``text`` and ``tool_use`` content blocks.
    ``thinking`` blocks are ignored here; subclass and override to capture them.
    """
    text_content: Optional[str] = None
    tool_calls: List[ArchonToolCall] = []

    for block in resp.content:
        if block.type == "text":
            text_content = block.text
        elif block.type == "tool_use":
            tool_calls.append(ArchonToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input,
            ))
        # thinking blocks intentionally skipped — subclass to handle extended thinking

    return ArchonMessage(
        role="assistant",
        content=text_content,
        tool_calls=tool_calls or None,
    )


# ---------------------------------------------------------------------------
# AnthropicAdapter  (claude-*)
# ---------------------------------------------------------------------------

class AnthropicAdapter(LLMAdapter):

    async def complete(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> LLMResponse:
        import anthropic

        system, anthropic_messages = to_anthropic_wire(messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        resp = await anthropic.AsyncAnthropic().messages.create(**kwargs)
        usage = LLMUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
        )
        return LLMResponse(
            choices=[LLMChoice(message=self._from_wire(resp))],
            usage=usage,
            cost=estimate_cost(model, usage.prompt_tokens, usage.completion_tokens),
        )

    def _from_wire(self, resp: Any) -> ArchonMessage:
        """Overridable so subclasses can handle extended thinking blocks."""
        return from_anthropic_wire(resp)
