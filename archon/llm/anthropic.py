"""Anthropic adapter — Claude model family."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from archon.types import ArchonMessage, ArchonToolCall
from archon.llm._base import (
    FINAL_OUTPUT_TOOL_DESCRIPTION,
    FINAL_OUTPUT_TOOL_NAME,
    LLMAdapter,
    LLMChoice,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    estimate_cost,
)


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

    def _build_kwargs(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system, anthropic_messages = to_anthropic_wire(messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        # Always mark the system prompt as an ephemeral cache breakpoint.
        # Cheap win with no downside: a cache miss costs 1.25x the one time,
        # a hit cuts the system-prompt input cost by 10x for 5 minutes.
        if system:
            kwargs["system"] = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]

        # Anthropic has no native JSON Schema response mode. The pattern is
        # tool-forcing: inject a synthetic tool whose input_schema is the
        # output schema; the agent loop treats a call to it as the
        # structured final answer.
        effective_tools = list(tools or [])
        if output_schema:
            effective_tools.append({
                "type": "function",
                "function": {
                    "name": FINAL_OUTPUT_TOOL_NAME,
                    "description": FINAL_OUTPUT_TOOL_DESCRIPTION,
                    "parameters": output_schema,
                },
            })
        if effective_tools:
            kwargs["tools"] = to_anthropic_tools(effective_tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        return kwargs

    async def complete(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        import anthropic

        kwargs = self._build_kwargs(
            model, messages, tools, temperature, top_p, output_schema
        )
        resp = await anthropic.AsyncAnthropic().messages.create(**kwargs)
        cached = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        # Anthropic reports input_tokens as the *uncached* portion; add cache
        # tokens back so prompt_tokens reflects the true input footprint.
        prompt_total = resp.usage.input_tokens + cached + cache_write
        usage = LLMUsage(
            prompt_tokens=prompt_total,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=prompt_total + resp.usage.output_tokens,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
        )
        return LLMResponse(
            choices=[LLMChoice(message=self._from_wire(resp))],
            usage=usage,
            cost=estimate_cost(
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            ),
        )

    def _from_wire(self, resp: Any) -> ArchonMessage:
        """Overridable so subclasses can handle extended thinking blocks."""
        return from_anthropic_wire(resp)

    async def astream(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        import anthropic

        kwargs = self._build_kwargs(
            model, messages, tools, temperature, top_p, output_schema
        )

        current_tool: Optional[Dict[str, Any]] = None
        tool_args_buf = ""
        final_message: Any = None

        async with anthropic.AsyncAnthropic().messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", None) == "tool_use":
                        current_tool = {"id": block.id, "name": block.name}
                        tool_args_buf = ""
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "text_delta":
                        yield LLMStreamEvent(kind="text_delta", text=delta.text)
                    elif dtype == "input_json_delta":
                        tool_args_buf += delta.partial_json
                elif etype == "content_block_stop":
                    if current_tool is not None:
                        try:
                            args = json.loads(tool_args_buf or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        yield LLMStreamEvent(
                            kind="tool_call_complete",
                            tool_call=ArchonToolCall(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                arguments=args,
                            ),
                        )
                        current_tool = None
                        tool_args_buf = ""
            final_message = await stream.get_final_message()

        cached = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(final_message.usage, "cache_creation_input_tokens", 0) or 0
        prompt_total = final_message.usage.input_tokens + cached + cache_write
        usage = LLMUsage(
            prompt_tokens=prompt_total,
            completion_tokens=final_message.usage.output_tokens,
            total_tokens=prompt_total + final_message.usage.output_tokens,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
        )
        cost = estimate_cost(
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        yield LLMStreamEvent(kind="done", usage=usage, cost=cost)
