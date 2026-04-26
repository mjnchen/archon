"""OpenAI adapters — Chat Completions, Reasoning (o1/o3), and Responses API."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from archon.types import ArchonMessage, ArchonToolCall
from archon.llm._base import (
    LLMAdapter,
    LLMChoice,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    estimate_cost,
)


def _openai_cached_tokens(usage: Any) -> int:
    """Extract cached-prompt token count from an OpenAI usage object.

    OpenAI reports cache hits via ``usage.prompt_tokens_details.cached_tokens``
    on Chat Completions and ``usage.input_tokens_details.cached_tokens`` on
    the Responses API. Returns 0 when unavailable.
    """
    details = getattr(usage, "prompt_tokens_details", None) or getattr(
        usage, "input_tokens_details", None
    )
    if details is None:
        return 0
    return getattr(details, "cached_tokens", 0) or 0


# ---------------------------------------------------------------------------
# Wire format helpers — OpenAI Chat Completions
# ---------------------------------------------------------------------------

def to_openai_wire(messages: List[ArchonMessage]) -> List[Dict[str, Any]]:
    """ArchonMessage list → OpenAI Chat Completions message dicts."""
    out = []
    for msg in messages:
        d: Dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            d["content"] = msg.content
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id
        out.append(d)
    return out


def from_openai_wire(raw_msg: Any) -> ArchonMessage:
    """OpenAI SDK response message → ArchonMessage."""
    tool_calls = None
    if getattr(raw_msg, "tool_calls", None):
        tool_calls = [
            ArchonToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in raw_msg.tool_calls
        ]
    return ArchonMessage(
        role="assistant",
        content=raw_msg.content,
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# Wire format helpers — OpenAI Responses API
# ---------------------------------------------------------------------------

def to_responses_wire(
    messages: List[ArchonMessage],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """ArchonMessage list → (instructions, Responses API input list).

    Differences from Chat Completions wire format:
      • System message → ``instructions`` parameter (not an input item)
      • Assistant tool calls → separate ``function_call`` items (not nested)
      • Tool results     → ``function_call_output`` items (not role="tool")
    """
    instructions: Optional[str] = None
    items: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            instructions = msg.content

        elif msg.role == "user":
            items.append({"role": "user", "content": msg.content})

        elif msg.role == "assistant":
            if msg.content:
                items.append({"role": "assistant", "content": msg.content})
            for tc in msg.tool_calls or []:
                items.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                })

        elif msg.role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.content or "",
            })

    return instructions, items


def from_responses_wire(response: Any) -> ArchonMessage:
    """OpenAI Responses API response object → ArchonMessage.

    The response ``output`` is a flat list of typed items; we merge text
    messages and function-call items into a single ArchonMessage.
    """
    text_content: Optional[str] = None
    tool_calls: List[ArchonToolCall] = []

    for item in response.output:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for part in getattr(item, "content", []):
                if getattr(part, "type", None) == "output_text":
                    text_content = part.text

        elif item_type == "function_call":
            tool_calls.append(ArchonToolCall(
                id=item.call_id,
                name=item.name,
                arguments=json.loads(item.arguments),
            ))

    return ArchonMessage(
        role="assistant",
        content=text_content,
        tool_calls=tool_calls or None,
    )


# ---------------------------------------------------------------------------
# OpenAIChatAdapter  (gpt-4o, gpt-4o-mini, gpt-4-turbo, …)
# ---------------------------------------------------------------------------

class OpenAIChatAdapter(LLMAdapter):

    async def complete(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        import openai

        kwargs = self._build_kwargs(model, messages, tools, temperature, top_p, output_schema)
        resp = await openai.AsyncOpenAI().chat.completions.create(**kwargs)
        cached = _openai_cached_tokens(resp.usage)
        usage = LLMUsage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            cached_tokens=cached,
        )
        return LLMResponse(
            choices=[LLMChoice(message=from_openai_wire(resp.choices[0].message))],
            usage=usage,
            cost=estimate_cost(
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
            ),
        )

    def _build_kwargs(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": to_openai_wire(messages),
        }
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if output_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "archon_output", "schema": output_schema},
            }
        return kwargs

    async def astream(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        import openai

        kwargs = self._build_kwargs(model, messages, tools, temperature, top_p, output_schema)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        # Partial tool-call fragments keyed by index.
        tc_buf: Dict[int, Dict[str, Any]] = {}
        final_usage: Optional[LLMUsage] = None

        stream = await openai.AsyncOpenAI().chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.usage is not None:
                final_usage = LLMUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=_openai_cached_tokens(chunk.usage),
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield LLMStreamEvent(kind="text_delta", text=delta.content)
            if getattr(delta, "tool_calls", None):
                for d in delta.tool_calls:
                    buf = tc_buf.setdefault(d.index, {"id": None, "name": None, "args": ""})
                    if d.id:
                        buf["id"] = d.id
                    if d.function:
                        if d.function.name:
                            buf["name"] = d.function.name
                        if d.function.arguments:
                            buf["args"] += d.function.arguments

        for _, buf in sorted(tc_buf.items()):
            if not (buf["id"] and buf["name"]):
                continue
            try:
                args = json.loads(buf["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield LLMStreamEvent(
                kind="tool_call_complete",
                tool_call=ArchonToolCall(id=buf["id"], name=buf["name"], arguments=args),
            )

        cost = 0.0
        if final_usage:
            cost = estimate_cost(
                model,
                final_usage.prompt_tokens,
                final_usage.completion_tokens,
                cached_tokens=final_usage.cached_tokens,
            )
        yield LLMStreamEvent(kind="done", usage=final_usage, cost=cost)


# ---------------------------------------------------------------------------
# OpenAIReasoningAdapter  (o1, o1-mini, o3, o3-mini, …)
#
# Differences from Chat Completions:
#   • system role → developer role  (wire-level remap, not an ArchonMessage concept)
#   • temperature and top_p are not accepted
# ---------------------------------------------------------------------------

class OpenAIReasoningAdapter(OpenAIChatAdapter):

    def _build_kwargs(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        wire = to_openai_wire(messages)
        wire = [
            {**m, "role": "developer"} if m["role"] == "system" else m
            for m in wire
        ]
        kwargs: Dict[str, Any] = {"model": model, "messages": wire}
        if tools:
            kwargs["tools"] = tools
        if output_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "archon_output", "schema": output_schema},
            }
        # temperature and top_p intentionally omitted — not supported
        return kwargs


# ---------------------------------------------------------------------------
# OpenAIResponsesAdapter  (any model via the /v1/responses endpoint)
#
# Differences from Chat Completions:
#   • system message → ``instructions`` parameter
#   • tool calls and tool results are flat input items (not nested under role)
#   • response object is ``response.output``, not ``response.choices``
# ---------------------------------------------------------------------------

class OpenAIResponsesAdapter(LLMAdapter):

    async def complete(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        import openai

        instructions, input_items = to_responses_wire(messages)
        kwargs: Dict[str, Any] = {"model": model, "input": input_items}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if output_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "archon_output",
                    "schema": output_schema,
                }
            }

        resp = await openai.AsyncOpenAI().responses.create(**kwargs)
        cached = _openai_cached_tokens(resp.usage)
        usage = LLMUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.total_tokens,
            cached_tokens=cached,
        )
        return LLMResponse(
            choices=[LLMChoice(message=from_responses_wire(resp))],
            usage=usage,
            cost=estimate_cost(
                model,
                usage.prompt_tokens,
                usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
            ),
        )
