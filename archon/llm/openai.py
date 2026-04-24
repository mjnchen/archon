"""OpenAI adapters — Chat Completions, Reasoning (o1/o3), and Responses API."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from archon.types import ArchonMessage, ArchonToolCall
from archon.llm._base import LLMAdapter, LLMChoice, LLMResponse, LLMUsage, estimate_cost


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
    ) -> LLMResponse:
        import openai

        kwargs = self._build_kwargs(model, messages, tools, temperature, top_p)
        resp = await openai.AsyncOpenAI().chat.completions.create(**kwargs)
        usage = LLMUsage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        )
        return LLMResponse(
            choices=[LLMChoice(message=from_openai_wire(resp.choices[0].message))],
            usage=usage,
            cost=estimate_cost(model, usage.prompt_tokens, usage.completion_tokens),
        )

    def _build_kwargs(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
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
        return kwargs


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
    ) -> Dict[str, Any]:
        wire = to_openai_wire(messages)
        wire = [
            {**m, "role": "developer"} if m["role"] == "system" else m
            for m in wire
        ]
        kwargs: Dict[str, Any] = {"model": model, "messages": wire}
        if tools:
            kwargs["tools"] = tools
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

        resp = await openai.AsyncOpenAI().responses.create(**kwargs)
        usage = LLMUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.total_tokens,
        )
        return LLMResponse(
            choices=[LLMChoice(message=from_responses_wire(resp))],
            usage=usage,
            cost=estimate_cost(model, usage.prompt_tokens, usage.completion_tokens),
        )
