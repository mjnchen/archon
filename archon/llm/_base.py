"""LLM base types — adapter protocol, response envelope, and cost estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from archon.types import ArchonMessage, ArchonToolCall


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0          # prompt tokens served from cache
    cache_write_tokens: int = 0     # prompt tokens written to cache (Anthropic)


class LLMChoice(BaseModel):
    message: ArchonMessage


class LLMResponse(BaseModel):
    choices: List[LLMChoice]
    usage: LLMUsage
    cost: float = 0.0


class LLMStreamEvent(BaseModel):
    """Adapter-level stream event. Normalised across providers.

    Adapters accumulate partial tool-call fragments internally and only emit
    ``tool_call_complete`` once a call's id/name/arguments are fully assembled.
    The final ``done`` event should carry the aggregated usage and cost.
    """

    kind: Literal["text_delta", "tool_call_complete", "done"]
    text: Optional[str] = None
    tool_call: Optional[ArchonToolCall] = None
    usage: Optional[LLMUsage] = None
    cost: float = 0.0


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

class LLMAdapter(ABC):
    """Owns the full conversion path for one model family.

    Subclass and override to handle family-specific wire quirks.
    Register in the ``_REGISTRY`` in ``archon/llm/__init__.py``.
    """

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        ...

    async def astream(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Default: call ``complete`` and yield equivalent events.

        Adapters that support real token streaming should override this.
        The default still satisfies ``Agent.astream`` — callers just see one
        big text_delta per response instead of progressive deltas.
        """
        resp = await self.complete(
            model, messages, tools, temperature, top_p, output_schema
        )
        msg = resp.choices[0].message
        if msg.content:
            yield LLMStreamEvent(kind="text_delta", text=msg.content)
        for tc in msg.tool_calls or []:
            yield LLMStreamEvent(kind="tool_call_complete", tool_call=tc)
        yield LLMStreamEvent(kind="done", usage=resp.usage, cost=resp.cost)


# ---------------------------------------------------------------------------
# Synthetic tool used by providers that lack a native JSON-Schema response
# mode (Anthropic). The agent loop detects a call to this name and treats
# its arguments as the structured final output.
# ---------------------------------------------------------------------------

FINAL_OUTPUT_TOOL_NAME = "__archon_final_output__"
FINAL_OUTPUT_TOOL_DESCRIPTION = (
    "Call this when your answer is ready. The arguments you pass become the "
    "final structured output returned to the caller."
)


# ---------------------------------------------------------------------------
# Cost estimation  (USD per 1 M tokens: input, output)
# ---------------------------------------------------------------------------

#
# Rates: (input, output, cache_read, cache_write) per 1M tokens.
# cache_read is the discounted price for tokens served from cache.
# cache_write only applies to Anthropic (ephemeral cache population).
# For providers without a cache-write premium, cache_write equals input.
#
_COST_PER_1M: Dict[str, Tuple[float, float, float, float]] = {
    # OpenAI chat — automatic caching at ~0.5x input
    "gpt-4o":            (2.50,  10.00, 1.25,   2.50),
    "gpt-4o-mini":       (0.15,   0.60, 0.075,  0.15),
    "gpt-4-turbo":       (10.00, 30.00, 5.00,   10.00),
    # OpenAI reasoning
    "o1":                (15.00, 60.00, 7.50,   15.00),
    "o1-mini":           (3.00,  12.00, 1.50,   3.00),
    "o3":                (10.00, 40.00, 5.00,   10.00),
    "o3-mini":           (1.10,   4.40, 0.55,   1.10),
    # Anthropic — cache read 0.1x input, cache write 1.25x input
    "claude-opus-4-7":   (15.00, 75.00, 1.50,   18.75),
    "claude-sonnet-4-6": (3.00,  15.00, 0.30,    3.75),
    "claude-haiku-4-5":  (0.80,   4.00, 0.08,    1.00),
    # Gemini — cache ~0.25x input
    "gemini-2.5-pro":    (1.25,  10.00, 0.3125,  1.25),
    "gemini-2.0-flash":  (0.10,   0.40, 0.025,   0.10),
    "gemini-1.5-pro":    (1.25,   5.00, 0.3125,  1.25),
    "gemini-1.5-flash":  (0.075,  0.30, 0.01875, 0.075),
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Return estimated USD cost for a completion call. Returns 0.0 for unknown models.

    ``cached_tokens`` and ``cache_write_tokens`` are subtracted from the regular
    input price and re-billed at the cache_read / cache_write rates.
    """
    rates = _COST_PER_1M.get(model)
    if not rates:
        # Longest prefix wins — "gpt-4o-mini" beats "gpt-4o" for "gpt-4o-mini-2024-07-18"
        match = max(
            (p for p in _COST_PER_1M if model.startswith(p)),
            key=len,
            default=None,
        )
        rates = _COST_PER_1M.get(match) if match else None
    if not rates:
        return 0.0
    regular_input = max(0, prompt_tokens - cached_tokens - cache_write_tokens)
    return (
        regular_input * rates[0]
        + completion_tokens * rates[1]
        + cached_tokens * rates[2]
        + cache_write_tokens * rates[3]
    ) / 1_000_000
