"""LLM base types — adapter protocol, response envelope, and cost estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from archon.types import ArchonMessage, ArchonToolCall


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMChoice(BaseModel):
    message: ArchonMessage


class LLMResponse(BaseModel):
    choices: List[LLMChoice]
    usage: LLMUsage
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
    ) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Cost estimation  (USD per 1 M tokens: input, output)
# ---------------------------------------------------------------------------

_COST_PER_1M: Dict[str, Tuple[float, float]] = {
    # OpenAI chat
    "gpt-4o":            (2.50,  10.00),
    "gpt-4o-mini":       (0.15,   0.60),
    "gpt-4-turbo":       (10.00, 30.00),
    # OpenAI reasoning
    "o1":                (15.00, 60.00),
    "o1-mini":           (3.00,  12.00),
    "o3":                (10.00, 40.00),
    "o3-mini":           (1.10,   4.40),
    # Anthropic
    "claude-opus-4-7":   (15.00, 75.00),
    "claude-sonnet-4-6": (3.00,  15.00),
    "claude-haiku-4-5":  (0.80,   4.00),
    # Gemini
    "gemini-2.5-pro":    (1.25,  10.00),
    "gemini-2.0-flash":  (0.10,   0.40),
    "gemini-1.5-pro":    (1.25,   5.00),
    "gemini-1.5-flash":  (0.075,  0.30),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a completion call. Returns 0.0 for unknown models."""
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
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000
