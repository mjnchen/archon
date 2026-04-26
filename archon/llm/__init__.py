"""archon.llm — provider-agnostic LLM client.

Public surface::

    from archon.llm import acompletion, LLMAdapter, LLMResponse

Adding a new provider or model family:
  1. Create ``archon/llm/<provider>.py`` with a class that extends ``LLMAdapter``
  2. Prepend ``("<model-prefix>", YourAdapter())`` to ``_REGISTRY`` below
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from archon.types import ArchonMessage
from archon.llm._base import (
    FINAL_OUTPUT_TOOL_NAME,
    LLMAdapter,
    LLMChoice,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    estimate_cost,
)
from archon.llm.openai import OpenAIChatAdapter, OpenAIReasoningAdapter, OpenAIResponsesAdapter
from archon.llm.anthropic import AnthropicAdapter
from archon.llm.gemini import GeminiAdapter


# ---------------------------------------------------------------------------
# Adapter registry
#
# Entries: (model-name prefix, adapter instance)
# Resolution: all matching prefixes → longest wins.
# More specific prefixes must appear before less specific ones for clarity,
# though the max-length selection makes order irrelevant for correctness.
# ---------------------------------------------------------------------------

_REGISTRY: List[Tuple[str, LLMAdapter]] = [
    ("o1",     OpenAIReasoningAdapter()),
    ("o3",     OpenAIReasoningAdapter()),
    ("gpt-",   OpenAIChatAdapter()),
    ("claude", AnthropicAdapter()),
    ("gemini", GeminiAdapter()),
]

_DEFAULT: LLMAdapter = OpenAIChatAdapter()


def _resolve(model: str) -> LLMAdapter:
    matches = [(p, a) for p, a in _REGISTRY if model.startswith(p)]
    if not matches:
        return _DEFAULT
    _, adapter = max(matches, key=lambda x: len(x[0]))
    return adapter


# ---------------------------------------------------------------------------
# Provider base URLs  (used by observability for trace records)
# ---------------------------------------------------------------------------

_BASE_URLS: Dict[str, str] = {
    "o1":     "https://api.openai.com/v1",
    "o3":     "https://api.openai.com/v1",
    "gpt-":   "https://api.openai.com/v1",
    "claude": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def provider_base_url(model: str) -> str:
    matches = [(p, url) for p, url in _BASE_URLS.items() if model.startswith(p)]
    if not matches:
        return ""
    _, url = max(matches, key=lambda x: len(x[0]))
    return url


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def acompletion(
    model: str,
    messages: List[ArchonMessage],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    max_attempts: int = 2,
) -> LLMResponse:
    """Resolve the adapter for *model* and return a normalised LLMResponse.

    Retries once by default on transient errors (rate limit, 5xx, connection).
    Set ``max_attempts=1`` to disable retries.
    """
    from archon.retry import with_retry

    adapter = _resolve(model)
    return await with_retry(
        lambda: adapter.complete(model, messages, tools, temperature, top_p, output_schema),
        max_attempts=max_attempts,
    )


async def astream(
    model: str,
    messages: List[ArchonMessage],
    tools: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    output_schema: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[LLMStreamEvent]:
    """Streaming counterpart to :func:`acompletion`.

    Yields LLMStreamEvents as the provider emits them. Retries are *not*
    applied here — a retry on a partial stream would be observable to the
    caller and semantically confusing.
    """
    async for event in _resolve(model).astream(
        model, messages, tools, temperature, top_p, output_schema
    ):
        yield event


__all__ = [
    "acompletion",
    "astream",
    "provider_base_url",
    "FINAL_OUTPUT_TOOL_NAME",
    "LLMAdapter",
    "LLMChoice",
    "LLMResponse",
    "LLMStreamEvent",
    "LLMUsage",
    "OpenAIChatAdapter",
    "OpenAIReasoningAdapter",
    "OpenAIResponsesAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
]
