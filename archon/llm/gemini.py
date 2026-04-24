"""Gemini adapter — Google models via OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from archon.types import ArchonMessage
from archon.llm._base import LLMAdapter, LLMChoice, LLMResponse, LLMUsage, estimate_cost
from archon.llm.openai import from_openai_wire, to_openai_wire


class GeminiAdapter(LLMAdapter):
    """Routes gemini-* models through Google's OpenAI-compatible endpoint.

    Requires ``GEMINI_API_KEY`` in the environment.
    Uses the same wire format as OpenAI Chat Completions, so it reuses
    ``to_openai_wire`` / ``from_openai_wire`` directly.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    async def complete(
        self,
        model: str,
        messages: List[ArchonMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> LLMResponse:
        import openai

        client = openai.AsyncOpenAI(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            base_url=self._BASE_URL,
        )
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

        resp = await client.chat.completions.create(**kwargs)
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
