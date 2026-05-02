"""Tests for prompt caching wire format and cost accounting."""

from __future__ import annotations

import pytest

from archon.llm._base import LLMUsage, estimate_cost
from archon.llm.anthropic import AnthropicAdapter, to_anthropic_wire
from archon.types import ArchonMessage


class TestAnthropicCacheControl:
    def test_kwargs_should_wrap_system_in_cache_breakpoint(self):
        adapter = AnthropicAdapter()
        kwargs = adapter._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[
                ArchonMessage(role="system", content="You are helpful."),
                ArchonMessage(role="user", content="hi"),
            ],
            tools=None,
            temperature=None,
            top_p=None,
        )
        system = kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are helpful."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_kwargs_should_omit_system_field_when_no_system_message(self):
        adapter = AnthropicAdapter()
        kwargs = adapter._build_kwargs(
            model="claude-sonnet-4-6",
            messages=[ArchonMessage(role="user", content="hi")],
            tools=None,
            temperature=None,
            top_p=None,
        )
        assert "system" not in kwargs


class TestCacheCostAccounting:
    def test_cached_tokens_priced_at_cache_read_rate(self):
        # Sonnet 4.6: input 3.00, cache_read 0.30 → 10x discount
        regular = estimate_cost(
            "claude-sonnet-4-6",
            prompt_tokens=10_000,
            completion_tokens=0,
            cached_tokens=0,
        )
        with_cache = estimate_cost(
            "claude-sonnet-4-6",
            prompt_tokens=10_000,
            completion_tokens=0,
            cached_tokens=10_000,
        )
        # All input was cached → cost should be 10x cheaper.
        assert with_cache < regular
        assert pytest.approx(regular / with_cache, rel=0.01) == 10.0

    def test_cache_write_tokens_priced_at_premium(self):
        # Sonnet 4.6: input 3.00, cache_write 3.75 → 1.25x premium
        regular = estimate_cost(
            "claude-sonnet-4-6",
            prompt_tokens=10_000,
            completion_tokens=0,
        )
        with_write = estimate_cost(
            "claude-sonnet-4-6",
            prompt_tokens=10_000,
            completion_tokens=0,
            cache_write_tokens=10_000,
        )
        # Cache write costs 1.25x input.
        assert with_write > regular
        assert pytest.approx(with_write / regular, rel=0.01) == 1.25

    def test_partial_cache_subtracts_from_regular_input(self):
        full_regular = estimate_cost(
            "claude-sonnet-4-6",
            prompt_tokens=10_000,
            completion_tokens=0,
        )
        half_cached = estimate_cost(
            "claude-sonnet-4-6",
            prompt_tokens=10_000,
            completion_tokens=0,
            cached_tokens=5_000,
        )
        # Half regular + half cached should be cheaper than all regular.
        assert half_cached < full_regular


class TestUsageHasCacheFields:
    def test_default_zeros(self):
        u = LLMUsage()
        assert u.cached_tokens == 0
        assert u.cache_write_tokens == 0
