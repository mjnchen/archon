"""Tests for GuardrailPipeline — input/output/tool-call checks."""

import pytest

from archon.exceptions import GuardrailBlocked
from archon.guardrails import (
    ContentPolicyGuardrail,
    DangerousToolCallGuardrail,
    GuardrailPipeline,
    PIIDetector,
)


class TestPIIDetector:
    @pytest.mark.asyncio
    async def test_should_detect_email(self):
        g = PIIDetector()
        result = await g.check("Contact me at user@example.com")
        assert not result.allowed
        assert "email" in result.reason

    @pytest.mark.asyncio
    async def test_should_detect_phone(self):
        g = PIIDetector()
        result = await g.check("Call 555-123-4567")
        assert not result.allowed
        assert "phone" in result.reason

    @pytest.mark.asyncio
    async def test_should_detect_ssn(self):
        g = PIIDetector()
        result = await g.check("SSN is 123-45-6789")
        assert not result.allowed
        assert "SSN" in result.reason

    @pytest.mark.asyncio
    async def test_should_allow_clean_text(self):
        g = PIIDetector()
        result = await g.check("What is the weather in Boston?")
        assert result.allowed


class TestContentPolicy:
    @pytest.mark.asyncio
    async def test_should_block_keyword(self):
        g = ContentPolicyGuardrail(blocked_keywords=["password", "secret"])
        result = await g.check("The password is hunter2")
        assert not result.allowed
        assert "password" in result.reason

    @pytest.mark.asyncio
    async def test_should_allow_clean_text(self):
        g = ContentPolicyGuardrail(blocked_keywords=["password"])
        result = await g.check("Hello world")
        assert result.allowed


class TestDangerousToolCall:
    @pytest.mark.asyncio
    async def test_should_block_drop_table(self):
        g = DangerousToolCallGuardrail()
        result = await g.check("run_sql", {"query": "DROP TABLE users"})
        assert not result.allowed

    @pytest.mark.asyncio
    async def test_should_allow_safe_sql(self):
        g = DangerousToolCallGuardrail()
        result = await g.check("run_sql", {"query": "SELECT * FROM users"})
        assert result.allowed


class TestGuardrailPipeline:
    @pytest.mark.asyncio
    async def test_should_raise_on_input_denial(self):
        pipeline = GuardrailPipeline(
            input_guardrails=[PIIDetector()],
        )
        with pytest.raises(GuardrailBlocked) as exc_info:
            await pipeline.check_input("Email me at test@example.com")
        assert "pii_detector" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_should_pass_clean_input(self):
        pipeline = GuardrailPipeline(
            input_guardrails=[PIIDetector()],
        )
        await pipeline.check_input("What is 2 + 2?")

    @pytest.mark.asyncio
    async def test_should_check_output(self):
        pipeline = GuardrailPipeline(
            output_guardrails=[ContentPolicyGuardrail(blocked_keywords=["classified"])],
        )
        with pytest.raises(GuardrailBlocked):
            await pipeline.check_output("This is classified information")

    @pytest.mark.asyncio
    async def test_should_check_tool_call(self):
        pipeline = GuardrailPipeline(
            tool_call_guardrails=[DangerousToolCallGuardrail()],
        )
        with pytest.raises(GuardrailBlocked):
            await pipeline.check_tool_call("sql", {"query": "DELETE FROM logs"})

    @pytest.mark.asyncio
    async def test_should_allow_safe_tool_call(self):
        pipeline = GuardrailPipeline(
            tool_call_guardrails=[DangerousToolCallGuardrail()],
        )
        await pipeline.check_tool_call("sql", {"query": "SELECT 1"})
