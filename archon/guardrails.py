"""Guardrails — input/output/tool-call validation pipeline with pluggable checks."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from archon.exceptions import GuardrailBlocked
from archon.types import GuardrailResult, TenantContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrail protocol
# ---------------------------------------------------------------------------

class Guardrail(ABC):
    """Base class for all guardrails."""

    name: str = "base"

    @abstractmethod
    async def check(self, content: str, context: Optional[TenantContext] = None) -> GuardrailResult:
        ...


class ToolCallGuardrail(ABC):
    """Base class for guardrails that inspect tool call arguments."""

    name: str = "base_tool"

    @abstractmethod
    async def check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> GuardrailResult:
        ...


# ---------------------------------------------------------------------------
# Built-in guardrails
# ---------------------------------------------------------------------------

class PIIDetector(Guardrail):
    """Regex-based PII detector for emails, phone numbers, and SSNs."""

    name = "pii_detector"

    _PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone number"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    ]

    async def check(self, content: str, context: Optional[TenantContext] = None) -> GuardrailResult:
        for pattern, label in self._PATTERNS:
            if re.search(pattern, content):
                return GuardrailResult(
                    allowed=False,
                    reason=f"Detected potential {label} in content",
                )
        return GuardrailResult(allowed=True)


class ContentPolicyGuardrail(Guardrail):
    """Keyword blocklist guardrail."""

    name = "content_policy"

    def __init__(self, blocked_keywords: Optional[List[str]] = None) -> None:
        self.blocked_keywords = [kw.lower() for kw in (blocked_keywords or [])]

    async def check(self, content: str, context: Optional[TenantContext] = None) -> GuardrailResult:
        lower = content.lower()
        for kw in self.blocked_keywords:
            if kw in lower:
                return GuardrailResult(
                    allowed=False,
                    reason=f"Content contains blocked keyword: '{kw}'",
                )
        return GuardrailResult(allowed=True)


class DangerousToolCallGuardrail(ToolCallGuardrail):
    """Blocks tool calls whose arguments match dangerous patterns."""

    name = "dangerous_tool_call"

    _DANGEROUS_SQL = re.compile(
        r"\b(DROP|DELETE|TRUNCATE|ALTER)\b", re.IGNORECASE
    )

    async def check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> GuardrailResult:
        for value in arguments.values():
            if isinstance(value, str) and self._DANGEROUS_SQL.search(value):
                return GuardrailResult(
                    allowed=False,
                    reason=f"Tool '{tool_name}' arguments contain dangerous SQL pattern",
                )
        return GuardrailResult(allowed=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class GuardrailPipeline:
    """Runs a sequence of guardrails, raising on first denial.

    Usage::

        pipeline = GuardrailPipeline(
            input_guardrails=[PIIDetector()],
            output_guardrails=[ContentPolicyGuardrail(blocked_keywords=["password"])],
            tool_call_guardrails=[DangerousToolCallGuardrail()],
        )
        await pipeline.check_input("hello")
    """

    def __init__(
        self,
        input_guardrails: Optional[List[Guardrail]] = None,
        output_guardrails: Optional[List[Guardrail]] = None,
        tool_call_guardrails: Optional[List[ToolCallGuardrail]] = None,
    ) -> None:
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.tool_call_guardrails = tool_call_guardrails or []

    async def check_input(self, content: str, context: Optional[TenantContext] = None) -> None:
        for g in self.input_guardrails:
            result = await g.check(content, context)
            if not result.allowed:
                raise GuardrailBlocked(g.name, result.reason)

    async def check_output(self, content: str, context: Optional[TenantContext] = None) -> None:
        for g in self.output_guardrails:
            result = await g.check(content, context)
            if not result.allowed:
                raise GuardrailBlocked(g.name, result.reason)

    async def check_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> None:
        for g in self.tool_call_guardrails:
            result = await g.check(tool_name, arguments, context)
            if not result.allowed:
                raise GuardrailBlocked(g.name, result.reason)
