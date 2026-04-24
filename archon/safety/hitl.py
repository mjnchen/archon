"""Human-in-the-Loop — policy-based approval middleware for tool execution."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional

from archon.exceptions import ApprovalDenied, ApprovalTimeout
from archon.types import ApprovalPolicy, TenantContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Approval handler protocol
# ---------------------------------------------------------------------------

class ApprovalHandler(ABC):
    """Interface that concrete approval channels implement."""

    @abstractmethod
    async def request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> bool:
        """Return ``True`` if approved, ``False`` if denied."""
        ...


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------

class ConsoleApprovalHandler(ApprovalHandler):
    """Prompts the user on stdin. Useful for local development."""

    async def request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> bool:
        loop = asyncio.get_running_loop()
        prompt = (
            f"\n[HITL] Agent wants to call '{tool_name}' with args: {arguments}\n"
            f"  Approve? (y/n): "
        )
        answer = await loop.run_in_executor(None, lambda: input(prompt).strip().lower())
        return answer in ("y", "yes")


class CallbackApprovalHandler(ApprovalHandler):
    """Delegates approval to a user-provided async callback."""

    def __init__(
        self,
        callback: Callable[
            [str, Dict[str, Any], Optional[TenantContext]],
            Coroutine[Any, Any, bool],
        ],
    ) -> None:
        self._callback = callback

    async def request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> bool:
        return await self._callback(tool_name, arguments, context)


class AutoApproveHandler(ApprovalHandler):
    """Always approves — for testing / low-risk environments."""

    async def request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> bool:
        return True


# ---------------------------------------------------------------------------
# HITL manager
# ---------------------------------------------------------------------------

class HumanApprovalManager:
    """Evaluates policies and routes approval requests to a handler.

    Usage::

        manager = HumanApprovalManager(
            policies=[
                ApprovalPolicy(tool_name_patterns=["send_*", "delete_*"]),
            ],
            handler=ConsoleApprovalHandler(),
        )
        await manager.check("send_email", {"to": "bob@example.com"})
    """

    def __init__(
        self,
        policies: Optional[List[ApprovalPolicy]] = None,
        handler: Optional[ApprovalHandler] = None,
        default_timeout: float = 300.0,
    ) -> None:
        self.policies = policies or []
        self.handler = handler or AutoApproveHandler()
        self.default_timeout = default_timeout

    def requires_approval(self, tool_name: str) -> bool:
        """Check whether any policy requires approval for *tool_name*."""
        for policy in self.policies:
            if policy.always_require:
                return True
            for pattern in policy.tool_name_patterns:
                if fnmatch.fnmatch(tool_name, pattern):
                    return True
        return False

    async def check(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[TenantContext] = None,
    ) -> None:
        """Check approval for a tool call; raise on denial or timeout."""
        if not self.requires_approval(tool_name):
            return

        timeout = self._get_timeout(tool_name)

        try:
            approved = await asyncio.wait_for(
                self.handler.request_approval(tool_name, arguments, context),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise ApprovalTimeout(tool_name, timeout)

        if not approved:
            raise ApprovalDenied(tool_name, "Human reviewer denied the action")

    def _get_timeout(self, tool_name: str) -> float:
        for policy in self.policies:
            for pattern in policy.tool_name_patterns:
                if fnmatch.fnmatch(tool_name, pattern):
                    return policy.timeout
        return self.default_timeout
