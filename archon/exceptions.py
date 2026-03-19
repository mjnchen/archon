"""Custom exceptions for the Archon agent framework."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ArchonError(Exception):
    """Base exception for all Archon errors."""


class MaxIterationsExceeded(ArchonError):
    """Agent hit the maximum number of ReAct loop iterations."""

    def __init__(self, iterations: int, max_iterations: int):
        self.iterations = iterations
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent exceeded max iterations ({iterations}/{max_iterations})"
        )


class BudgetExceeded(ArchonError):
    """Agent hit the maximum cost budget for a single run."""

    def __init__(self, spent: float, max_cost: float):
        self.spent = spent
        self.max_cost = max_cost
        super().__init__(
            f"Agent exceeded budget (${spent:.4f}/${max_cost:.4f})"
        )


class GuardrailBlocked(ArchonError):
    """A guardrail denied the request."""

    def __init__(self, guardrail_name: str, reason: str):
        self.guardrail_name = guardrail_name
        self.reason = reason
        super().__init__(f"Guardrail '{guardrail_name}' blocked: {reason}")


class ApprovalDenied(ArchonError):
    """Human-in-the-loop approval was denied."""

    def __init__(self, tool_name: str, reason: str = ""):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Approval denied for tool '{tool_name}': {reason}")


class ApprovalTimeout(ArchonError):
    """Human-in-the-loop approval timed out."""

    def __init__(self, tool_name: str, timeout: float):
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"Approval timed out for tool '{tool_name}' after {timeout}s"
        )


class ToolExecutionError(ArchonError):
    """A tool failed during execution."""

    def __init__(self, tool_name: str, original_error: Exception):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {original_error}")


class ToolNotFoundError(ArchonError):
    """Requested tool is not registered."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' is not registered")


class HandoverRequest(ArchonError):
    """Raised when an agent requests handover to another agent.

    This is not really an *error* — it's a control-flow signal caught by the
    orchestration layer to switch the active agent.
    """

    def __init__(
        self,
        target_agent: str,
        context: Optional[Dict[str, Any]] = None,
        summary: str = "",
    ):
        self.target_agent = target_agent
        self.context = context or {}
        self.summary = summary
        super().__init__(f"Handover requested → '{target_agent}'")


class AccessDenied(ArchonError):
    """The tenant/user does not have permission for the requested action."""

    def __init__(self, action: str, role: str):
        self.action = action
        self.role = role
        super().__init__(f"Role '{role}' is not permitted to '{action}'")
