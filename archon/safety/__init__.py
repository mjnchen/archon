"""archon.safety — guardrails, human-in-the-loop, and access control."""

from archon.safety.guardrails import (
    ContentPolicyGuardrail,
    DangerousToolCallGuardrail,
    Guardrail,
    GuardrailPipeline,
    PIIDetector,
    ToolCallGuardrail,
)
from archon.safety.hitl import (
    ApprovalHandler,
    AutoApproveHandler,
    CallbackApprovalHandler,
    ConsoleApprovalHandler,
    HumanApprovalManager,
)
from archon.safety.access import require_permission, require_role

__all__ = [
    # Guardrails
    "ContentPolicyGuardrail",
    "DangerousToolCallGuardrail",
    "Guardrail",
    "GuardrailPipeline",
    "PIIDetector",
    "ToolCallGuardrail",
    # HITL
    "ApprovalHandler",
    "AutoApproveHandler",
    "CallbackApprovalHandler",
    "ConsoleApprovalHandler",
    "HumanApprovalManager",
    # Access
    "require_permission",
    "require_role",
]
