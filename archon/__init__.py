"""Archon — Minimal enterprise agent framework built on LiteLLM."""

__version__ = "0.1.0"

# Core
from archon.agent import Agent
from archon.state import AgentState
from archon.tools import ToolRegistry
from archon.observer import ArchonLogger
from archon.types import (
    AgentConfig,
    AgentResult,
    ApprovalPolicy,
    AuditEvent,
    AuditEventType,
    GuardrailResult,
    OrchestrationResult,
    RawHttpRecord,
    Role,
    StepType,
    TenantContext,
    TokenUsage,
    ToolDef,
    TraceStep,
)
from archon.exceptions import (
    AccessDenied,
    ApprovalDenied,
    ApprovalTimeout,
    ArchonError,
    BudgetExceeded,
    GuardrailBlocked,
    HandoverRequest,
    MaxIterationsExceeded,
    ToolExecutionError,
    ToolNotFoundError,
)

# Enterprise
from archon.guardrails import (
    ContentPolicyGuardrail,
    DangerousToolCallGuardrail,
    Guardrail,
    GuardrailPipeline,
    PIIDetector,
    ToolCallGuardrail,
)
from archon.hitl import (
    ApprovalHandler,
    AutoApproveHandler,
    CallbackApprovalHandler,
    ConsoleApprovalHandler,
    HumanApprovalManager,
)
from archon.audit import (
    AuditBackend,
    AuditTrail,
    InMemoryAuditBackend,
    JsonLinesAuditBackend,
)
from archon.access import require_permission, require_role

# Orchestration
from archon.orchestrator import (
    AgentRegistry,
    FanOut,
    Pipeline,
    Supervisor,
    run_with_handover,
)

# Config
from archon.config import load_agent_configs, load_guardrail_pipeline, load_hitl_policies

__all__ = [
    # Core
    "Agent",
    "AgentState",
    "ToolRegistry",
    "ArchonLogger",
    # Types
    "AgentConfig",
    "AgentResult",
    "ApprovalPolicy",
    "AuditEvent",
    "AuditEventType",
    "GuardrailResult",
    "OrchestrationResult",
    "RawHttpRecord",
    "Role",
    "StepType",
    "TenantContext",
    "TokenUsage",
    "ToolDef",
    "TraceStep",
    # Exceptions
    "AccessDenied",
    "ApprovalDenied",
    "ApprovalTimeout",
    "ArchonError",
    "BudgetExceeded",
    "GuardrailBlocked",
    "HandoverRequest",
    "MaxIterationsExceeded",
    "ToolExecutionError",
    "ToolNotFoundError",
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
    # Audit
    "AuditBackend",
    "AuditTrail",
    "InMemoryAuditBackend",
    "JsonLinesAuditBackend",
    # Access
    "require_permission",
    "require_role",
    # Orchestration
    "AgentRegistry",
    "FanOut",
    "Pipeline",
    "Supervisor",
    "run_with_handover",
    # Config
    "load_agent_configs",
    "load_guardrail_pipeline",
    "load_hitl_policies",
]
