"""Archon — Minimal enterprise agent framework."""

__version__ = "0.1.0"

# Core
from archon.agent import Agent
from archon.state import AgentState
from archon.tools import ToolRegistry
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

# Observability
from archon.observability import (
    ArchonLogger,
    AuditBackend,
    AuditTrail,
    InMemoryAuditBackend,
    JsonLinesAuditBackend,
)

# Safety
from archon.safety import (
    ContentPolicyGuardrail,
    DangerousToolCallGuardrail,
    Guardrail,
    GuardrailPipeline,
    PIIDetector,
    ToolCallGuardrail,
    ApprovalHandler,
    AutoApproveHandler,
    CallbackApprovalHandler,
    ConsoleApprovalHandler,
    HumanApprovalManager,
    require_permission,
    require_role,
)

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
    "Agent", "AgentState", "ToolRegistry",
    # Types
    "AgentConfig", "AgentResult", "ApprovalPolicy", "AuditEvent", "AuditEventType",
    "GuardrailResult", "OrchestrationResult", "RawHttpRecord", "Role", "StepType",
    "TenantContext", "TokenUsage", "ToolDef", "TraceStep",
    # Exceptions
    "AccessDenied", "ApprovalDenied", "ApprovalTimeout", "ArchonError",
    "BudgetExceeded", "GuardrailBlocked", "HandoverRequest",
    "MaxIterationsExceeded", "ToolExecutionError", "ToolNotFoundError",
    # Observability
    "ArchonLogger", "AuditBackend", "AuditTrail", "InMemoryAuditBackend", "JsonLinesAuditBackend",
    # Safety
    "ContentPolicyGuardrail", "DangerousToolCallGuardrail", "Guardrail", "GuardrailPipeline",
    "PIIDetector", "ToolCallGuardrail",
    "ApprovalHandler", "AutoApproveHandler", "CallbackApprovalHandler",
    "ConsoleApprovalHandler", "HumanApprovalManager",
    "require_permission", "require_role",
    # Orchestration
    "AgentRegistry", "FanOut", "Pipeline", "Supervisor", "run_with_handover",
    # Config
    "load_agent_configs", "load_guardrail_pipeline", "load_hitl_policies",
]
