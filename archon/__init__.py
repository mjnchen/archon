"""Archon — Minimal enterprise agent framework."""

from dotenv import load_dotenv

load_dotenv()

__version__ = "0.1.0"

# Core
from archon.agent import Agent
from archon.hooks import AgentHooks
from archon.session import Session
from archon.state import AgentState
from archon.tools import ToolRegistry
from archon.types import (
    AgentConfig,
    AgentEvent,
    AgentResult,
    ApprovalPolicy,
    AuditEvent,
    AuditEventType,
    CompleteEvent,
    GuardrailResult,
    IterationEvent,
    OrchestrationResult,
    RawHttpRecord,
    Role,
    StepType,
    TenantContext,
    TextDeltaEvent,
    TokenUsage,
    ToolCallEvent,
    ToolDef,
    ToolEndEvent,
    ToolStartEvent,
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
    "Agent", "AgentHooks", "AgentState", "Session", "ToolRegistry",
    # Types
    "AgentConfig", "AgentResult", "ApprovalPolicy", "AuditEvent", "AuditEventType",
    "GuardrailResult", "OrchestrationResult", "RawHttpRecord", "Role", "StepType",
    "TenantContext", "TokenUsage", "ToolDef", "TraceStep",
    # Streaming events
    "AgentEvent", "IterationEvent", "TextDeltaEvent", "ToolCallEvent",
    "ToolStartEvent", "ToolEndEvent", "CompleteEvent",
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
