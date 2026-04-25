"""Core types for the Archon agent framework."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Canonical message types (provider-neutral)
# ---------------------------------------------------------------------------

class ArchonToolCall(BaseModel):
    """A tool invocation requested by the assistant."""
    id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)  # always parsed, never a JSON string


class ArchonMessage(BaseModel):
    """Provider-neutral conversation message.

    Each provider adapter converts between this type and its own wire format.
    Role semantics:
      system    — top-level instructions, never sent as a turn
      user      — human turn (or injected tool results)
      assistant — model turn, optionally containing tool_calls
      tool      — tool execution result (identified by tool_call_id)
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List["ArchonToolCall"]] = None
    tool_call_id: Optional[str] = None  # present only when role == "tool"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_INVOKE = "tool_invoke"
    TOOL_RESULT = "tool_result"
    HANDOVER = "handover"
    GUARDRAIL_CHECK = "guardrail_check"
    APPROVAL_REQUEST = "approval_request"


class AuditEventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    LLM_CALL = "llm_call"
    TOOL_INVOKE = "tool_invoke"
    TOOL_RESULT = "tool_result"
    HANDOVER = "handover"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    GUARDRAIL_BLOCKED = "guardrail_blocked"


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class ToolDef(BaseModel):
    """Definition of a tool that an agent can invoke."""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[float] = 30.0
    requires_approval: bool = False

    # Not serialized — the actual callable is stored on the registry, not here.
    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# Token usage & cost
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Trace / observability
# ---------------------------------------------------------------------------

class RawHttpRecord(BaseModel):
    """Captured raw HTTP request/response for a single LLM call."""

    api_base: str = ""
    request_body: Dict[str, Any] = Field(default_factory=dict)
    request_headers: Dict[str, Any] = Field(default_factory=dict)
    response_body: Optional[Dict[str, Any]] = None


class TraceStep(BaseModel):
    """One step in an agent run trace."""

    step_index: int = 0
    step_type: StepType = StepType.LLM_CALL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    input: Any = None
    output: Any = None
    raw_http: Optional[RawHttpRecord] = None
    token_usage: Optional[TokenUsage] = None
    cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str = "default"
    model: str = "gpt-4o-mini"
    system_prompt: str = ""
    max_iterations: int = 10
    max_cost: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tool_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Provider-enforced JSON Schema for the final answer. When set, the
    # adapter configures native constrained decoding (OpenAI/Gemini
    # response_format; Anthropic tool-forcing) and AgentResult.final_output
    # is populated with the parsed dict.
    output_schema: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Result returned by an agent run."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = ""
    output: str = ""
    messages: List[ArchonMessage] = Field(default_factory=list)
    trace: List[TraceStep] = Field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    iterations: int = 0
    stop_reason: str = "completed"

    # Populated when AgentConfig.output_schema is set. dict matching that schema.
    final_output: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class GuardrailResult(BaseModel):
    """Result of a guardrail check."""

    allowed: bool = True
    reason: str = ""
    modified_content: Optional[str] = None


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------

class ApprovalPolicy(BaseModel):
    """Policy defining when human approval is required."""

    tool_name_patterns: List[str] = Field(default_factory=list)
    always_require: bool = False
    timeout: float = 300.0


# ---------------------------------------------------------------------------
# Tenant / access
# ---------------------------------------------------------------------------

class TenantContext(BaseModel):
    """Multi-tenancy context threaded through all operations."""

    tenant_id: str = "default"
    user_id: str = "anonymous"
    role: Role = Role.OPERATOR

    def has_permission(self, action: str) -> bool:
        """Check whether this context permits *action*."""
        permissions: Dict[str, set] = {
            "run_agent": {Role.ADMIN, Role.OPERATOR},
            "view_traces": {Role.ADMIN, Role.OPERATOR, Role.VIEWER},
            "manage_agents": {Role.ADMIN},
            "manage_tenants": {Role.ADMIN},
            "export_audit": {Role.ADMIN},
        }
        allowed_roles = permissions.get(action, {Role.ADMIN})
        return self.role in allowed_roles


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditEvent(BaseModel):
    """A single immutable audit event."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str = ""
    step_index: int = 0
    tenant: Optional[TenantContext] = None
    data: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming events (yielded from Agent.astream)
# ---------------------------------------------------------------------------

class AgentEvent(BaseModel):
    """Base class for events yielded from Agent.astream.

    Subclasses discriminate on ``kind``. Use isinstance checks in consumers;
    the base class is rarely yielded directly.
    """

    kind: str


class IterationEvent(AgentEvent):
    kind: Literal["iteration"] = "iteration"
    n: int


class TextDeltaEvent(AgentEvent):
    kind: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallEvent(AgentEvent):
    """Emitted when the model's tool call has been fully assembled."""
    kind: Literal["tool_call"] = "tool_call"
    tool_call: "ArchonToolCall"


class ToolStartEvent(AgentEvent):
    """Emitted just before a tool is executed (after guardrails/HITL)."""
    kind: Literal["tool_start"] = "tool_start"
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolEndEvent(AgentEvent):
    kind: Literal["tool_end"] = "tool_end"
    tool_call_id: str
    tool_name: str
    output: str
    duration_ms: float


class CompleteEvent(AgentEvent):
    """Final event in a stream. Carries the full AgentResult."""
    kind: Literal["complete"] = "complete"
    result: "AgentResult"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class OrchestrationResult(BaseModel):
    """Result of a multi-agent orchestration run."""

    final_output: str = ""
    agent_results: List[AgentResult] = Field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)


# ---------------------------------------------------------------------------
# Type aliases for callables
# ---------------------------------------------------------------------------

ApprovalHandlerFn = Callable[
    [str, Dict[str, Any], Optional[TenantContext]],
    Coroutine[Any, Any, bool],
]

GuardrailFn = Callable[
    [str, Dict[str, Any]],
    Coroutine[Any, Any, GuardrailResult],
]
