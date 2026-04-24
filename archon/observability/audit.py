"""Audit trail — immutable event log with pluggable backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from archon.types import AuditEvent, AuditEventType, TenantContext

logger = logging.getLogger(__name__)


class AuditBackend(ABC):
    """Interface for audit event storage."""

    @abstractmethod
    async def record(self, event: AuditEvent) -> None: ...

    @abstractmethod
    async def query(
        self,
        trace_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]: ...

    @abstractmethod
    async def export(
        self,
        trace_id: str,
        fmt: Literal["json", "csv"] = "json",
    ) -> str: ...


class JsonLinesAuditBackend(AuditBackend):
    """Append-only JSON-lines file backend."""

    def __init__(self, path: Union[str, Path] = "audit.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, event: AuditEvent) -> None:
        line = event.model_dump_json() + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    async def query(
        self,
        trace_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        if not self._path.exists():
            return []
        events: List[AuditEvent] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = AuditEvent.model_validate_json(line)
                if trace_id and ev.trace_id != trace_id:
                    continue
                if event_type and ev.event_type != event_type:
                    continue
                if tenant_id and (not ev.tenant or ev.tenant.tenant_id != tenant_id):
                    continue
                events.append(ev)
                if len(events) >= limit:
                    break
        return events

    async def export(self, trace_id: str, fmt: Literal["json", "csv"] = "json") -> str:
        events = await self.query(trace_id=trace_id, limit=10_000)
        if fmt == "csv":
            header = "event_id,event_type,timestamp,trace_id,step_index,tenant_id,user_id\n"
            rows = [
                f"{e.event_id},{e.event_type.value},{e.timestamp.isoformat()},"
                f"{e.trace_id},{e.step_index},"
                f"{e.tenant.tenant_id if e.tenant else ''},"
                f"{e.tenant.user_id if e.tenant else ''}\n"
                for e in events
            ]
            return header + "".join(rows)
        return json.dumps([e.model_dump(mode="json") for e in events], indent=2)


class InMemoryAuditBackend(AuditBackend):
    """Simple in-memory store for tests."""

    def __init__(self) -> None:
        self._events: List[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self._events.append(event)

    async def query(
        self,
        trace_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        results = []
        for ev in self._events:
            if trace_id and ev.trace_id != trace_id:
                continue
            if event_type and ev.event_type != event_type:
                continue
            if tenant_id and (not ev.tenant or ev.tenant.tenant_id != tenant_id):
                continue
            results.append(ev)
            if len(results) >= limit:
                break
        return results

    async def export(self, trace_id: str, fmt: Literal["json", "csv"] = "json") -> str:
        events = await self.query(trace_id=trace_id, limit=10_000)
        return json.dumps([e.model_dump(mode="json") for e in events], indent=2)


class AuditTrail:
    """High-level audit API that wraps a backend.

    Usage::

        audit = AuditTrail(backend=JsonLinesAuditBackend("audit.jsonl"))
        audit.record_run_started(run_id, "my_agent", tenant)
    """

    def __init__(self, backend: Optional[AuditBackend] = None) -> None:
        self.backend = backend or InMemoryAuditBackend()
        self._step_counters: Dict[str, int] = {}

    def _next_step(self, trace_id: str) -> int:
        idx = self._step_counters.get(trace_id, 0)
        self._step_counters[trace_id] = idx + 1
        return idx

    def _emit(
        self,
        event_type: AuditEventType,
        trace_id: str,
        tenant: Optional[TenantContext],
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        import asyncio

        event = AuditEvent(
            event_type=event_type,
            trace_id=trace_id,
            step_index=self._next_step(trace_id),
            tenant=tenant,
            data=data or {},
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.backend.record(event))
        except RuntimeError:
            asyncio.run(self.backend.record(event))

    def record_run_started(self, run_id: str, agent_name: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.RUN_STARTED, run_id, tenant, {"agent_name": agent_name})

    def record_run_completed(self, run_id: str, stop_reason: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.RUN_COMPLETED, run_id, tenant, {"stop_reason": stop_reason})

    def record_run_failed(self, run_id: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.RUN_FAILED, run_id, tenant)

    def record_tool_invoke(self, run_id: str, tool_name: str, args: Dict[str, Any], tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.TOOL_INVOKE, run_id, tenant, {"tool_name": tool_name, "arguments": args})

    def record_tool_result(self, run_id: str, tool_name: str, result: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.TOOL_RESULT, run_id, tenant, {"tool_name": tool_name, "result": result})

    def record_handover(self, run_id: str, target: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.HANDOVER, run_id, tenant, {"target_agent": target})

    def record_approval_requested(self, run_id: str, tool_name: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.APPROVAL_REQUESTED, run_id, tenant, {"tool_name": tool_name})

    def record_approval_granted(self, run_id: str, tool_name: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.APPROVAL_GRANTED, run_id, tenant, {"tool_name": tool_name})

    def record_approval_denied(self, run_id: str, tool_name: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.APPROVAL_DENIED, run_id, tenant, {"tool_name": tool_name})

    def record_guardrail_blocked(self, run_id: str, guardrail_name: str, reason: str, tenant: Optional[TenantContext]) -> None:
        self._emit(AuditEventType.GUARDRAIL_BLOCKED, run_id, tenant, {"guardrail": guardrail_name, "reason": reason})
