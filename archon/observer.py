"""Observability — LiteLLM CustomLogger that captures raw HTTP requests, step traces, and costs."""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger

from archon.types import RawHttpRecord, StepType, TokenUsage, TraceStep

logger = logging.getLogger(__name__)


class ArchonLogger(CustomLogger):
    """LiteLLM callback logger that captures per-call trace data.

    Each agent run is identified by a ``run_id``.  The agent sets the current
    run_id via :meth:`set_run_id` before making LLM calls.  The logger
    accumulates :class:`TraceStep` objects keyed by run_id.

    Usage::

        observer = ArchonLogger()
        observer.install()            # registers with litellm.callbacks
        observer.set_run_id("abc123") # set before each agent run
    """

    def __init__(self) -> None:
        super().__init__()
        self._traces: Dict[str, List[TraceStep]] = defaultdict(list)
        self._step_counters: Dict[str, int] = defaultdict(int)
        self._current_run_id: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Register this logger with LiteLLM and enable raw request logging."""
        litellm.log_raw_request_response = True
        if not isinstance(litellm.callbacks, list):
            litellm.callbacks = []
        if self not in litellm.callbacks:
            litellm.callbacks.append(self)

    def uninstall(self) -> None:
        if isinstance(litellm.callbacks, list) and self in litellm.callbacks:
            litellm.callbacks.remove(self)

    # ------------------------------------------------------------------
    # Run-id management (set by agent before each LLM call)
    # ------------------------------------------------------------------

    def set_run_id(self, run_id: str) -> None:
        self._current_run_id = run_id

    def clear_run_id(self) -> None:
        self._current_run_id = None

    # ------------------------------------------------------------------
    # Trace access
    # ------------------------------------------------------------------

    def get_trace(self, run_id: str) -> List[TraceStep]:
        return list(self._traces.get(run_id, []))

    def get_all_run_ids(self) -> List[str]:
        return list(self._traces.keys())

    def clear(self, run_id: Optional[str] = None) -> None:
        if run_id:
            self._traces.pop(run_id, None)
            self._step_counters.pop(run_id, None)
        else:
            self._traces.clear()
            self._step_counters.clear()

    # ------------------------------------------------------------------
    # Record arbitrary steps (called directly by the Agent)
    # ------------------------------------------------------------------

    def record_step(self, run_id: str, step: TraceStep) -> None:
        with self._lock:
            step.step_index = self._step_counters[run_id]
            self._step_counters[run_id] += 1
            self._traces[run_id].append(step)

    # ------------------------------------------------------------------
    # LiteLLM callback hooks
    # ------------------------------------------------------------------

    def log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._record_llm_step(kwargs, response_obj, start_time, end_time, success=True)

    def log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._record_llm_step(kwargs, response_obj, start_time, end_time, success=False)

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._record_llm_step(kwargs, response_obj, start_time, end_time, success=True)

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        self._record_llm_step(kwargs, response_obj, start_time, end_time, success=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_llm_step(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        *,
        success: bool,
    ) -> None:
        run_id = self._current_run_id
        if not run_id:
            return

        duration_ms = _duration_ms(start_time, end_time)

        raw_http = self._extract_raw_http(kwargs)
        token_usage = self._extract_token_usage(response_obj)
        cost = kwargs.get("response_cost", 0.0) or 0.0

        step = TraceStep(
            step_type=StepType.LLM_CALL,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            input=kwargs.get("messages"),
            output=_safe_response_text(response_obj),
            raw_http=raw_http,
            token_usage=token_usage,
            cost=cost,
            metadata={"success": success, "model": kwargs.get("model", "")},
        )

        self.record_step(run_id, step)

    @staticmethod
    def _extract_raw_http(kwargs: dict) -> Optional[RawHttpRecord]:
        raw = kwargs.get("raw_request_typed_dict")
        if not raw:
            return None
        return RawHttpRecord(
            api_base=raw.get("raw_request_api_base", ""),
            request_body=raw.get("raw_request_body", {}),
            request_headers=raw.get("raw_request_headers", {}),
        )

    @staticmethod
    def _extract_token_usage(response_obj: Any) -> Optional[TokenUsage]:
        usage = getattr(response_obj, "usage", None)
        if not usage:
            return None
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _duration_ms(start: Any, end: Any) -> float:
    """Compute duration in milliseconds between two timestamps."""
    if isinstance(start, datetime) and isinstance(end, datetime):
        return (end - start).total_seconds() * 1000
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return (end - start) * 1000
    return 0.0


def _safe_response_text(response_obj: Any) -> Optional[str]:
    """Best-effort extraction of assistant text from a litellm response."""
    try:
        choices = getattr(response_obj, "choices", None)
        if choices and len(choices) > 0:
            msg = choices[0].message
            return getattr(msg, "content", None)
    except Exception:
        pass
    return None
