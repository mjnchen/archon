"""Observability — per-run trace collection with direct LLM step recording."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from archon.types import ArchonMessage, RawHttpRecord, StepType, TokenUsage, TraceStep

if TYPE_CHECKING:
    from archon.llm import LLMResponse

logger = logging.getLogger(__name__)


class ArchonLogger:
    """Collects per-call trace data for agent runs.

    The agent sets the current run_id via :meth:`set_run_id` before entering
    its ReAct loop and calls :meth:`record_llm_step` after each LLM call.
    Tool invocation steps are recorded directly via :meth:`record_step`.

    Usage::

        observer = ArchonLogger()
        observer.set_run_id("abc123")
        # … agent runs …
        trace = observer.get_trace("abc123")
    """

    def __init__(self) -> None:
        self._traces: Dict[str, List[TraceStep]] = defaultdict(list)
        self._step_counters: Dict[str, int] = defaultdict(int)
        self._current_run_id: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Run-id management
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
    # Step recording
    # ------------------------------------------------------------------

    def record_step(self, run_id: str, step: TraceStep) -> None:
        """Record an arbitrary trace step (tool invoke, tool result, etc.)."""
        with self._lock:
            step.step_index = self._step_counters[run_id]
            self._step_counters[run_id] += 1
            self._traces[run_id].append(step)

    def record_llm_step(
        self,
        run_id: str,
        messages: List[ArchonMessage],
        response: "LLMResponse",
        duration_ms: float,
        model: str,
        success: bool = True,
    ) -> None:
        """Record one LLM call into the run trace.

        Called directly by the agent after each acompletion() call.
        """
        from archon.llm import provider_base_url

        token_usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

        raw_http = RawHttpRecord(
            api_base=provider_base_url(model),
            request_body={
                "model": model,
                "messages": [m.model_dump(exclude_none=True) for m in messages],
            },
            request_headers={},
        )

        step = TraceStep(
            step_type=StepType.LLM_CALL,
            timestamp=datetime.now(timezone.utc),
            duration_ms=duration_ms,
            input=messages,
            output=_safe_response_text(response),
            raw_http=raw_http,
            token_usage=token_usage,
            cost=response.cost,
            metadata={"success": success, "model": model},
        )
        self.record_step(run_id, step)


def _safe_response_text(response: Any) -> Optional[str]:
    try:
        choices = getattr(response, "choices", None)
        if choices:
            return getattr(choices[0].message, "content", None)
    except Exception:
        pass
    return None
