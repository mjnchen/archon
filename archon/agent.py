"""Core Agent — ReAct loop, tool execution, handover, budget/iteration caps."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, AsyncIterator, List, Optional

from archon import llm as _llm

from archon.exceptions import (
    ApprovalDenied,
    BudgetExceeded,
    HandoverRequest,
    MaxIterationsExceeded,
    ToolExecutionError,
)
from archon.hooks import AgentHooks
from archon.llm._base import LLMChoice, LLMResponse, LLMUsage
from archon.observability import ArchonLogger
from archon.session import Session
from archon.state import AgentState
from archon.tools import ToolRegistry
from archon.types import (
    AgentConfig,
    AgentEvent,
    AgentResult,
    ArchonMessage,
    ArchonToolCall,
    CompleteEvent,
    IterationEvent,
    StepType,
    TenantContext,
    TextDeltaEvent,
    TokenUsage,
    ToolCallEvent,
    ToolEndEvent,
    ToolStartEvent,
    TraceStep,
)

if TYPE_CHECKING:
    from archon.observability import AuditTrail
    from archon.safety import GuardrailPipeline, HumanApprovalManager

logger = logging.getLogger(__name__)


class Agent:
    """A single LLM agent with a ReAct loop.

    Usage::

        agent = Agent(
            config=AgentConfig(model="gpt-4o-mini", system_prompt="You are helpful."),
            tools=my_registry,
            observer=observer,
        )
        result = await agent.arun("What is the weather in Boston?")

    For token-by-token streaming, use :meth:`astream`::

        async for event in agent.astream("..."):
            if isinstance(event, TextDeltaEvent):
                print(event.text, end="", flush=True)
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[ToolRegistry] = None,
        observer: Optional[ArchonLogger] = None,
        guardrails: Optional[GuardrailPipeline] = None,
        hitl: Optional[HumanApprovalManager] = None,
        audit: Optional[AuditTrail] = None,
        tenant: Optional[TenantContext] = None,
        hooks: Optional[AgentHooks] = None,
    ) -> None:
        self.config = config
        self.tools = tools or ToolRegistry()
        self.observer = observer
        self.guardrails = guardrails
        self.hitl = hitl
        self.audit = audit
        self.tenant = tenant
        self.hooks = hooks or AgentHooks()

        self._register_handover_tool()

    # ------------------------------------------------------------------
    # Built-in handover tool
    # ------------------------------------------------------------------

    def _register_handover_tool(self) -> None:
        """Register the ``handover_to_agent`` tool so the LLM can request delegation."""
        if self.tools.has("handover_to_agent"):
            return

        async def handover_to_agent(target_agent: str, summary: str = "") -> str:
            """Hand over the conversation to another agent."""
            raise HandoverRequest(
                target_agent=target_agent,
                summary=summary,
            )

        self.tools.register(
            handover_to_agent,
            name="handover_to_agent",
            description="Delegate the conversation to a different specialist agent.",
            timeout=5.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        state: Optional[AgentState] = None,
        session: Optional[Session] = None,
    ) -> AgentResult:
        """Synchronous wrapper around :meth:`arun`."""
        return asyncio.run(self.arun(user_input, state=state, session=session))

    async def arun(
        self,
        user_input: str,
        state: Optional[AgentState] = None,
        session: Optional[Session] = None,
    ) -> AgentResult:
        """Drain :meth:`astream` and return the final :class:`AgentResult`."""
        result: Optional[AgentResult] = None
        async for event in self.astream(user_input, state=state, session=session):
            if isinstance(event, CompleteEvent):
                result = event.result
        if result is None:
            raise RuntimeError("astream completed without CompleteEvent")
        return result

    async def astream(
        self,
        user_input: str,
        state: Optional[AgentState] = None,
        session: Optional[Session] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the ReAct loop, yielding events as they happen.

        Emits IterationEvent, TextDeltaEvent, ToolCallEvent, ToolStartEvent,
        ToolEndEvent, and a terminal CompleteEvent carrying the AgentResult.

        If a *session* is provided, its accumulated state is used as the
        starting point and updated in place after the run completes.
        """
        if state is None and session is not None and session.state is not None:
            # Preserve conversation history but start a fresh run.
            prev = session.state
            state = AgentState(
                agent_name=self.config.name,
                config=self.config,
                messages=list(prev.messages),
                total_cost=prev.total_cost,
            )
        state = state or AgentState(
            agent_name=self.config.name,
            config=self.config,
        )
        run_id = state.run_id

        if self.observer:
            self.observer.set_run_id(run_id)
        if self.audit:
            self.audit.record_run_started(run_id, self.config.name, self.tenant)

        # Seed messages
        if self.config.system_prompt:
            state.add_system(self.config.system_prompt)
        state.add_user(user_input)

        await self.hooks.on_agent_start(self, state)

        total_cost = state.total_cost
        total_tokens = TokenUsage()

        try:
            for iteration in range(self.config.max_iterations):
                state.iteration = iteration + 1
                yield IterationEvent(n=iteration + 1)

                # --- Input guardrails (first iteration only) ---
                if iteration == 0 and self.guardrails:
                    await self.guardrails.check_input(user_input, self.tenant)

                # --- LLM call (streaming) ---
                openai_tools = self.tools.to_openai_tools(self.config.tool_names or None)
                snapshot = list(state.messages)

                accumulated_text: List[str] = []
                accumulated_tool_calls: List[ArchonToolCall] = []
                step_usage: Optional[LLMUsage] = None
                step_cost: float = 0.0

                await self.hooks.on_llm_start(self, snapshot)
                t0 = time.monotonic()
                async for ev in _llm.astream(
                    model=self.config.model,
                    messages=snapshot,
                    tools=openai_tools or None,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    output_schema=self.config.output_schema,
                ):
                    if ev.kind == "text_delta" and ev.text:
                        accumulated_text.append(ev.text)
                        yield TextDeltaEvent(text=ev.text)
                    elif ev.kind == "tool_call_complete" and ev.tool_call:
                        accumulated_tool_calls.append(ev.tool_call)
                        yield ToolCallEvent(tool_call=ev.tool_call)
                    elif ev.kind == "done":
                        step_usage = ev.usage
                        step_cost = ev.cost
                t1 = time.monotonic()

                content = "".join(accumulated_text) or None
                assistant_msg = ArchonMessage(
                    role="assistant",
                    content=content,
                    tool_calls=accumulated_tool_calls or None,
                )
                state.add_assistant(assistant_msg)

                # Observer: synthesise an LLMResponse for trace compat.
                synth = LLMResponse(
                    choices=[LLMChoice(message=assistant_msg)],
                    usage=step_usage or LLMUsage(),
                    cost=step_cost,
                )
                if self.observer:
                    self.observer.record_llm_step(
                        run_id=run_id,
                        messages=snapshot,
                        response=synth,
                        duration_ms=(t1 - t0) * 1000,
                        model=self.config.model,
                    )
                await self.hooks.on_llm_end(self, synth)

                total_cost += step_cost
                if step_usage:
                    total_tokens.prompt_tokens += step_usage.prompt_tokens
                    total_tokens.completion_tokens += step_usage.completion_tokens
                    total_tokens.total_tokens += step_usage.total_tokens

                if self.config.max_cost and total_cost > self.config.max_cost:
                    raise BudgetExceeded(total_cost, self.config.max_cost)

                if self.guardrails and content:
                    await self.guardrails.check_output(content, self.tenant)

                # --- Structured output: synthetic-tool path (Anthropic). ---
                final_output: Optional[dict] = None
                if self.config.output_schema and assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        if tc.name == _llm.FINAL_OUTPUT_TOOL_NAME:
                            final_output = tc.arguments
                            break
                    if final_output is not None:
                        ev = self._complete_event(
                            state, run_id, total_cost, total_tokens,
                            "completed", final_output=final_output,
                        )
                        await self.hooks.on_agent_end(self, ev.result)
                        yield ev
                        return

                # --- No tool calls → run complete. ---
                if not assistant_msg.tool_calls:
                    if self.config.output_schema and content:
                        try:
                            final_output = json.loads(content)
                        except json.JSONDecodeError:
                            final_output = None
                    ev = self._complete_event(
                        state, run_id, total_cost, total_tokens,
                        "completed", final_output=final_output,
                    )
                    await self.hooks.on_agent_end(self, ev.result)
                    yield ev
                    return

                # --- Pre-execution checks run serially. ---
                for tc in assistant_msg.tool_calls:
                    if self.guardrails:
                        await self.guardrails.check_tool_call(tc.name, tc.arguments, self.tenant)
                    if self.hitl:
                        await self.hitl.check(tc.name, tc.arguments, self.tenant)
                    if self.observer:
                        self.observer.record_step(
                            run_id,
                            TraceStep(
                                step_type=StepType.TOOL_INVOKE,
                                input={"name": tc.name, "arguments": tc.arguments},
                            ),
                        )
                    if self.audit:
                        self.audit.record_tool_invoke(run_id, tc.name, tc.arguments, self.tenant)
                    await self.hooks.on_tool_start(self, tc)
                    yield ToolStartEvent(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                    )

                # --- Execute tool calls concurrently. ---
                tool_results = await asyncio.gather(
                    *[self._execute_one_tool(tc) for tc in assistant_msg.tool_calls]
                )

                # --- Commit results in tool_calls order (wire format demands it). ---
                first_handover: Optional[HandoverRequest] = None
                for tc, result_str, handover, duration_ms in tool_results:
                    state.add_tool_result(tc.id, result_str)
                    if self.observer:
                        self.observer.record_step(
                            run_id,
                            TraceStep(
                                step_type=StepType.TOOL_RESULT,
                                duration_ms=duration_ms,
                                input={"tool_call_id": tc.id},
                                output=result_str,
                            ),
                        )
                    if self.audit:
                        self.audit.record_tool_result(run_id, tc.name, result_str, self.tenant)
                    await self.hooks.on_tool_end(self, tc, result_str)
                    yield ToolEndEvent(
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        output=result_str,
                        duration_ms=duration_ms,
                    )
                    if handover and first_handover is None:
                        first_handover = handover

                if first_handover is not None:
                    raise first_handover

            # Exhausted iterations
            raise MaxIterationsExceeded(
                self.config.max_iterations, self.config.max_iterations
            )

        except (MaxIterationsExceeded, BudgetExceeded) as exc:
            stop_reason = (
                "max_iterations"
                if isinstance(exc, MaxIterationsExceeded)
                else "budget_exceeded"
            )
            if self.audit:
                self.audit.record_run_completed(run_id, stop_reason, self.tenant)
            ev = self._complete_event(
                state, run_id, total_cost, total_tokens, stop_reason
            )
            await self.hooks.on_agent_end(self, ev.result)
            yield ev

        except HandoverRequest as exc:
            if self.audit:
                self.audit.record_run_completed(run_id, "handover", self.tenant)
            await self.hooks.on_handoff(self, exc)
            raise

        except Exception:
            if self.audit:
                self.audit.record_run_failed(run_id, self.tenant)
            raise

        finally:
            state.total_cost = total_cost
            if session is not None:
                session.state = state
            if self.observer:
                self.observer.clear_run_id()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_one_tool(self, tc):
        """Run a single tool call. Returns (tc, result_str, handover_or_none, duration_ms).

        HandoverRequest is captured and returned rather than raised, so sibling
        tool calls in the same turn still complete. The caller re-raises the
        first handover after all results are committed to state.
        """
        t0 = time.monotonic()
        handover: Optional[HandoverRequest] = None
        try:
            result = await self.tools.execute(tc.name, tc.arguments)
            result_str = result if isinstance(result, str) else json.dumps(result)
        except HandoverRequest as exc:
            handover = exc
            result_str = f"[handover requested to {exc.target_agent}]"
        except ToolExecutionError as exc:
            result_str = f"Error: {exc}"
        return tc, result_str, handover, (time.monotonic() - t0) * 1000

    def _complete_event(
        self,
        state: AgentState,
        run_id: str,
        total_cost: float,
        total_tokens: TokenUsage,
        stop_reason: str,
        final_output: Optional[dict] = None,
    ) -> CompleteEvent:
        return CompleteEvent(
            result=self._build_result(
                state, run_id, total_cost, total_tokens, stop_reason, final_output
            )
        )

    def _build_result(
        self,
        state: AgentState,
        run_id: str,
        total_cost: float,
        total_tokens: TokenUsage,
        stop_reason: str,
        final_output: Optional[dict] = None,
    ) -> AgentResult:
        last_content = ""
        for msg in reversed(state.messages):
            if msg.role == "assistant" and msg.content:
                last_content = msg.content
                break

        trace = self.observer.get_trace(run_id) if self.observer else []

        if self.audit:
            self.audit.record_run_completed(run_id, stop_reason, self.tenant)

        return AgentResult(
            run_id=run_id,
            agent_name=self.config.name,
            output=last_content,
            messages=state.messages,
            trace=trace,
            total_cost=total_cost,
            total_tokens=total_tokens,
            iterations=state.iteration,
            stop_reason=stop_reason,
            final_output=final_output,
        )
