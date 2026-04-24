"""Core Agent — ReAct loop, tool execution, handover, budget/iteration caps."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, List, Optional

from archon import llm as _llm

from archon.exceptions import (
    ApprovalDenied,
    BudgetExceeded,
    HandoverRequest,
    MaxIterationsExceeded,
    ToolExecutionError,
)
from archon.observability import ArchonLogger
from archon.state import AgentState
from archon.tools import ToolRegistry
from archon.types import (
    AgentConfig,
    AgentResult,
    StepType,
    TenantContext,
    TokenUsage,
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
    ) -> None:
        self.config = config
        self.tools = tools or ToolRegistry()
        self.observer = observer
        self.guardrails = guardrails
        self.hitl = hitl
        self.audit = audit
        self.tenant = tenant

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

    def run(self, user_input: str, state: Optional[AgentState] = None) -> AgentResult:
        """Synchronous wrapper around :meth:`arun`."""
        return asyncio.run(self.arun(user_input, state=state))

    async def arun(
        self,
        user_input: str,
        state: Optional[AgentState] = None,
    ) -> AgentResult:
        """Execute the ReAct loop until completion or a stop condition is hit."""
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

        total_cost = state.total_cost
        total_tokens = TokenUsage()

        try:
            for iteration in range(self.config.max_iterations):
                state.iteration = iteration + 1

                # --- Input guardrails (first iteration only) ---
                if iteration == 0 and self.guardrails:
                    await self.guardrails.check_input(user_input, self.tenant)

                # --- LLM call ---
                openai_tools = self.tools.to_openai_tools(self.config.tool_names or None)

                t0 = time.monotonic()
                response = await _llm.acompletion(
                    model=self.config.model,
                    messages=state.messages,
                    tools=openai_tools or None,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                t1 = time.monotonic()

                if self.observer:
                    self.observer.record_llm_step(
                        run_id=run_id,
                        messages=state.messages,
                        response=response,
                        duration_ms=(t1 - t0) * 1000,
                        model=self.config.model,
                    )

                # Accumulate cost
                total_cost += response.cost
                total_tokens.prompt_tokens += response.usage.prompt_tokens
                total_tokens.completion_tokens += response.usage.completion_tokens
                total_tokens.total_tokens += response.usage.total_tokens

                # Budget check
                if self.config.max_cost and total_cost > self.config.max_cost:
                    raise BudgetExceeded(total_cost, self.config.max_cost)

                assistant_msg = response.choices[0].message
                state.add_assistant(assistant_msg)

                # --- Output guardrails ---
                if self.guardrails and assistant_msg.content:
                    await self.guardrails.check_output(assistant_msg.content, self.tenant)

                # --- Check for tool calls ---
                if not assistant_msg.tool_calls:
                    # No tool calls → agent is done
                    return self._build_result(
                        state, run_id, total_cost, total_tokens, "completed"
                    )

                # --- Execute each tool call ---
                for tc in assistant_msg.tool_calls:
                    fn_name = tc.name
                    fn_args = tc.arguments

                    # Tool-call guardrails
                    if self.guardrails:
                        await self.guardrails.check_tool_call(fn_name, fn_args, self.tenant)

                    # HITL approval
                    if self.hitl:
                        await self.hitl.check(fn_name, fn_args, self.tenant)

                    # Record tool invocation in observer
                    if self.observer:
                        self.observer.record_step(
                            run_id,
                            TraceStep(
                                step_type=StepType.TOOL_INVOKE,
                                input={"name": fn_name, "arguments": fn_args},
                            ),
                        )
                    if self.audit:
                        self.audit.record_tool_invoke(run_id, fn_name, fn_args, self.tenant)

                    # Execute
                    tool_t0 = time.monotonic()
                    try:
                        result = await self.tools.execute(fn_name, fn_args)
                        result_str = result if isinstance(result, str) else json.dumps(result)
                    except HandoverRequest:
                        raise
                    except ToolExecutionError as exc:
                        result_str = f"Error: {exc}"
                    tool_t1 = time.monotonic()

                    state.add_tool_result(tc.id, result_str)

                    if self.observer:
                        self.observer.record_step(
                            run_id,
                            TraceStep(
                                step_type=StepType.TOOL_RESULT,
                                duration_ms=(tool_t1 - tool_t0) * 1000,
                                input={"tool_call_id": tc.id},
                                output=result_str,
                            ),
                        )
                    if self.audit:
                        self.audit.record_tool_result(run_id, fn_name, result_str, self.tenant)

            # Exhausted iterations
            raise MaxIterationsExceeded(self.config.max_iterations, self.config.max_iterations)

        except (MaxIterationsExceeded, BudgetExceeded) as exc:
            stop_reason = "max_iterations" if isinstance(exc, MaxIterationsExceeded) else "budget_exceeded"
            if self.audit:
                self.audit.record_run_completed(run_id, stop_reason, self.tenant)
            return self._build_result(state, run_id, total_cost, total_tokens, stop_reason)

        except HandoverRequest:
            if self.audit:
                self.audit.record_run_completed(run_id, "handover", self.tenant)
            raise

        except Exception:
            if self.audit:
                self.audit.record_run_failed(run_id, self.tenant)
            raise

        finally:
            state.total_cost = total_cost
            if self.observer:
                self.observer.clear_run_id()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        state: AgentState,
        run_id: str,
        total_cost: float,
        total_tokens: TokenUsage,
        stop_reason: str,
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
        )
