"""Multi-agent orchestration — Pipeline, FanOut, Supervisor, and AgentRegistry."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from archon.agent import Agent
from archon.exceptions import HandoverRequest
from archon.observability import ArchonLogger, AuditTrail
from archon.safety import GuardrailPipeline, HumanApprovalManager
from archon.tools import ToolRegistry
from archon.types import (
    AgentConfig,
    AgentResult,
    OrchestrationResult,
    TenantContext,
    TokenUsage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Named registry of agent configurations and their tool registries.

    Usage::

        registry = AgentRegistry()
        registry.register("researcher", config=researcher_config, tools=research_tools)
        registry.register("writer", config=writer_config, tools=writer_tools)
        agent = registry.build("researcher", observer=obs)
    """

    def __init__(self) -> None:
        self._configs: Dict[str, AgentConfig] = {}
        self._tools: Dict[str, ToolRegistry] = {}

    def register(
        self,
        name: str,
        config: AgentConfig,
        tools: Optional[ToolRegistry] = None,
    ) -> None:
        config = config.model_copy(update={"name": name})
        self._configs[name] = config
        self._tools[name] = tools or ToolRegistry()

    def build(
        self,
        name: str,
        observer: Optional[ArchonLogger] = None,
        guardrails: Optional[GuardrailPipeline] = None,
        hitl: Optional[HumanApprovalManager] = None,
        audit: Optional[AuditTrail] = None,
        tenant: Optional[TenantContext] = None,
    ) -> Agent:
        if name not in self._configs:
            raise KeyError(f"Agent '{name}' is not registered")
        return Agent(
            config=self._configs[name],
            tools=self._tools[name],
            observer=observer,
            guardrails=guardrails,
            hitl=hitl,
            audit=audit,
            tenant=tenant,
        )

    def list_agents(self) -> List[str]:
        return list(self._configs.keys())

    def get_config(self, name: str) -> AgentConfig:
        return self._configs[name]


# ---------------------------------------------------------------------------
# Orchestration patterns
# ---------------------------------------------------------------------------

MergeFn = Callable[[List[AgentResult]], str]


def _default_merge(results: List[AgentResult]) -> str:
    return "\n\n".join(r.output for r in results if r.output)


class Pipeline:
    """Sequential agent pipeline — output of one feeds into the next."""

    def __init__(
        self,
        registry: AgentRegistry,
        agent_names: List[str],
        observer: Optional[ArchonLogger] = None,
        guardrails: Optional[GuardrailPipeline] = None,
        hitl: Optional[HumanApprovalManager] = None,
        audit: Optional[AuditTrail] = None,
        tenant: Optional[TenantContext] = None,
    ) -> None:
        self.registry = registry
        self.agent_names = agent_names
        self.observer = observer
        self.guardrails = guardrails
        self.hitl = hitl
        self.audit = audit
        self.tenant = tenant

    async def arun(self, user_input: str) -> OrchestrationResult:
        agent_results: List[AgentResult] = []
        current_input = user_input
        total_cost = 0.0
        total_tokens = TokenUsage()

        for name in self.agent_names:
            agent = self.registry.build(
                name,
                observer=self.observer,
                guardrails=self.guardrails,
                hitl=self.hitl,
                audit=self.audit,
                tenant=self.tenant,
            )
            result = await agent.arun(current_input)
            agent_results.append(result)
            current_input = result.output
            total_cost += result.total_cost
            total_tokens.prompt_tokens += result.total_tokens.prompt_tokens
            total_tokens.completion_tokens += result.total_tokens.completion_tokens
            total_tokens.total_tokens += result.total_tokens.total_tokens

        return OrchestrationResult(
            final_output=agent_results[-1].output if agent_results else "",
            agent_results=agent_results,
            total_cost=total_cost,
            total_tokens=total_tokens,
        )

    def run(self, user_input: str) -> OrchestrationResult:
        return asyncio.run(self.arun(user_input))


class FanOut:
    """Parallel fan-out — all agents run concurrently on the same input, then merge."""

    def __init__(
        self,
        registry: AgentRegistry,
        agent_names: List[str],
        merge_fn: Optional[MergeFn] = None,
        observer: Optional[ArchonLogger] = None,
        guardrails: Optional[GuardrailPipeline] = None,
        hitl: Optional[HumanApprovalManager] = None,
        audit: Optional[AuditTrail] = None,
        tenant: Optional[TenantContext] = None,
    ) -> None:
        self.registry = registry
        self.agent_names = agent_names
        self.merge_fn = merge_fn or _default_merge
        self.observer = observer
        self.guardrails = guardrails
        self.hitl = hitl
        self.audit = audit
        self.tenant = tenant

    async def arun(self, user_input: str) -> OrchestrationResult:
        agents = [
            self.registry.build(
                name,
                observer=self.observer,
                guardrails=self.guardrails,
                hitl=self.hitl,
                audit=self.audit,
                tenant=self.tenant,
            )
            for name in self.agent_names
        ]
        agent_results = list(await asyncio.gather(*[a.arun(user_input) for a in agents]))
        merged = self.merge_fn(agent_results)
        total_cost = sum(r.total_cost for r in agent_results)
        total_tokens = TokenUsage(
            prompt_tokens=sum(r.total_tokens.prompt_tokens for r in agent_results),
            completion_tokens=sum(r.total_tokens.completion_tokens for r in agent_results),
            total_tokens=sum(r.total_tokens.total_tokens for r in agent_results),
        )
        return OrchestrationResult(
            final_output=merged,
            agent_results=agent_results,
            total_cost=total_cost,
            total_tokens=total_tokens,
        )

    def run(self, user_input: str) -> OrchestrationResult:
        return asyncio.run(self.arun(user_input))


class Supervisor:
    """Supervisor pattern — a coordinator delegates sub-tasks to worker agents."""

    def __init__(
        self,
        registry: AgentRegistry,
        coordinator: str,
        workers: List[str],
        observer: Optional[ArchonLogger] = None,
        guardrails: Optional[GuardrailPipeline] = None,
        hitl: Optional[HumanApprovalManager] = None,
        audit: Optional[AuditTrail] = None,
        tenant: Optional[TenantContext] = None,
        max_delegations: int = 10,
    ) -> None:
        self.registry = registry
        self.coordinator_name = coordinator
        self.workers = workers
        self.observer = observer
        self.guardrails = guardrails
        self.hitl = hitl
        self.audit = audit
        self.tenant = tenant
        self.max_delegations = max_delegations

    async def arun(self, user_input: str) -> OrchestrationResult:
        agent_results: List[AgentResult] = []
        total_cost = 0.0
        total_tokens = TokenUsage()

        coord_tools = ToolRegistry()
        worker_names = self.workers

        @coord_tools.register(name="delegate_to")
        async def delegate_to(agent_name: str, task: str) -> str:
            """Delegate a sub-task to a specialist worker agent."""
            if agent_name not in worker_names:
                return f"Error: unknown worker '{agent_name}'. Available: {worker_names}"
            worker = self.registry.build(
                agent_name,
                observer=self.observer,
                guardrails=self.guardrails,
                hitl=self.hitl,
                audit=self.audit,
                tenant=self.tenant,
            )
            result = await worker.arun(task)
            agent_results.append(result)
            return result.output

        if self.coordinator_name in self.registry._tools:
            coord_tools.merge(self.registry._tools[self.coordinator_name])

        coordinator = Agent(
            config=self.registry.get_config(self.coordinator_name).model_copy(
                update={"max_iterations": self.max_delegations, "tool_names": []}
            ),
            tools=coord_tools,
            observer=self.observer,
            guardrails=self.guardrails,
            hitl=self.hitl,
            audit=self.audit,
            tenant=self.tenant,
        )

        coord_result = await coordinator.arun(user_input)
        agent_results.insert(0, coord_result)

        total_cost = sum(r.total_cost for r in agent_results)
        total_tokens = TokenUsage(
            prompt_tokens=sum(r.total_tokens.prompt_tokens for r in agent_results),
            completion_tokens=sum(r.total_tokens.completion_tokens for r in agent_results),
            total_tokens=sum(r.total_tokens.total_tokens for r in agent_results),
        )
        return OrchestrationResult(
            final_output=coord_result.output,
            agent_results=agent_results,
            total_cost=total_cost,
            total_tokens=total_tokens,
        )

    def run(self, user_input: str) -> OrchestrationResult:
        return asyncio.run(self.arun(user_input))


# ---------------------------------------------------------------------------
# Handover runner utility
# ---------------------------------------------------------------------------

async def run_with_handover(
    registry: AgentRegistry,
    starting_agent: str,
    user_input: str,
    observer: Optional[ArchonLogger] = None,
    guardrails: Optional[GuardrailPipeline] = None,
    hitl: Optional[HumanApprovalManager] = None,
    audit: Optional[AuditTrail] = None,
    tenant: Optional[TenantContext] = None,
    max_handovers: int = 5,
) -> OrchestrationResult:
    """Run agents with automatic handover support."""
    agent_results: List[AgentResult] = []
    current_agent = starting_agent
    current_input = user_input

    for _ in range(max_handovers + 1):
        agent = registry.build(
            current_agent,
            observer=observer,
            guardrails=guardrails,
            hitl=hitl,
            audit=audit,
            tenant=tenant,
        )
        try:
            result = await agent.arun(current_input)
            agent_results.append(result)
            break
        except HandoverRequest as hr:
            partial = AgentResult(
                agent_name=current_agent,
                output=hr.summary,
                stop_reason="handover",
            )
            agent_results.append(partial)
            current_agent = hr.target_agent
            current_input = hr.summary or current_input
            if audit:
                audit.record_handover(partial.run_id, hr.target_agent, tenant)

    total_cost = sum(r.total_cost for r in agent_results)
    total_tokens = TokenUsage(
        prompt_tokens=sum(r.total_tokens.prompt_tokens for r in agent_results),
        completion_tokens=sum(r.total_tokens.completion_tokens for r in agent_results),
        total_tokens=sum(r.total_tokens.total_tokens for r in agent_results),
    )
    return OrchestrationResult(
        final_output=agent_results[-1].output if agent_results else "",
        agent_results=agent_results,
        total_cost=total_cost,
        total_tokens=total_tokens,
    )
