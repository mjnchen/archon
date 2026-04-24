"""Configuration loader — read agent definitions, guardrails, and HITL policies from YAML/JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from archon.safety import (
    ContentPolicyGuardrail,
    DangerousToolCallGuardrail,
    GuardrailPipeline,
    PIIDetector,
    HumanApprovalManager,
)
from archon.safety.hitl import AutoApproveHandler
from archon.types import AgentConfig, ApprovalPolicy

logger = logging.getLogger(__name__)


def _load_file(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON or YAML file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            raise ImportError("PyYAML is required to load .yaml config files: pip install pyyaml")
    return json.loads(text)


def load_agent_configs(path: Union[str, Path]) -> Dict[str, AgentConfig]:
    """Load agent configurations from a config file.

    Expected structure::

        {
          "agents": {
            "researcher": {
              "model": "gpt-4o",
              "system_prompt": "You are a researcher.",
              "max_iterations": 15,
              "tool_names": ["web_search"]
            },
            ...
          }
        }
    """
    data = _load_file(path)
    agents_data = data.get("agents", {})
    configs: Dict[str, AgentConfig] = {}
    for name, agent_dict in agents_data.items():
        configs[name] = AgentConfig(name=name, **agent_dict)
    return configs


def load_guardrail_pipeline(path: Union[str, Path]) -> GuardrailPipeline:
    """Load guardrail configuration.

    Expected structure::

        {
          "guardrails": {
            "input": ["pii_detector"],
            "output": ["content_policy"],
            "tool_call": ["dangerous_tool_call"],
            "content_policy_keywords": ["password", "secret"]
          }
        }
    """
    data = _load_file(path)
    g_data = data.get("guardrails", {})
    keywords = g_data.get("content_policy_keywords", [])

    builtin_map = {
        "pii_detector": PIIDetector,
        "content_policy": lambda: ContentPolicyGuardrail(blocked_keywords=keywords),
    }
    tool_builtin_map = {
        "dangerous_tool_call": DangerousToolCallGuardrail,
    }

    input_gs = [
        builtin_map[n]() if callable(builtin_map.get(n)) else builtin_map[n]()
        for n in g_data.get("input", [])
        if n in builtin_map
    ]
    output_gs = [
        builtin_map[n]() if callable(builtin_map.get(n)) else builtin_map[n]()
        for n in g_data.get("output", [])
        if n in builtin_map
    ]
    tool_gs = [
        tool_builtin_map[n]() for n in g_data.get("tool_call", []) if n in tool_builtin_map
    ]

    return GuardrailPipeline(
        input_guardrails=input_gs,
        output_guardrails=output_gs,
        tool_call_guardrails=tool_gs,
    )


def load_hitl_policies(path: Union[str, Path]) -> HumanApprovalManager:
    """Load HITL policies from config.

    Expected structure::

        {
          "hitl": {
            "policies": [
              {"tool_name_patterns": ["send_*", "delete_*"], "timeout": 120}
            ]
          }
        }
    """
    data = _load_file(path)
    hitl_data = data.get("hitl", {})
    policies = [ApprovalPolicy(**p) for p in hitl_data.get("policies", [])]
    return HumanApprovalManager(policies=policies)
