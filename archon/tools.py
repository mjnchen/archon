"""Tool registry — register Python callables, auto-generate JSON Schema, execute with timeout."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from archon.exceptions import ToolExecutionError, ToolNotFoundError
from archon.types import ToolDef

logger = logging.getLogger(__name__)

# Python type → JSON Schema type
_TYPE_MAP: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp: type) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = getattr(tp, "__origin__", None)

    if origin is list:
        args = getattr(tp, "__args__", ())
        items = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": items}

    if origin is dict:
        return {"type": "object"}

    if tp in _TYPE_MAP:
        return {"type": _TYPE_MAP[tp]}

    return {"type": "string"}


def _schema_from_callable(fn: Callable) -> Dict[str, Any]:
    """Build a JSON Schema ``parameters`` object from *fn*'s type hints and docstring."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prop = _python_type_to_json_schema(hints.get(name, str))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        properties[name] = prop

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _description_from_callable(fn: Callable) -> str:
    """Extract the first paragraph of *fn*'s docstring as a description."""
    doc = inspect.getdoc(fn)
    if not doc:
        return ""
    return doc.split("\n\n")[0].strip()


class ToolRegistry:
    """Registry of tools that agents can invoke.

    Usage::

        registry = ToolRegistry()

        @registry.register
        def get_weather(location: str, unit: str = "celsius") -> str:
            \"\"\"Get the current weather for a location.\"\"\"
            return f"72°F in {location}"

        # Or register imperatively:
        registry.register(some_function, name="my_tool", requires_approval=True)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}
        self._callables: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        fn: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        timeout: Optional[float] = 30.0,
        requires_approval: bool = False,
    ) -> Callable:
        """Register a callable as a tool. Works as a decorator or a direct call."""

        def _register(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or _description_from_callable(func)
            params = _schema_from_callable(func)

            self._tools[tool_name] = ToolDef(
                name=tool_name,
                description=tool_desc,
                parameters=params,
                timeout=timeout,
                requires_approval=requires_approval,
            )
            self._callables[tool_name] = func
            return func

        if fn is not None:
            return _register(fn)
        return _register

    def get(self, name: str) -> ToolDef:
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(self) -> List[ToolDef]:
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute a registered tool by name, with an optional timeout."""
        if name not in self._callables:
            raise ToolNotFoundError(name)

        tool_def = self._tools[name]
        fn = self._callables[name]
        effective_timeout = timeout or tool_def.timeout

        try:
            if inspect.iscoroutinefunction(fn):
                coro = fn(**arguments)
            else:
                loop = asyncio.get_running_loop()
                coro = loop.run_in_executor(None, lambda: fn(**arguments))

            if effective_timeout:
                result = await asyncio.wait_for(coro, timeout=effective_timeout)
            else:
                result = await coro

            return result
        except asyncio.TimeoutError as exc:
            raise ToolExecutionError(name, exc) from exc
        except (ToolNotFoundError, ToolExecutionError):
            raise
        except Exception as exc:
            from archon.exceptions import HandoverRequest
            if isinstance(exc, HandoverRequest):
                raise
            raise ToolExecutionError(name, exc) from exc

    # ------------------------------------------------------------------
    # OpenAI format
    # ------------------------------------------------------------------

    def to_openai_tools(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return tool definitions in OpenAI function-calling format.

        If *names* is provided, only include those tools.
        """
        selected = self._tools.values() if names is None else [
            self._tools[n] for n in names if n in self._tools
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in selected
        ]

    def merge(self, other: ToolRegistry) -> None:
        """Merge another registry's tools into this one."""
        self._tools.update(other._tools)
        self._callables.update(other._callables)

    def register_raw(
        self,
        name: str,
        fn: Callable,
        *,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = 30.0,
        requires_approval: bool = False,
    ) -> None:
        """Register a tool with a pre-built JSON Schema, skipping introspection.

        Used by integrations that already know their tools' shapes (e.g. MCP).
        """
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            timeout=timeout,
            requires_approval=requires_approval,
        )
        self._callables[name] = fn
