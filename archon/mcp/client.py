"""MCP client — connect to an MCP server and mount its tools.

Usage::

    async with MCPClient(command=["python", "-m", "weather_mcp"]) as mcp:
        registry = ToolRegistry()
        mcp.mount(registry)
        agent = Agent(config=..., tools=registry)
        await agent.arun("what's the weather?")

Lifecycle: the client owns a persistent connection to the MCP server. Use
``async with`` so the connection closes cleanly. The agent runs *inside*
the context manager — outside it, the registered tools' callables become
unusable.

Out of scope (not in v1): MCP resources, MCP prompts, streamable-HTTP and
WebSocket transports, tool-list-changed notifications.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from archon.tools import ToolRegistry


class MCPClient:
    """Connects to an MCP server (stdio or SSE) and mounts its tools.

    Provide either ``command=`` (stdio) or ``url=`` (SSE), not both.
    """

    def __init__(
        self,
        *,
        command: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if (command is None) == (url is None):
            raise ValueError(
                "MCPClient requires exactly one of command= (stdio) or url= (SSE)"
            )
        self.command = command
        self.url = url
        self.env = env
        self._session: Any = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools_cache: List[Any] = []

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        """Open the MCP transport, initialise the session, discover tools."""
        try:
            from mcp import ClientSession
        except ImportError as exc:
            raise ImportError(
                "MCP support requires the 'mcp' package. "
                "Install with: pip install archon[mcp]"
            ) from exc

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        if self.command is not None:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self.command[0],
                args=list(self.command[1:]),
                env=self.env,
            )
            transport = stdio_client(params)
        else:
            from mcp.client.sse import sse_client

            transport = sse_client(self.url)

        read, write = await self._stack.enter_async_context(transport)
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

        result = await self._session.list_tools()
        self._tools_cache = list(result.tools)

    async def disconnect(self) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(None, None, None)
            self._stack = None
            self._session = None
            self._tools_cache = []

    def mount(self, registry: ToolRegistry) -> None:
        """Register this server's tools into *registry*."""
        if self._session is None:
            raise RuntimeError("MCPClient is not connected; call connect() first")
        for tool in self._tools_cache:
            self._mount_one(registry, tool)

    def _mount_one(self, registry: ToolRegistry, tool: Any) -> None:
        session = self._session
        name = tool.name
        description = tool.description or ""
        schema = tool.inputSchema or {"type": "object", "properties": {}}

        async def _call(**kwargs: Any) -> str:
            result = await session.call_tool(name, kwargs)
            chunks: List[str] = []
            for content in (result.content or []):
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
            return "\n".join(chunks)

        registry.register_raw(
            name,
            _call,
            description=description,
            parameters=schema,
        )
