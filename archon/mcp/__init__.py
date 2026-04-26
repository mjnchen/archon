"""archon.mcp — Model Context Protocol client.

Mount the tools exposed by an MCP server into an Archon ``ToolRegistry``.
Requires the optional ``mcp`` extra: ``pip install archon[mcp]``.
"""

from archon.mcp.client import MCPClient

__all__ = ["MCPClient"]
