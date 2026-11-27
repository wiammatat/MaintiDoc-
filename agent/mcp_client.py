"""Client MCP de l'agent : appelle les outils du serveur de maintenance (mcp_server/server.py).

- StdioMCPClient : lance le serveur dans un processus séparé (transport stdio), comme en production.
- InMemoryMCPClient : même protocole MCP, serveur dans le même processus (tests).
Les deux renvoient le résultat JSON de l'outil, ou lèvent MCPError (refus d'accès, approbation invalide...).
"""

import asyncio
import json
import os
import sys
from typing import Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from common.config import ROOT


class MCPError(RuntimeError):
    pass


class MCPClient(Protocol):
    def call(self, tool: str, arguments: dict) -> dict: ...


def _parse(result) -> dict:
    text = "".join(getattr(c, "text", "") for c in result.content)
    if result.isError:
        raise MCPError(text or "erreur de l'outil MCP")
    if result.structuredContent:
        return dict(result.structuredContent.get("result", result.structuredContent))
    return json.loads(text)


class StdioMCPClient:
    """Une session MCP par appel : simple et robuste (au plus 3 appels par demande)."""

    def __init__(self, command: str = sys.executable, args: tuple[str, ...] = ("-m", "mcp_server.server")):
        self.params = StdioServerParameters(command=command, args=list(args), cwd=str(ROOT), env=dict(os.environ))

    async def _call(self, tool: str, arguments: dict):
        async with stdio_client(self.params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, arguments)

    def call(self, tool: str, arguments: dict) -> dict:
        # Erreur analysée hors de la session : sinon elle remonte enveloppée dans un ExceptionGroup.
        return _parse(asyncio.run(self._call(tool, arguments)))


class InMemoryMCPClient:
    def __init__(self):
        from mcp_server.server import mcp
        self.server = mcp
        self.calls: list[str] = []

    async def _call(self, tool: str, arguments: dict):
        from mcp.shared.memory import create_connected_server_and_client_session
        async with create_connected_server_and_client_session(self.server._mcp_server) as session:
            return await session.call_tool(tool, arguments)

    def call(self, tool: str, arguments: dict) -> dict:
        self.calls.append(tool)
        return _parse(asyncio.run(self._call(tool, arguments)))
