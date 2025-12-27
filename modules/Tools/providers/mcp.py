"""Model Context Protocol (MCP) tool provider implementation."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

from .base import ToolProvider, ToolProviderSpec
from .registry import tool_provider_registry


@dataclass
class _McpClientComponents:
    session_cls: Any
    stdio_transport_cls: Optional[type]
    websocket_transport_cls: Optional[type]


@dataclass
class _McpSessionHandle:
    name: str
    session: Any
    transport: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class McpToolProvider(ToolProvider):
    """Provider that forwards tool invocations to MCP servers."""

    def __init__(self, spec: ToolProviderSpec, *, tool_name: str, fallback_callable=None) -> None:
        super().__init__(spec, tool_name=tool_name, fallback_callable=fallback_callable)

        raw_servers = self.config.get("servers")
        self._servers: Mapping[str, Mapping[str, Any]] = self._normalize_servers(raw_servers)
        if not self._servers:
            raise ValueError("MCP provider requires at least one server entry under 'servers'")

        self._default_server = str(self.config.get("server") or "").strip()
        if not self._default_server and self._servers:
            self._default_server = next(iter(self._servers))

        self._default_tool = str(self.config.get("tool") or "").strip()

        timeout_value = self.config.get("timeout_seconds", self.config.get("timeout", 30.0))
        try:
            self._timeout_seconds = float(timeout_value)
        except (TypeError, ValueError):
            self._timeout_seconds = 30.0
        if self._timeout_seconds <= 0:
            self._timeout_seconds = 0.0

        self._components: Optional[_McpClientComponents] = None
        self._sessions: MutableMapping[str, _McpSessionHandle] = {}
        self._sessions_lock = asyncio.Lock()

    async def call(self, **kwargs: Any) -> Any:
        server_name = self._resolve_server(kwargs)
        tool_name = self._resolve_tool(kwargs)
        arguments = self._extract_arguments(kwargs)

        session_handle = await self._get_or_create_session(server_name)

        async def _execute() -> Any:
            async with session_handle.lock:
                return await self._invoke_tool(session_handle.session, tool_name, arguments)

        try:
            if self._timeout_seconds > 0:
                return await asyncio.wait_for(_execute(), timeout=self._timeout_seconds)
            return await _execute()
        except asyncio.TimeoutError as exc:
            self.logger.warning(
                "MCP tool '%s' on server '%s' timed out after %.2fs",
                tool_name,
                server_name,
                self._timeout_seconds,
            )
            raise RuntimeError(f"MCP call to '{tool_name}' on '{server_name}' timed out") from exc
        except Exception as exc:
            self.logger.warning(
                "MCP call for tool '%s' on server '%s' failed: %s",
                tool_name,
                server_name,
                exc,
                exc_info=True,
            )
            raise

    async def health_check(self) -> bool:
        server_name = self._default_server or (next(iter(self._servers)) if self._servers else "")
        if not server_name:
            return False

        try:
            session_handle = await self._get_or_create_session(server_name)
            async with session_handle.lock:
                session = session_handle.session
                if hasattr(session, "ping"):
                    result = session.ping()  # type: ignore[attr-defined]
                    if inspect.isawaitable(result):
                        await result
                    return True

                list_tools = getattr(session, "list_tools", None)
                if list_tools is None:
                    return False
                result = list_tools()
                if inspect.isawaitable(result):
                    result = await result
                return bool(result)
        except Exception as exc:  # pragma: no cover - defensive logging for health checks
            self.logger.debug(
                "MCP health check for server '%s' failed: %s",
                server_name,
                exc,
                exc_info=True,
            )
            return False

    def _resolve_server(self, params: Mapping[str, Any]) -> str:
        candidate = params.get("server") or params.get("server_name") or self._default_server
        server_name = str(candidate or "").strip()
        if not server_name:
            raise ValueError("MCP provider requires a configured 'server'")
        if self._servers and server_name not in self._servers:
            raise ValueError(f"Unknown MCP server '{server_name}'")
        return server_name

    def _resolve_tool(self, params: Mapping[str, Any]) -> str:
        candidate = params.get("tool") or params.get("tool_name") or self._default_tool or self.tool_name
        tool_name = str(candidate or "").strip()
        if not tool_name:
            raise ValueError("MCP provider requires a target tool name")
        return tool_name

    def _extract_arguments(self, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if "payload" in params and params["payload"] is not None:
            arguments = params["payload"]
        elif "arguments" in params and params["arguments"] is not None:
            arguments = params["arguments"]
        else:
            arguments = {k: v for k, v in params.items() if k not in {"server", "server_name", "tool", "tool_name"}}

        if not isinstance(arguments, Mapping):
            raise ValueError("Invocation payload must be a mapping")
        return arguments

    async def _get_or_create_session(self, server_name: str) -> _McpSessionHandle:
        async with self._sessions_lock:
            existing = self._sessions.get(server_name)
            if existing is not None:
                return existing

        session_handle = await self._create_session(server_name)

        async with self._sessions_lock:
            existing = self._sessions.get(server_name)
            if existing is not None:
                return existing
            self._sessions[server_name] = session_handle
            return session_handle

    async def _create_session(self, server_name: str) -> _McpSessionHandle:
        config = self._servers.get(server_name, {}) if self._servers else {}
        components = self._get_components()

        transport_type = str(config.get("transport", config.get("type", "stdio"))).lower()
        transport = None
        if transport_type == "stdio":
            command = config.get("command") or config.get("cmd")
            args = config.get("args") or config.get("command_args") or []
            env = config.get("env")
            cwd = config.get("cwd")
            if components.stdio_transport_cls is None:
                raise RuntimeError("MCP stdio transport is unavailable; ensure the 'mcp' package is installed")
            transport = components.stdio_transport_cls(command, args=args, env=env, cwd=cwd)
        elif transport_type in {"ws", "wss", "websocket", "http", "https"}:
            url = config.get("url") or config.get("uri") or config.get("endpoint")
            headers = config.get("headers")
            if components.websocket_transport_cls is None:
                raise RuntimeError("MCP websocket transport is unavailable; ensure the 'mcp' package is installed")
            transport = components.websocket_transport_cls(url, headers=headers)
        else:
            raise ValueError(f"Unsupported MCP transport '{transport_type}'")

        if transport is None:
            raise RuntimeError(f"Failed to build MCP transport for server '{server_name}'")

        self.logger.info("Creating MCP session for server '%s' using transport '%s'", server_name, transport_type)

        session = components.session_cls(transport)
        connect = getattr(session, "connect", None)
        if connect is not None:
            result = connect()
            if inspect.isawaitable(result):
                await result
        return _McpSessionHandle(name=server_name, session=session, transport=transport)

    async def _invoke_tool(self, session: Any, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        candidates = [
            getattr(session, "call_tool", None),
            getattr(session, "invoke_tool", None),
            getattr(session, "call", None),
        ]

        last_error: Optional[Exception] = None
        for callable_obj in candidates:
            if callable_obj is None:
                continue
            try:
                result = callable_obj(tool_name, arguments=arguments)
            except TypeError:
                try:
                    result = callable_obj(tool_name, arguments)
                except Exception as exc:  # pragma: no cover - defensive retry path
                    last_error = exc
                    continue
            except Exception as exc:
                last_error = exc
                continue

            if inspect.isawaitable(result):
                result = await result
            return result

        if last_error is not None:
            raise last_error
        raise RuntimeError("MCP session does not expose a callable tool invocation method")

    def _get_components(self) -> _McpClientComponents:
        if self._components is not None:
            return self._components

        existing_module = sys.modules.get("mcp")
        if existing_module is None and importlib.util.find_spec("mcp") is None:
            raise RuntimeError("The 'mcp' package is required to use the MCP tool provider")

        mcp_module = existing_module or importlib.import_module("mcp")
        transport_module = sys.modules.get("mcp.transport") or importlib.import_module("mcp.transport")

        session_cls = getattr(mcp_module, "ClientSession", None)
        if session_cls is None and hasattr(mcp_module, "client"):
            session_cls = getattr(mcp_module.client, "ClientSession", None)
        if session_cls is None:
            raise RuntimeError("Unable to locate MCP client session class; upgrade the 'mcp' package")

        stdio_cls = getattr(transport_module, "StdioTransport", None)
        if stdio_cls is None:
            stdio_cls = getattr(transport_module, "StdioClientTransport", None)
        websocket_cls = getattr(transport_module, "WebSocketTransport", None)
        if websocket_cls is None:
            websocket_cls = getattr(transport_module, "WebsocketTransport", None)

        self._components = _McpClientComponents(
            session_cls=session_cls,
            stdio_transport_cls=stdio_cls,
            websocket_transport_cls=websocket_cls,
        )
        return self._components

    @staticmethod
    def _normalize_servers(raw: Any) -> Mapping[str, Mapping[str, Any]]:
        servers: dict[str, Mapping[str, Any]] = {}
        if isinstance(raw, Mapping):
            for name, config in raw.items():
                if not isinstance(config, Mapping):
                    continue
                server_name = str(name or "").strip()
                if not server_name:
                    continue
                servers[server_name] = dict(config)
        return servers


tool_provider_registry.register("mcp", McpToolProvider)

__all__ = ["McpToolProvider"]
