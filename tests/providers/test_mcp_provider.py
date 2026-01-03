import asyncio
from types import MappingProxyType

import pytest

from modules.Tools.providers.base import ToolProviderSpec
from modules.Tools.providers.mcp import McpToolProvider, _McpClientComponents


class _StubTransport:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _StubSession:
    def __init__(self, transport):
        self.transport = transport
        self.calls = []
        self.raise_on_call = None
        self.tool_list = [
            {"name": "echo", "description": "Echo tool", "input_schema": {"type": "object"}}
        ]
        self.ping_result = True
        self.connected = False

    async def connect(self):
        self.connected = True

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        if self.raise_on_call:
            raise self.raise_on_call
        return {"name": name, "arguments": arguments}

    def list_tools(self):
        return list(self.tool_list)

    def ping(self):
        return self.ping_result

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture()
def stub_components(monkeypatch):
    components = _McpClientComponents(
        session_cls=_StubSession,
        stdio_transport_cls=_StubTransport,
        websocket_transport_cls=None,
    )
    monkeypatch.setattr(McpToolProvider, "_get_components", lambda self: components)
    return components


def _build_provider(**config_overrides: object) -> McpToolProvider:
    config = {
        "server": "local",
        "servers": {"local": {"transport": "stdio", "command": "stub"}},
    }
    config.update(config_overrides)
    spec = ToolProviderSpec(name="mcp", config=MappingProxyType(config))
    return McpToolProvider(spec, tool_name="echo")


def test_mcp_provider_invokes_tool(monkeypatch, stub_components):
    provider = _build_provider()

    result = asyncio.run(provider.call(server="local", payload={"message": "hi"}))

    assert result["name"] == "echo"
    assert result["arguments"] == {"message": "hi"}
    session_handle = provider._sessions.get("local")
    assert session_handle is not None
    assert session_handle.session.calls == [("echo", {"message": "hi"})]


def test_mcp_provider_records_failure_backoff(monkeypatch, stub_components):
    provider = _build_provider()
    session_handle = asyncio.run(provider._get_or_create_session("local"))
    session_handle.session.raise_on_call = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(provider.call(server="local", payload={}))

    timestamps = iter([100.0, 110.0])
    monkeypatch.setattr("modules.Tools.providers.base.time.time", lambda: next(timestamps))

    first_delay = provider.health.record_failure()
    second_delay = provider.health.record_failure()

    assert first_delay == pytest.approx(1.0)
    assert second_delay == pytest.approx(2.0)
    assert provider.health.backoff_until == pytest.approx(112.0)


def test_mcp_provider_health_check_uses_ping(monkeypatch, stub_components):
    provider = _build_provider()
    result = asyncio.run(provider.health_check())
    assert result is True


def test_mcp_provider_health_check_falls_back_to_list(monkeypatch):
    class _ListingSession:
        def __init__(self, transport):
            self.transport = transport
            self.checked = False

        async def connect(self):
            return None

        def list_tools(self):
            self.checked = True
            return [{"name": "alpha", "description": "tool", "input_schema": {"type": "object"}}]

    components = _McpClientComponents(
        session_cls=_ListingSession,
        stdio_transport_cls=_StubTransport,
        websocket_transport_cls=None,
    )
    monkeypatch.setattr(McpToolProvider, "_get_components", lambda self: components)

    provider = _build_provider()
    result = asyncio.run(provider.health_check())
    assert result is True
    session_handle = provider._sessions.get("local")
    assert session_handle is not None
    assert session_handle.session.checked is True
