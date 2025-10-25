import asyncio
import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pytest

from modules.Tools.Base_Tools.vault_secrets import VaultSecretsTool


class _StubConfigManager:
    def __init__(self, snapshot: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.snapshot = snapshot or {
            "alpha": {
                "settings": {},
                "credentials": {
                    "API_KEY": {"configured": True, "hint": "***", "source": "env"}
                },
            }
        }
        self.snapshot_calls: list[Dict[str, Any]] = []
        self.credential_calls: list[Dict[str, Any]] = []

    def get_tool_config_snapshot(
        self,
        *,
        manifest_lookup: Optional[Mapping[str, Mapping[str, Any]]] = None,
        tool_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        self.snapshot_calls.append(
            {
                "manifest_lookup": manifest_lookup,
                "tool_names": tuple(tool_names) if tool_names else None,
            }
        )
        return self.snapshot

    def set_tool_credentials(
        self,
        tool_name: str,
        credentials: Mapping[str, Any],
        *,
        manifest_auth: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        payload = {
            "tool_name": tool_name,
            "credentials": dict(credentials),
            "manifest_auth": manifest_auth,
        }
        self.credential_calls.append(payload)
        return {
            tool_name: {
                "credentials": {
                    key: {"configured": value is not None, "hint": "***"}
                    for key, value in credentials.items()
                }
            }
        }


def test_vault_secrets_tool_returns_masked_snapshot() -> None:
    stub = _StubConfigManager()
    tool = VaultSecretsTool(config_manager=stub)

    result = asyncio.run(
        tool.run(
            operation="get_snapshot",
            tool_names=["alpha"],
            manifest_lookup={"alpha": {"auth": {"env": "API_KEY"}}},
            persona="PersonaA",
        )
    )

    assert "alpha" in result
    assert result["alpha"]["credentials"]["API_KEY"]["hint"] == "***"
    assert stub.snapshot_calls[0]["tool_names"] == ("alpha",)


def test_vault_secrets_tool_persists_and_clears_credentials() -> None:
    stub = _StubConfigManager()
    tool = VaultSecretsTool(config_manager=stub)

    store_result = asyncio.run(
        tool.run(
            operation="store_credentials",
            tool_name="alpha",
            credentials={"API_KEY": "secret", " EXPIRED ": "value"},
            manifest_auth={"env": "API_KEY"},
            persona="PersonaB",
        )
    )

    assert "alpha" in store_result
    assert stub.credential_calls[0]["tool_name"] == "alpha"
    assert stub.credential_calls[0]["credentials"] == {
        "API_KEY": "secret",
        "EXPIRED": "value",
    }

    clear_result = asyncio.run(
        tool.run(
            operation="clear_credentials",
            tool_name="alpha",
            env_keys=["API_KEY"],
            persona="PersonaB",
        )
    )

    assert "alpha" in clear_result
    assert stub.credential_calls[1]["credentials"] == {"API_KEY": None}


def test_tool_manager_exposes_vault_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_manager = importlib.import_module("ATLAS.ToolManager")

    aiohttp_stub = types.ModuleType("aiohttp")

    class _DummyClientTimeout:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total")

    class _DummyClientSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):  # pragma: no cover - not exercised in tests
            return self

        async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover - not exercised
            return False

        async def get(self, *args, **kwargs):  # pragma: no cover - not exercised
            raise RuntimeError("HTTP requests are not supported in tests")

    aiohttp_stub.ClientTimeout = _DummyClientTimeout
    aiohttp_stub.ClientSession = _DummyClientSession
    sys.modules["aiohttp"] = aiohttp_stub

    sys.modules.pop("modules.Tools.Base_Tools.webpage_fetch", None)
    sys.modules.pop("modules.Tools.tool_maps.maps", None)

    tool_manager = importlib.reload(tool_manager)

    monkeypatch.setattr(
        tool_manager.ConfigManager,
        "get_app_root",
        lambda self: os.fspath(Path.cwd()),
    )

    config_manager = tool_manager.ConfigManager()
    shared_map = tool_manager.load_default_function_map(
        refresh=True, config_manager=config_manager
    )

    assert "vault.secrets" in shared_map
    entry = shared_map["vault.secrets"]
    assert callable(entry.get("callable"))
    metadata = entry.get("metadata")
    assert metadata["capabilities"] == [
        "credential_management",
        "configuration_access",
    ]
