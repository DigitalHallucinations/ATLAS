"""Authorization checks for server-managed mutation helpers."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List

import pytest

from modules.Server import AtlasServer, RequestContext
from modules.Server.conversation_routes import ConversationAuthorizationError
import modules.Server.routes as server_routes


class _ConfigStub:
    def __init__(self, tenant_id: str) -> None:
        self.config = {"tenant_id": tenant_id}
        self.yaml_config = {"tenant_id": tenant_id}


def _setup_persona_patches(monkeypatch: pytest.MonkeyPatch, metadata_tenant: str) -> List[Dict[str, Any]]:
    persona_payload: Dict[str, Any] = {
        "name": "Example",
        "metadata": {"tenant_id": metadata_tenant},
        "allowed_tools": [],
        "allowed_skills": [],
    }

    monkeypatch.setattr(
        server_routes,
        "load_persona_definition",
        lambda name, config_manager=None: persona_payload,
    )
    monkeypatch.setattr(
        server_routes,
        "load_tool_metadata",
        lambda config_manager=None: ([], {}),
    )
    monkeypatch.setattr(
        server_routes,
        "load_skill_catalog",
        lambda config_manager=None: ([], {}),
    )
    monkeypatch.setattr(
        server_routes,
        "normalize_allowed_tools",
        lambda tools, metadata_order=None: list(tools or []),
    )
    monkeypatch.setattr(
        server_routes,
        "normalize_allowed_skills",
        lambda skills, metadata_order=None: list(skills or []),
    )
    monkeypatch.setattr(
        server_routes,
        "_validate_persona_payload",
        lambda *args, **kwargs: None,
    )

    persisted: List[Dict[str, Any]] = []

    def _capture(name: str, persona: Dict[str, Any], **_: Any) -> None:
        persisted.append({"name": name, "persona": persona})

    monkeypatch.setattr(server_routes, "persist_persona_definition", _capture)
    return persisted


def _encoded_task_bundle(metadata: Dict[str, Any]) -> str:
    payload = {
        "metadata": metadata,
        "task": {"name": "demo", "persona": metadata.get("persona")},
        "signature": {"algorithm": "HS256", "value": "placeholder"},
    }
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def test_update_persona_tools_requires_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer(config_manager=_ConfigStub("tenant-alpha"))
    persisted = _setup_persona_patches(monkeypatch, "tenant-alpha")

    context = RequestContext(tenant_id="tenant-alpha", roles=("viewer",))

    with pytest.raises(ConversationAuthorizationError):
        server.update_persona_tools("Example", tools=["search"], context=context)

    assert not persisted


def test_update_persona_tools_rejects_mismatched_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer(config_manager=_ConfigStub("tenant-alpha"))
    persisted = _setup_persona_patches(monkeypatch, "tenant-alpha")

    context = RequestContext(tenant_id="tenant-beta", roles=("admin",))

    with pytest.raises(ConversationAuthorizationError):
        server.update_persona_tools("Example", tools=["search"], context=context)

    assert not persisted


def test_update_persona_tools_allows_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer(config_manager=_ConfigStub("tenant-alpha"))
    persisted = _setup_persona_patches(monkeypatch, "tenant-alpha")

    context = RequestContext(tenant_id="tenant-alpha", roles=("admin",))

    result = server.update_persona_tools("Example", tools=["search"], context=context)

    assert result["success"] is True
    assert persisted, "persona definition should be persisted"


def test_import_task_bundle_requires_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer(config_manager=_ConfigStub("tenant-alpha"))
    monkeypatch.setattr(server, "_resolve_signing_key", lambda asset: "secret")

    encoded = _encoded_task_bundle({"tenant_id": "tenant-alpha"})

    context = RequestContext(tenant_id="tenant-alpha", roles=())

    with pytest.raises(ConversationAuthorizationError):
        server.import_task_bundle(bundle_base64=encoded, context=context)


def test_import_task_bundle_rejects_cross_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer(config_manager=_ConfigStub("tenant-alpha"))
    monkeypatch.setattr(server, "_resolve_signing_key", lambda asset: "secret")

    encoded = _encoded_task_bundle({"tenant_id": "tenant-alpha"})

    context = RequestContext(tenant_id="tenant-beta", roles=("admin",))

    with pytest.raises(ConversationAuthorizationError):
        server.import_task_bundle(bundle_base64=encoded, context=context)


def test_import_task_bundle_allows_matching_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer(config_manager=_ConfigStub("tenant-alpha"))
    monkeypatch.setattr(server, "_resolve_signing_key", lambda asset: "secret")

    encoded = _encoded_task_bundle({"tenant_id": "tenant-alpha"})

    captured: Dict[str, Any] = {}

    def _fake_import(bundle_bytes: bytes, *, signing_key: str, config_manager: object | None, rationale: str) -> Dict[str, Any]:
        captured["bundle_bytes"] = bundle_bytes
        captured["rationale"] = rationale
        return {"success": True, "metadata": {"tenant_id": "tenant-alpha"}}

    monkeypatch.setattr(server_routes, "import_task_bundle_bytes", _fake_import)

    context = RequestContext(tenant_id="tenant-alpha", roles=("admin",))

    result = server.import_task_bundle(bundle_base64=encoded, context=context)

    assert result["success"] is True
    assert captured["bundle_bytes"], "import helper should be invoked"
