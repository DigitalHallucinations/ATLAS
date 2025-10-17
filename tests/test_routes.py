"""Tests for modules.Server.routes helper functions."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import List, Optional

import pytest


class _DummyValidator:
    def __init__(self, _schema):
        self.schema = _schema

    def iter_errors(self, _payload):
        return []


if "jsonschema" not in sys.modules:
    sys.modules["jsonschema"] = SimpleNamespace(
        Draft202012Validator=_DummyValidator,
        exceptions=SimpleNamespace(SchemaError=Exception),
    )

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: {},
        dump=lambda *_args, **_kwargs: None,
    )

from modules.Server.routes import AtlasServer
from modules.Tools.manifest_loader import ToolManifestEntry


def _make_entry(name: str, *, persona: Optional[str]) -> ToolManifestEntry:
    return ToolManifestEntry(
        name=name,
        persona=persona,
        description=f"{name} description",
        version="1.0.0",
        capabilities=["test"],
        auth={"required": False},
        safety_level="low",
        requires_consent=False,
        allow_parallel=True,
        idempotency_key=None,
        default_timeout=None,
        side_effects=None,
        cost_per_call=None,
        cost_unit=None,
        persona_allowlist=[],
        providers=[],
        source="tests",
    )


def _mock_manifest_entries(*entries: ToolManifestEntry) -> List[ToolManifestEntry]:
    return list(entries)


def test_get_tools_includes_shared_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_entry = _make_entry("shared_tool", persona=None)
    atlas_entry = _make_entry("atlas_only", persona="Atlas")
    other_entry = _make_entry("other_tool", persona="Other")

    monkeypatch.setattr(
        "modules.Server.routes.load_manifest_entries",
        lambda: _mock_manifest_entries(shared_entry, atlas_entry, other_entry),
    )

    response = server.get_tools(persona="Atlas")

    assert response["count"] == 2
    names = {tool["name"] for tool in response["tools"]}
    assert names == {"shared_tool", "atlas_only"}


def test_get_tools_can_exclude_shared_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_entry = _make_entry("shared_tool", persona=None)
    atlas_entry = _make_entry("atlas_only", persona="Atlas")

    monkeypatch.setattr(
        "modules.Server.routes.load_manifest_entries",
        lambda: _mock_manifest_entries(shared_entry, atlas_entry),
    )

    response = server.get_tools(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    names = {tool["name"] for tool in response["tools"]}
    assert names == {"atlas_only"}
