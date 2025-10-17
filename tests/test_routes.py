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
from modules.Skills.manifest_loader import SkillMetadata


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


def _mock_skill_entries(*entries: SkillMetadata) -> List[SkillMetadata]:
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


def test_get_skills_returns_serialized_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_skill = SkillMetadata(
        name="summarize",
        version="1.0.0",
        instruction_prompt="Do summary",
        required_tools=["web_search"],
        required_capabilities=["summaries"],
        safety_notes="",
        summary="Quick summary",
        category="Context",
        capability_tags=["summaries"],
        persona=None,
        source="tests",
    )
    atlas_skill = SkillMetadata(
        name="atlas_only",
        version="2.0.0",
        instruction_prompt="Do atlas",
        required_tools=[],
        required_capabilities=[],
        safety_notes="",
        summary="Atlas details",
        category="Atlas",
        capability_tags=[],
        persona="Atlas",
        source="tests",
    )

    monkeypatch.setattr(
        "modules.Server.routes.load_skill_metadata",
        lambda config_manager=None: _mock_skill_entries(shared_skill, atlas_skill),
    )

    response = server.get_skills(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    assert response["skills"][0]["name"] == "atlas_only"
    assert response["skills"][0]["summary"] == "Atlas details"

    routed = server.handle_request("/skills", method="GET")
    assert routed["count"] == 2
    assert {entry["category"] for entry in routed["skills"]} == {"Context", "Atlas"}
