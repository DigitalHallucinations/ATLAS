"""Tests for modules.Server.routes helper functions."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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

from types import MappingProxyType

from modules.Server.conversation_routes import RequestContext
from modules.Server.routes import _as_bool, _parse_query_timestamp, AtlasServer
from modules.Tools.manifest_loader import ToolManifestEntry
from modules.Skills.manifest_loader import SkillMetadata
from modules.orchestration.capability_registry import ToolCapabilityView, SkillCapabilityView


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
        requires_flags={},
        providers=[],
        source="tests",
    )


def _make_tool_view(entry: ToolManifestEntry) -> ToolCapabilityView:
    empty_health = MappingProxyType(
        {
            "tool": MappingProxyType(
                {
                    "total": 0,
                    "success": 0,
                    "failure": 0,
                    "success_rate": 0.0,
                    "average_latency_ms": None,
                    "p95_latency_ms": None,
                    "last_sample_at": None,
                }
            ),
            "providers": MappingProxyType({}),
        }
    )
    return ToolCapabilityView(
        manifest=entry,
        capability_tags=tuple(entry.capabilities),
        auth_scopes=tuple(),
        health=empty_health,
    )


def _make_skill_view(entry: SkillMetadata) -> SkillCapabilityView:
    return SkillCapabilityView(
        manifest=entry,
        capability_tags=tuple(entry.capability_tags),
        required_capabilities=tuple(entry.required_capabilities),
    )


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (
            "2024-05-22T10:00:00Z",
            datetime(2024, 5, 22, 10, 0, tzinfo=timezone.utc),
        ),
        (
            "2024-05-22T10:00:00",
            datetime(2024, 5, 22, 10, 0, tzinfo=timezone.utc),
        ),
        ("not-a-timestamp", None),
    ],
)
def test_parse_query_timestamp_variants(raw_value, expected):
    parsed = _parse_query_timestamp(raw_value)
    assert parsed == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, False),
        ("", False),
        ("true", True),
        ("ON", True),
        ("false", False),
        (False, False),
        (True, True),
    ],
)
def test_as_bool_variants(raw_value, expected):
    assert _as_bool(raw_value) is expected


class _RegistryStub:
    def __init__(self, tools: List[ToolCapabilityView], skills: List[SkillCapabilityView]):
        self.tools = tools
        self.skills = skills

    def query_tools(self, **_filters):
        return list(self.tools)

    def query_skills(self, **filters):
        persona_filters = [token.lower() for token in filters.get("persona_filters") or []]
        if not persona_filters:
            return list(self.skills)

        exclude_shared = any(token == "-shared" for token in persona_filters)
        positive = [token for token in persona_filters if not token.startswith("-")]

        results: List[SkillCapabilityView] = []
        for view in self.skills:
            persona = view.manifest.persona or "shared"
            persona_token = persona.lower()
            if exclude_shared and persona_token == "shared":
                continue
            if positive and persona_token not in positive:
                continue
            results.append(view)
        return results


def test_get_tools_includes_shared_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_entry = _make_entry("shared_tool", persona=None)
    atlas_entry = _make_entry("atlas_only", persona="Atlas")
    other_entry = _make_entry("other_tool", persona="Other")

    registry = _RegistryStub(
        [_make_tool_view(shared_entry), _make_tool_view(atlas_entry), _make_tool_view(other_entry)],
        [],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_tools(persona="Atlas")

    assert response["count"] == 2
    names = {tool["name"] for tool in response["tools"]}
    assert names == {"shared_tool", "atlas_only"}


def test_get_tools_can_exclude_shared_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_entry = _make_entry("shared_tool", persona=None)
    atlas_entry = _make_entry("atlas_only", persona="Atlas")

    registry = _RegistryStub(
        [_make_tool_view(shared_entry), _make_tool_view(atlas_entry)],
        [],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
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
        collaboration=None,
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
        collaboration=None,
    )

    registry = _RegistryStub(
        [_make_tool_view(_make_entry("shared_tool", persona=None))],
        [_make_skill_view(shared_skill), _make_skill_view(atlas_skill)],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_skills(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    assert response["skills"][0]["name"] == "atlas_only"
    assert response["skills"][0]["summary"] == "Atlas details"

    routed = server.handle_request("/skills", method="GET")
    assert routed["count"] == 2
    assert {entry["category"] for entry in routed["skills"]} == {"Context", "Atlas"}


def test_handle_request_skills_details_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = AtlasServer()
    skill = SkillMetadata(
        name="custom_skill",
        version="1.0.0",
        instruction_prompt="Do the custom task",
        required_tools=[],
        required_capabilities=[],
        safety_notes="",
        summary="Custom skill summary",
        category="Custom",
        capability_tags=[],
        persona=None,
        source="tests",
        collaboration=None,
    )

    registry = _RegistryStub([], [_make_skill_view(skill)])
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    context = RequestContext(tenant_id="tenant-123", user_id="user-456")

    response = server.handle_request(
        "/skills/custom_skill",
        method="GET",
        context=context,
    )

    assert response["success"] is True
    assert response["skill"]["name"] == "custom_skill"
    assert response["tenant_id"] == context.tenant_id
