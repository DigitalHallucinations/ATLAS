from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Optional

import pytest

from modules.Server.routes import AtlasServer
from modules.Tools.manifest_loader import ToolManifestEntry
from modules.Skills.manifest_loader import SkillMetadata
from modules.orchestration.capability_registry import (
    SkillCapabilityView,
    ToolCapabilityView,
)


def _make_tool_entry(name: str, *, persona: Optional[str]) -> ToolManifestEntry:
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


class _RegistryStub:
    def __init__(
        self,
        *,
        tools: Iterable[ToolCapabilityView] = (),
        skills: Iterable[SkillCapabilityView] = (),
    ) -> None:
        self._tools = list(tools)
        self._skills = list(skills)

    def query_tools(self, **_filters: Any) -> List[ToolCapabilityView]:
        return list(self._tools)

    def query_skills(self, **filters: Any) -> List[SkillCapabilityView]:
        persona_filters = [token.lower() for token in filters.get("persona_filters") or []]
        if not persona_filters:
            return list(self._skills)

        exclude_shared = any(token == "-shared" for token in persona_filters)
        positive = [token for token in persona_filters if not token.startswith("-")]

        results: List[SkillCapabilityView] = []
        for view in self._skills:
            persona = (view.manifest.persona or "shared").lower()
            if exclude_shared and persona == "shared":
                continue
            if positive and persona not in positive:
                continue
            results.append(view)
        return results


class _ConfigStub:
    def __init__(self) -> None:
        self.tool_manifest_lookup: Optional[Dict[str, Dict[str, Any]]] = None
        self.skill_snapshot_args: Optional[Dict[str, Any]] = None

    def get_tool_config_snapshot(
        self,
        *,
        manifest_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        self.tool_manifest_lookup = manifest_lookup
        return {
            "atlas_only": {
                "settings": {"enabled": True},
                "credentials": {"token": "***"},
            },
            "shared_tool": {
                "settings": {"enabled": False},
                "credentials": {"token": "---"},
            },
        }

    def get_skill_config_snapshot(
        self,
        *,
        manifest_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        skill_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        self.skill_snapshot_args = {
            "manifest_lookup": manifest_lookup,
            "skill_names": list(skill_names or []),
        }
        return {
            "atlas_skill": {
                "settings": {"temperature": 0.1},
                "credentials": {"api_key": "abc"},
            },
            "shared_skill": {
                "settings": {"temperature": 0.5},
                "credentials": {"api_key": "xyz"},
            },
        }


def test_get_tools_injects_configuration_overlays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _ConfigStub()
    server = AtlasServer(config_manager=config)

    shared_entry = _make_tool_entry("shared_tool", persona=None)
    atlas_entry = _make_tool_entry("atlas_only", persona="Atlas")

    registry = _RegistryStub(
        tools=[_make_tool_view(shared_entry), _make_tool_view(atlas_entry)],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_tools(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    tool = response["tools"][0]
    assert tool["name"] == "atlas_only"
    assert tool["settings"] == {"enabled": True}
    assert tool["credentials"] == {"token": "***"}

    assert config.tool_manifest_lookup == {"atlas_only": {"auth": atlas_entry.auth}}


def test_get_skills_injects_configuration_overlays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _ConfigStub()
    server = AtlasServer(config_manager=config)

    shared_skill = SkillMetadata(
        name="shared_skill",
        version="1.0.0",
        instruction_prompt="Do shared things",
        required_tools=["tool_a"],
        required_capabilities=["shared"],
        safety_notes="",
        summary="Shared skill",
        category="Shared",
        capability_tags=["shared"],
        persona=None,
        source="tests",
        collaboration=None,
    )
    atlas_skill = SkillMetadata(
        name="atlas_skill",
        version="2.0.0",
        instruction_prompt="Do atlas things",
        required_tools=[],
        required_capabilities=[],
        safety_notes="",
        summary="Atlas skill",
        category="Atlas",
        capability_tags=[],
        persona="Atlas",
        source="tests",
        collaboration=None,
    )

    registry = _RegistryStub(
        skills=[_make_skill_view(shared_skill), _make_skill_view(atlas_skill)],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_skills(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    skill = response["skills"][0]
    assert skill["name"] == "atlas_skill"
    assert skill["settings"] == {"temperature": 0.1}
    assert skill["credentials"] == {"api_key": "abc"}

    assert config.skill_snapshot_args == {
        "manifest_lookup": {"atlas_skill": {}},
        "skill_names": ["atlas_skill"],
    }
