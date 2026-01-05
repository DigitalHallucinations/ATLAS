from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from core.SkillManager import SkillRunResult
from modules.Server.routes import AtlasServer
from modules.logging.audit import SkillAuditEntry
from modules.orchestration.capability_registry import SkillCapabilityView
from modules.Skills.manifest_loader import SkillMetadata


class StubConfigManager:
    """Lightweight config manager stub used for server route tests."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = {"tenant_id": "tenant-test"}
        self.yaml_config: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, str]] = {}

    def get_skill_metadata(self, skill_name: str) -> Dict[str, Any]:
        return dict(self._metadata.get(skill_name, {}))

    def set_skill_metadata(
        self, skill_name: str, metadata: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        record = dict(self._metadata.get(skill_name, {}))
        if metadata is None:
            record = {}
        else:
            if "review_status" in metadata:
                value = metadata.get("review_status")
                if value is None:
                    record.pop("review_status", None)
                else:
                    text = str(value).strip()
                    if text:
                        record["review_status"] = text
                    else:
                        record.pop("review_status", None)
            if "tester_notes" in metadata:
                value = metadata.get("tester_notes")
                if value is None:
                    record.pop("tester_notes", None)
                else:
                    text = str(value).strip()
                    if text:
                        record["tester_notes"] = text
                    else:
                        record.pop("tester_notes", None)
        self._metadata[skill_name] = dict(record)
        return dict(record)

    def get_skill_config_snapshot(self, **_kwargs: Any) -> Dict[str, Any]:
        return {}


@dataclass
class DummyRegistry:
    views: list[SkillCapabilityView]

    def query_skills(
        self,
        *,
        persona_filters: Any = None,
        capability_filters: Any = None,
        version_constraint: Any = None,
    ) -> list[SkillCapabilityView]:
        return list(self.views)


def _build_view(*, persona: str | None = None) -> SkillCapabilityView:
    manifest = SkillMetadata(
        name="ExampleSkill",
        version="1.0.0",
        instruction_prompt="Demonstration prompt.",
        required_tools=["example_tool"],
        required_capabilities=["capability.a"],
        safety_notes="Use responsibly.",
        summary="Example summary",
        category="demo",
        capability_tags=["capability.a"],
        persona=persona,
        source="unit-test",
        collaboration=None,
    )
    return SkillCapabilityView(
        manifest=manifest,
        capability_tags=tuple(manifest.capability_tags),
        required_capabilities=tuple(manifest.required_capabilities),
    )


def _patch_registry(monkeypatch: pytest.MonkeyPatch, registry: DummyRegistry) -> None:
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda **_kwargs: registry,
    )


def test_get_skill_details_returns_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    config_manager = StubConfigManager()
    config_manager.set_skill_metadata(
        "ExampleSkill",
        {"review_status": "pending", "tester_notes": "Needs review"},
    )
    view = _build_view()

    registry = DummyRegistry([view])
    _patch_registry(monkeypatch, registry)

    server = AtlasServer(config_manager=config_manager)
    result = server.get_skill_details("ExampleSkill")

    assert result["success"] is True
    assert result["skill"]["name"] == "ExampleSkill"
    assert result["metadata"]["review_status"] == "pending"
    assert result["metadata"]["tester_notes"] == "Needs review"
    assert result["instruction_prompt"] == "Demonstration prompt."


def test_validate_skill_success(monkeypatch: pytest.MonkeyPatch) -> None:
    config_manager = StubConfigManager()
    view = _build_view()
    registry = DummyRegistry([view])
    _patch_registry(monkeypatch, registry)

    async def fake_use_skill(*_args: Any, **_kwargs: Any) -> SkillRunResult:
        return SkillRunResult(
            skill_name="ExampleSkill",
            tool_results={"status": "ok"},
            metadata={"details": "completed"},
            version="1.0.0",
            required_capabilities=("capability.a",),
        )

    monkeypatch.setattr("ATLAS.SkillManager.use_skill", fake_use_skill)

    server = AtlasServer(config_manager=config_manager)
    response = server.validate_skill(
        "ExampleSkill",
        context={"tenant_id": "tenant-alpha", "user_id": "tester"},
    )

    assert response["success"] is True
    assert response["result"]["tool_results"] == {"status": "ok"}
    assert response["result"]["metadata"] == {"details": "completed"}


def test_validate_skill_failure_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    config_manager = StubConfigManager()
    view = _build_view()
    registry = DummyRegistry([view])
    _patch_registry(monkeypatch, registry)

    async def failing_use_skill(*_args: Any, **_kwargs: Any) -> SkillRunResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("ATLAS.SkillManager.use_skill", failing_use_skill)

    server = AtlasServer(config_manager=config_manager)
    result = server.validate_skill("ExampleSkill")

    assert result["success"] is False
    assert "boom" in result["error"]


def test_set_skill_metadata_persists_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    config_manager = StubConfigManager()
    config_manager.set_skill_metadata(
        "ExampleSkill", {"review_status": "pending", "tester_notes": "Initial"}
    )
    view = _build_view(persona="alpha")
    registry = DummyRegistry([view])
    _patch_registry(monkeypatch, registry)

    logged: dict[str, Any] = {}

    class DummyAuditLogger:
        def record_change(
            self,
            persona_name: str,
            skill_name: str,
            *,
            old_review_status: str | None = None,
            new_review_status: str | None = None,
            old_tester_notes: str | None = None,
            new_tester_notes: str | None = None,
            summary: str | None = None,
            username: str | None = None,
            timestamp: Any = None,
        ) -> SkillAuditEntry:
            logged.update(
                {
                    "persona": persona_name,
                    "skill": skill_name,
                    "old_review_status": old_review_status,
                    "new_review_status": new_review_status,
                    "old_tester_notes": old_tester_notes,
                    "new_tester_notes": new_tester_notes,
                    "username": username,
                }
            )
            return SkillAuditEntry(
                timestamp="2024-01-01T00:00:00Z",
                persona_name=persona_name,
                skill_name=skill_name,
                username=username or "tester",
                review_status_before=old_review_status or "",
                review_status_after=new_review_status or "",
                tester_notes_before=old_tester_notes or "",
                tester_notes_after=new_tester_notes or "",
                summary=summary or "updated",
            )

    dummy_logger = DummyAuditLogger()
    monkeypatch.setattr(
        "modules.Server.routes.get_skill_audit_logger", lambda: dummy_logger
    )

    server = AtlasServer(config_manager=config_manager)
    response = server.set_skill_metadata(
        "ExampleSkill",
        persona="alpha",
        metadata={"review_status": "reviewed", "tester_notes": "Looks good"},
        context={"tenant_id": "tenant-test", "user_id": "auditor"},
    )

    assert response["success"] is True
    assert response["metadata"]["review_status"] == "reviewed"
    assert config_manager.get_skill_metadata("ExampleSkill")["tester_notes"] == "Looks good"
    assert logged["persona"] == "alpha"
    assert logged["skill"] == "ExampleSkill"
    assert logged["new_review_status"] == "reviewed"
    assert logged["username"] == "auditor"
