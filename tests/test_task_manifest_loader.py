import json
from pathlib import Path

import pytest

from modules.Tasks.manifest_loader import TaskMetadata


@pytest.fixture()
def task_app_root(tmp_path: Path) -> Path:
    (tmp_path / "modules" / "Tasks").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _shared_task_payload() -> dict:
    return {
        "name": "SharedTask",
        "summary": "Baseline summary",
        "description": "Shared task description",
        "required_skills": ["analysis"],
        "required_tools": ["browser"],
        "acceptance_criteria": ["Complete analysis report"],
        "escalation_policy": {
            "level": "tier1",
            "contact": "oncall@example.com",
            "timeframe": "30m",
            "triggers": ["sla_breach"],
            "actions": ["page"],
        },
        "tags": ["reporting"],
        "priority": "standard",
    }


def test_task_loader_supports_persona_overrides(task_app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_manifest = task_app_root / "modules" / "Tasks" / "tasks.json"
    _write_json(shared_manifest, [_shared_task_payload()])

    persona_manifest = task_app_root / "modules" / "Personas" / "Atlas" / "Tasks" / "tasks.json"
    persona_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        persona_manifest,
        [
            {
                "name": "SharedTask",
                "extends": "SharedTask",
                "summary": "Atlas specific summary",
                "acceptance_criteria": ["Atlas-approved report"],
                "tags": ["reporting", "atlas"],
            },
            {
                "name": "AtlasExclusive",
                "summary": "Persona specific task",
                "description": "Only available to Atlas persona",
                "required_skills": ["analysis", "coordination"],
                "required_tools": ["browser", "notebook"],
                "acceptance_criteria": ["Persona report delivered"],
                "escalation_policy": {
                    "level": "tier1",
                    "contact": "atlas-oncall@example.com",
                    "actions": ["notify"],
                },
                "tags": ["atlas"],
            },
        ],
    )

    from modules import Tasks as tasks_pkg

    monkeypatch.setattr(tasks_pkg.manifest_loader, "_resolve_app_root", lambda *_: task_app_root)

    entries = tasks_pkg.manifest_loader.load_task_metadata()

    assert entries == sorted(entries, key=lambda entry: ((entry.persona or ""), entry.name.lower()))

    shared = next(entry for entry in entries if entry.persona is None and entry.name == "SharedTask")
    atlas_override = next(
        entry for entry in entries if entry.persona == "Atlas" and entry.name == "SharedTask"
    )
    atlas_exclusive = next(
        entry for entry in entries if entry.persona == "Atlas" and entry.name == "AtlasExclusive"
    )

    assert shared.summary == "Baseline summary"
    assert atlas_override.summary == "Atlas specific summary"
    assert atlas_override.required_skills == shared.required_skills
    assert atlas_override.acceptance_criteria == ("Atlas-approved report",)
    assert set(atlas_override.tags) == {"reporting", "atlas"}
    assert atlas_exclusive.required_skills == ("analysis", "coordination")
    assert atlas_exclusive.escalation_policy["contact"] == "atlas-oncall@example.com"


@pytest.mark.parametrize(
    "invalid_entry",
    [
        {"name": "Broken"},
        "not-a-dict",
    ],
)
def test_invalid_task_entries_are_skipped(
    task_app_root: Path, monkeypatch: pytest.MonkeyPatch, invalid_entry
) -> None:
    shared_manifest = task_app_root / "modules" / "Tasks" / "tasks.json"
    valid_task = _shared_task_payload()
    _write_json(shared_manifest, [valid_task, invalid_entry])

    from modules import Tasks as tasks_pkg

    monkeypatch.setattr(tasks_pkg.manifest_loader, "_resolve_app_root", lambda *_: task_app_root)

    entries = tasks_pkg.manifest_loader.load_task_metadata()

    assert entries == [
        TaskMetadata(
            name="SharedTask",
            summary="Baseline summary",
            description="Shared task description",
            required_skills=("analysis",),
            required_tools=("browser",),
            acceptance_criteria=("Complete analysis report",),
            escalation_policy={
                "level": "tier1",
                "contact": "oncall@example.com",
                "timeframe": "30m",
                "triggers": ["sla_breach"],
                "actions": ["page"],
            },
            tags=("reporting",),
            priority="standard",
            persona=None,
            source="modules/Tasks/tasks.json",
        )
    ]
