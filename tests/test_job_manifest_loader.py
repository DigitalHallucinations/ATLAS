import json
from pathlib import Path

import pytest

from modules.Jobs.manifest_loader import JobManifestError, JobMetadata


@pytest.fixture()
def job_app_root(tmp_path: Path) -> Path:
    (tmp_path / "modules" / "Jobs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _shared_job_payload() -> dict:
    return {
        "name": "SharedJob",
        "summary": "Baseline job summary",
        "description": "Coordinates multiple tasks to deliver a shared outcome.",
        "personas": ["ATLAS", "ResumeGenius"],
        "required_skills": ["analysis"],
        "required_tools": ["browser"],
        "task_graph": [
            {"task": "CollectData"},
            {"task": "SynthesizeInsights", "depends_on": ["CollectData"]},
        ],
        "recurrence": {"frequency": "weekly", "timezone": "UTC"},
        "acceptance_criteria": ["Insights delivered"],
        "escalation_policy": {"level": "tier1", "contact": "oncall@example.com"},
    }


def test_job_loader_supports_persona_overrides(job_app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_manifest = job_app_root / "modules" / "Jobs" / "jobs.json"
    _write_json(shared_manifest, [_shared_job_payload()])

    persona_manifest = job_app_root / "modules" / "Personas" / "Atlas" / "Jobs" / "jobs.json"
    persona_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        persona_manifest,
        [
            {
                "name": "SharedJob",
                "extends": "SharedJob",
                "summary": "Atlas specific overview",
                "personas": ["Atlas"],
                "required_tools": ["browser", "atlas_dashboard"],
                "task_graph": [
                    {"task": "CollectData"},
                    {
                        "task": "AtlasSignalRouting",
                        "depends_on": ["CollectData"],
                        "description": "Map findings to Atlas initiatives.",
                    },
                    {
                        "task": "SynthesizeInsights",
                        "depends_on": ["AtlasSignalRouting"],
                    },
                ],
                "acceptance_criteria": ["Atlas stakeholders receive action plan"],
            },
            {
                "name": "AtlasExclusive",
                "summary": "Persona specific job",
                "description": "Only available to Atlas persona",
                "personas": ["Atlas"],
                "required_skills": ["analysis", "coordination"],
                "required_tools": ["browser", "notebook"],
                "task_graph": [
                    {"task": "Plan"},
                    {"task": "Execute", "depends_on": ["Plan"]},
                ],
                "recurrence": {"frequency": "monthly"},
                "acceptance_criteria": ["Persona deliverable complete"],
                "escalation_policy": {
                    "level": "tier1",
                    "contact": "atlas-oncall@example.com",
                    "actions": ["notify"],
                },
            },
        ],
    )

    from modules import Jobs as jobs_pkg

    monkeypatch.setattr(jobs_pkg.manifest_loader, "_resolve_app_root", lambda *_: job_app_root)

    entries = jobs_pkg.manifest_loader.load_job_metadata()

    assert entries == sorted(entries, key=lambda entry: ((entry.persona or ""), entry.name.lower()))

    shared = next(entry for entry in entries if entry.persona is None and entry.name == "SharedJob")
    atlas_override = next(
        entry for entry in entries if entry.persona == "Atlas" and entry.name == "SharedJob"
    )
    atlas_exclusive = next(
        entry for entry in entries if entry.persona == "Atlas" and entry.name == "AtlasExclusive"
    )

    assert shared.summary == "Baseline job summary"
    assert atlas_override.summary == "Atlas specific overview"
    assert atlas_override.required_skills == shared.required_skills
    assert atlas_override.required_tools == ("browser", "atlas_dashboard")
    assert atlas_override.acceptance_criteria == ("Atlas stakeholders receive action plan",)
    assert any(node["task"] == "AtlasSignalRouting" for node in atlas_override.task_graph)
    assert atlas_exclusive.required_skills == ("analysis", "coordination")
    assert atlas_exclusive.escalation_policy["contact"] == "atlas-oncall@example.com"


def test_invalid_manifest_payload_raises(job_app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_manifest = job_app_root / "modules" / "Jobs" / "jobs.json"
    _write_json(shared_manifest, {"not": "a list"})

    from modules import Jobs as jobs_pkg

    monkeypatch.setattr(jobs_pkg.manifest_loader, "_resolve_app_root", lambda *_: job_app_root)

    with pytest.raises(JobManifestError) as exc:
        jobs_pkg.manifest_loader.load_job_metadata()

    assert "must be a JSON array" in str(exc.value)


def test_invalid_job_entry_reports_schema_error(
    job_app_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_manifest = job_app_root / "modules" / "Jobs" / "jobs.json"
    _write_json(shared_manifest, [{"name": "Broken"}])

    from modules import Jobs as jobs_pkg

    monkeypatch.setattr(jobs_pkg.manifest_loader, "_resolve_app_root", lambda *_: job_app_root)

    with pytest.raises(JobManifestError) as exc:
        jobs_pkg.manifest_loader.load_job_metadata()

    message = str(exc.value)
    assert "job manifest validation error" in message.lower()
    assert "SharedJob" not in message


def test_unknown_base_reference_raises(job_app_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_manifest = job_app_root / "modules" / "Jobs" / "jobs.json"
    _write_json(shared_manifest, [_shared_job_payload()])

    persona_manifest = job_app_root / "modules" / "Personas" / "Atlas" / "Jobs" / "jobs.json"
    persona_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        persona_manifest,
        [
            {
                "name": "SharedJob",
                "extends": "Missing",
                "summary": "Atlas specific overview",
                "personas": ["Atlas"],
                "required_tools": ["browser"],
                "task_graph": [{"task": "CollectData"}],
                "acceptance_criteria": ["Atlas stakeholders receive action plan"],
            }
        ],
    )

    from modules import Jobs as jobs_pkg

    monkeypatch.setattr(jobs_pkg.manifest_loader, "_resolve_app_root", lambda *_: job_app_root)

    with pytest.raises(JobManifestError) as exc:
        jobs_pkg.manifest_loader.load_job_metadata()

    assert "references unknown base" in str(exc.value)
