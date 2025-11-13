import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import pytest

from modules.orchestration.capability_registry import CapabilityRegistry

class _DummyConfig:
    def __init__(self, root: Path) -> None:
        self._root = root

    def get_app_root(self) -> str:
        return str(self._root)


@pytest.fixture()
def capability_root(tmp_path: Path) -> Path:
    (tmp_path / "modules" / "Tools" / "tool_maps").mkdir(parents=True, exist_ok=True)
    (tmp_path / "modules" / "Skills").mkdir(parents=True, exist_ok=True)
    (tmp_path / "modules" / "Tasks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "modules" / "Jobs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_shared_manifests(
    root: Path,
    tools: list[dict],
    skills: list[dict],
    tasks: list[dict],
    jobs: Optional[list[dict]] = None,
) -> None:
    tool_path = root / "modules" / "Tools" / "tool_maps" / "functions.json"
    skill_path = root / "modules" / "Skills" / "skills.json"
    task_path = root / "modules" / "Tasks" / "tasks.json"
    job_path = root / "modules" / "Jobs" / "jobs.json"
    tool_path.write_text(json.dumps(tools), encoding="utf-8")
    skill_path.write_text(json.dumps(skills), encoding="utf-8")
    task_path.write_text(json.dumps(tasks), encoding="utf-8")
    job_payload = jobs if jobs is not None else []
    job_path.write_text(json.dumps(job_payload), encoding="utf-8")


def _write_persona_tools(root: Path, persona: str, tools: list[dict]) -> None:
    manifest_dir = root / "modules" / "Personas" / persona / "Toolbox"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "functions.json"
    manifest_path.write_text(json.dumps(tools), encoding="utf-8")


def _write_persona_tasks(root: Path, persona: str, tasks: list[dict]) -> None:
    manifest_dir = root / "modules" / "Personas" / persona / "Tasks"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "tasks.json"
    manifest_path.write_text(json.dumps(tasks), encoding="utf-8")


def _write_persona_jobs(root: Path, persona: str, jobs: list[dict]) -> None:
    manifest_dir = root / "modules" / "Personas" / persona / "Jobs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "jobs.json"
    manifest_path.write_text(json.dumps(jobs), encoding="utf-8")


def _minimal_tool(name: str) -> dict:
    return {
        "name": name,
        "description": "sample",
        "parameters": {
            "type": "object",
            "properties": {},
        },
        "version": "1.0.0",
        "side_effects": "none",
        "default_timeout": 30,
        "auth": {"required": False},
        "allow_parallel": True,
        "capabilities": ["sample"],
        "providers": [{"name": "primary"}],
    }


def _minimal_skill(name: str) -> dict:
    return {
        "name": name,
        "version": "1.0.0",
        "instruction_prompt": "run",
        "required_tools": [],
        "required_capabilities": ["sample"],
        "safety_notes": "Follow standard guardrails",
        "summary": "Sample skill",
        "category": "general",
        "capability_tags": ["sample"],
    }


def _minimal_task(name: str) -> dict:
    return {
        "name": name,
        "summary": f"Task {name}",
        "description": "",
        "required_skills": ["sample"],
        "required_tools": ["sample"],
        "acceptance_criteria": ["completed"],
        "escalation_policy": {
            "level": "tier1",
            "contact": "oncall@example.com",
        },
        "tags": ["sample"],
    }


def _minimal_job(name: str, *, personas: list[str]) -> dict:
    return {
        "name": name,
        "summary": f"Job {name}",
        "description": "Run sample workflow",
        "personas": personas,
        "required_skills": ["summary_skill"],
        "required_tools": ["summary_tool"],
        "task_graph": [
            {
                "task": "SummaryTask",
                "description": "Execute summary task",
            }
        ],
        "recurrence": {"frequency": "adhoc"},
        "acceptance_criteria": ["complete"],
        "escalation_policy": {
            "level": "tier1",
            "contact": "alerts@example.com",
        },
    }


def test_registry_filters_invalid_entries(capability_root: Path) -> None:
    tools = [_minimal_tool("valid_tool"), {"description": "missing name"}]
    skills = [_minimal_skill("valid_skill")]
    tasks = [_minimal_task("shared_task")]
    _write_shared_manifests(capability_root, tools, skills, tasks)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    assert registry.refresh(force=True)

    tools = registry.query_tools()
    assert len(tools) == 1
    assert tools[0].manifest.name == "valid_tool"


def test_registry_records_tool_metrics(capability_root: Path) -> None:
    tools = [_minimal_tool("measured_tool")]
    skills = [_minimal_skill("skill")]
    tasks = [_minimal_task("shared_task")]
    _write_shared_manifests(capability_root, tools, skills, tasks)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    registry.record_tool_execution(
        persona=None,
        tool_name="measured_tool",
        success=True,
        latency_ms=100,
    )
    registry.record_tool_execution(
        persona=None,
        tool_name="measured_tool",
        success=False,
        latency_ms=200,
    )

    view = registry.query_tools()[0]
    tool_health = view.health["tool"]
    assert tool_health["total"] == 2
    assert pytest.approx(tool_health["success_rate"], rel=1e-6) == 0.5
    assert pytest.approx(tool_health["average_latency_ms"], rel=1e-6) == 150.0


def test_registry_records_provider_metrics(capability_root: Path) -> None:
    tools = [_minimal_tool("provider_tool")]
    skills = [_minimal_skill("skill")]
    tasks = [_minimal_task("shared_task")]
    _write_shared_manifests(capability_root, tools, skills, tasks)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    summary = {
        "tool": "provider_tool",
        "selected": "primary",
        "success": True,
        "latency_ms": 42.0,
        "timestamp": 123.0,
        "providers": {
            "primary": {
                "successes": 1,
                "failures": 0,
                "consecutive_failures": 0,
                "failure_rate": 0.0,
                "last_success": 123.0,
                "last_failure": None,
                "last_check": 123.0,
                "backoff_until": None,
            }
        },
    }
    registry.record_provider_metrics(persona=None, tool_name="provider_tool", summary=summary)

    view = registry.query_tools()[0]
    provider_health = view.health["providers"]["primary"]["metrics"]
    assert provider_health["total"] == 1
    assert provider_health["success"] == 1
    assert provider_health["failure"] == 0
    last_call = view.health["providers"]["primary"].get("last_call")
    assert isinstance(last_call, Mapping)
    assert last_call.get("latency_ms") == 42.0
    assert last_call.get("success") is True
    assert view.health.get("last_invocation", {}).get("provider") == "primary"


def test_query_tools_includes_shared_persona_when_filtered(capability_root: Path) -> None:
    shared_tool = _minimal_tool("shared_tool")
    atlas_tool = _minimal_tool("atlas_tool")
    _write_shared_manifests(capability_root, [shared_tool], [], [_minimal_task("shared_task")])
    _write_persona_tools(capability_root, "Atlas", [atlas_tool])

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    atlas_results = registry.query_tools(persona_filters=["atlas"])
    atlas_names = {view.manifest.name for view in atlas_results}
    assert atlas_names == {"shared_tool", "atlas_tool"}

    atlas_without_shared = registry.query_tools(persona_filters=["atlas", "-shared"])
    atlas_without_shared_names = {view.manifest.name for view in atlas_without_shared}
    assert atlas_without_shared_names == {"atlas_tool"}


def test_query_tasks_supports_filters(capability_root: Path) -> None:
    shared_tool = _minimal_tool("shared_tool")
    shared_skill = _minimal_skill("shared_skill")
    shared_task = _minimal_task("SharedTask")
    _write_shared_manifests(capability_root, [shared_tool], [shared_skill], [shared_task])

    persona_tasks = [
        {
            "name": "SharedTask",
            "extends": "SharedTask",
            "summary": "Persona tuned summary",
            "tags": ["sample", "atlas"],
        },
        {
            "name": "AtlasTask",
            "summary": "Atlas specific task",
            "description": "Persona exclusive",
            "required_skills": ["sample", "advanced"],
            "required_tools": ["sample"],
            "acceptance_criteria": ["completed"],
            "escalation_policy": {
                "level": "tier1",
                "contact": "atlas@example.com",
            },
            "tags": ["atlas", "priority"],
        },
    ]
    _write_persona_tasks(capability_root, "Atlas", persona_tasks)

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    atlas_results = registry.query_tasks(persona_filters=["atlas"])
    atlas_names = {view.manifest.name for view in atlas_results}
    assert atlas_names == {"SharedTask", "AtlasTask"}

    persona_only = registry.query_tasks(persona_filters=["atlas", "-shared"])
    assert {view.manifest.persona for view in persona_only} == {"Atlas"}

    tag_filtered = registry.query_tasks(tag_filters=["priority"])
    assert {view.manifest.name for view in tag_filtered} == {"AtlasTask"}

    skill_filtered = registry.query_tasks(required_skill_filters=["advanced"])
    assert {view.manifest.name for view in skill_filtered} == {"AtlasTask"}


def test_query_jobs_respects_persona_allowlist(capability_root: Path) -> None:
    shared_tool = _minimal_tool("summary_tool")
    shared_skill = _minimal_skill("summary_skill")
    shared_task = _minimal_task("SummaryTask")
    shared_job = _minimal_job("GlobalJob", personas=["Atlas", "Nova"])
    _write_shared_manifests(
        capability_root,
        [shared_tool],
        [shared_skill],
        [shared_task],
        jobs=[shared_job],
    )

    atlas_job = _minimal_job("AtlasOnly", personas=["Atlas"])
    _write_persona_jobs(capability_root, "Atlas", [atlas_job])

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    atlas_views = registry.query_jobs(persona_filters=["atlas"])
    assert {view.manifest.name for view in atlas_views} == {"GlobalJob", "AtlasOnly"}

    nova_views = registry.query_jobs(persona_filters=["nova"])
    assert {view.manifest.name for view in nova_views} == {"GlobalJob"}

    persona_only = registry.query_jobs(persona_filters=["atlas", "-shared"])
    assert {view.manifest.name for view in persona_only} == {"AtlasOnly"}


def test_get_task_catalog_prefers_persona_variants(capability_root: Path) -> None:
    shared_tool = _minimal_tool("shared_tool")
    shared_skill = _minimal_skill("shared_skill")
    shared_task = _minimal_task("SharedTask")
    _write_shared_manifests(capability_root, [shared_tool], [shared_skill], [shared_task])

    _write_persona_tasks(
        capability_root,
        "Atlas",
        [
            {
                "name": "SharedTask",
                "extends": "SharedTask",
                "summary": "Atlas version",
            }
        ],
    )

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)

    shared_catalog = registry.get_task_catalog(persona=None)
    assert len(shared_catalog) == 1
    assert shared_catalog[0].manifest.summary == "Task SharedTask"

    atlas_catalog = registry.get_task_catalog(persona="Atlas")
    assert len(atlas_catalog) == 1
    assert atlas_catalog[0].manifest.summary == "Atlas version"


def test_registry_refresh_detects_task_manifest_changes(capability_root: Path) -> None:
    shared_tool = _minimal_tool("shared_tool")
    shared_skill = _minimal_skill("shared_skill")
    shared_task = _minimal_task("SharedTask")
    _write_shared_manifests(capability_root, [shared_tool], [shared_skill], [shared_task])

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    assert registry.refresh(force=True)
    initial_revision = registry.revision

    assert not registry.refresh(force=False)

    additional_task = _minimal_task("SecondTask")
    _write_shared_manifests(
        capability_root,
        [shared_tool],
        [shared_skill],
        [shared_task, additional_task],
    )
    manifest_path = capability_root / "modules" / "Tasks" / "tasks.json"
    os.utime(manifest_path, (time.time() + 5, time.time() + 5))

    assert registry.refresh(force=False)
    assert registry.revision == initial_revision + 1
    catalog = registry.get_task_catalog(persona=None)
    assert {view.manifest.name for view in catalog} == {"SharedTask", "SecondTask"}


def test_registry_summary_includes_tasks_and_health(capability_root: Path) -> None:
    tool = _minimal_tool("summary_tool")
    skill = _minimal_skill("summary_skill")
    task = _minimal_task("SummaryTask")
    job = _minimal_job("SummaryJob", personas=["Atlas", "Nova"])
    _write_shared_manifests(capability_root, [tool], [skill], [task], jobs=[job])

    registry = CapabilityRegistry(config_manager=_DummyConfig(capability_root))
    registry.refresh(force=True)
    registry.record_tool_execution(
        persona=None,
        tool_name="summary_tool",
        success=True,
        latency_ms=120.0,
    )
    registry.record_job_execution(
        persona=None,
        job_name="SummaryJob",
        success=False,
        latency_ms=45.0,
    )

    summary = registry.summary(persona="Atlas")
    assert summary["persona"] == "Atlas"
    assert summary["revision"] == registry.revision

    tool_entry = next(entry for entry in summary["tools"] if entry["name"] == "summary_tool")
    assert tool_entry["compatibility"]["persona_specific"] is False
    assert tool_entry["health"]["tool"]["total"] == 1

    skill_entry = next(entry for entry in summary["skills"] if entry["name"] == "summary_skill")
    assert skill_entry["compatibility"]["matches_request"] is True

    task_entry = next(entry for entry in summary["tasks"] if entry["name"] == "SummaryTask")
    assert task_entry["compatibility"]["persona_specific"] is False
    assert set(task_entry["required_skills"]) == {"sample"}

    job_entry = next(entry for entry in summary["jobs"] if entry["name"] == "SummaryJob")
    assert job_entry["compatibility"]["matches_request"] is True
    assert set(job_entry["personas"]) == {"Atlas", "Nova"}
    assert set(job_entry["required_capabilities"]) == {"sample"}
    assert job_entry["health"]["job"]["total"] == 1
