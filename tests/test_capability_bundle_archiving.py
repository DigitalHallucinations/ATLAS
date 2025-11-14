from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.Tools import (
    export_tool_bundle_bytes,
    import_tool_bundle_bytes,
)
from modules.Skills import (
    export_skill_bundle_bytes,
    import_skill_bundle_bytes,
)
from modules.Jobs import (
    export_job_bundle_bytes,
    import_job_bundle_bytes,
)
from modules.Tasks import (
    export_task_bundle_bytes,
    import_task_bundle_bytes,
)
from modules.store_common.package_bundles import (
    export_asset_package_bytes,
    import_asset_package_bytes,
)


class _StubConfigManager:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_tool_bundle_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    tool_manifest = root / "modules" / "Tools" / "tool_maps" / "functions.json"
    sample_tool = [
        {
            "name": "alpha_tool",
            "description": "Example tool",
            "auth": {"required": False},
            "capabilities": ["demo"],
        }
    ]
    _write_json(tool_manifest, sample_tool)

    bundle_bytes, tool_entry = export_tool_bundle_bytes(
        "alpha_tool",
        signing_key="secret",
        config_manager=config,
    )
    assert tool_entry["name"] == "alpha_tool"

    tool_manifest.unlink()

    result = import_tool_bundle_bytes(
        bundle_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Test import",
    )
    assert result["success"] is True

    reloaded = json.loads(tool_manifest.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "alpha_tool" for entry in reloaded)


def test_skill_bundle_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    skill_manifest = root / "modules" / "Skills" / "skills.json"
    sample_skill = [
        {
            "name": "memory_skill",
            "version": "1.0",
            "instruction_prompt": "Remember context.",
            "required_tools": ["alpha_tool"],
            "required_capabilities": ["memory"],
            "safety_notes": "Verify stored data before use.",
        }
    ]
    _write_json(skill_manifest, sample_skill)

    bundle_bytes, skill_entry = export_skill_bundle_bytes(
        "memory_skill",
        signing_key="secret",
        config_manager=config,
    )
    assert skill_entry["name"] == "memory_skill"

    skill_manifest.unlink()

    result = import_skill_bundle_bytes(
        bundle_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Test import",
    )
    assert result["success"] is True

    reloaded = json.loads(skill_manifest.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "memory_skill" for entry in reloaded)


def test_job_bundle_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    job_manifest = root / "modules" / "Jobs" / "jobs.json"
    sample_job = [
        {
            "name": "DailySummary",
            "summary": "Summarize conversations",
            "description": "Compile a daily summary.",
            "personas": ["Atlas"],
            "required_skills": [],
            "required_tools": [],
            "task_graph": [
                {
                    "task": "prepare-summary",
                    "description": "Collect messages and draft a summary.",
                }
            ],
            "recurrence": {},
            "acceptance_criteria": [
                "Summary document generated",
            ],
            "escalation_policy": {
                "level": "info",
                "contact": "alerts@example.com",
            },
        }
    ]
    _write_json(job_manifest, sample_job)

    bundle_bytes, job_entry = export_job_bundle_bytes(
        "DailySummary",
        signing_key="secret",
        config_manager=config,
    )
    assert job_entry["name"] == "DailySummary"

    job_manifest.unlink()

    result = import_job_bundle_bytes(
        bundle_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Test import",
    )
    assert result["success"] is True

    reloaded = json.loads(job_manifest.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "DailySummary" for entry in reloaded)


def test_task_bundle_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    task_manifest = root / "modules" / "Tasks" / "tasks.json"
    sample_task = [
        {
            "name": "ReviewDocs",
            "summary": "Review documentation updates",
            "description": "Ensure docs are accurate.",
            "required_skills": ["memory_skill"],
            "required_tools": ["alpha_tool"],
            "acceptance_criteria": ["All pages reviewed"],
            "escalation_policy": {"level": "info", "contact": "ops@example.com"},
            "tags": ["docs"],
            "priority": "medium",
        }
    ]
    _write_json(task_manifest, sample_task)

    bundle_bytes, task_entry = export_task_bundle_bytes(
        "ReviewDocs",
        signing_key="secret",
        config_manager=config,
    )
    assert task_entry["name"] == "ReviewDocs"

    task_manifest.unlink()

    result = import_task_bundle_bytes(
        bundle_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Test import",
    )
    assert result["success"] is True

    reloaded = json.loads(task_manifest.read_text(encoding="utf-8"))
    assert any(entry.get("name") == "ReviewDocs" for entry in reloaded)


def test_asset_package_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "app"
    config = _StubConfigManager(root)

    tool_manifest = root / "modules" / "Tools" / "tool_maps" / "functions.json"
    skill_manifest = root / "modules" / "Skills" / "skills.json"
    task_manifest = root / "modules" / "Tasks" / "tasks.json"
    job_manifest = root / "modules" / "Jobs" / "jobs.json"

    _write_json(
        tool_manifest,
        [
            {
                "name": "alpha_tool",
                "description": "Example tool",
                "auth": {"required": False},
                "capabilities": ["demo"],
            }
        ],
    )
    _write_json(
        skill_manifest,
        [
            {
                "name": "memory_skill",
                "version": "1.0",
                "instruction_prompt": "Remember context.",
                "required_tools": ["alpha_tool"],
                "required_capabilities": ["memory"],
                "safety_notes": "Verify stored data before use.",
            }
        ],
    )
    _write_json(
        task_manifest,
        [
            {
                "name": "ReviewDocs",
                "summary": "Review documentation updates",
                "description": "Ensure docs are accurate.",
                "required_skills": ["memory_skill"],
                "required_tools": ["alpha_tool"],
                "acceptance_criteria": ["All pages reviewed"],
                "escalation_policy": {"level": "info", "contact": "ops@example.com"},
                "tags": ["docs"],
                "priority": "medium",
            }
        ],
    )
    _write_json(
        job_manifest,
        [
            {
                "name": "DailySummary",
                "summary": "Summarize conversations",
                "description": "Compile a daily summary.",
                "personas": ["Atlas"],
                "required_skills": ["memory_skill"],
                "required_tools": ["alpha_tool"],
                "task_graph": [
                    {
                        "task": "prepare-summary",
                        "description": "Collect messages and draft a summary.",
                    }
                ],
                "recurrence": {},
                "acceptance_criteria": ["Summary document generated"],
                "escalation_policy": {"level": "info", "contact": "ops@example.com"},
            }
        ],
    )

    package_bytes, metadata = export_asset_package_bytes(
        personas=None,
        tools=["alpha_tool"],
        skills=["memory_skill"],
        tasks=["ReviewDocs"],
        jobs=["DailySummary"],
        signing_key="secret",
        config_manager=config,
    )
    assert metadata["counts"]["tools"] == 1
    assert metadata["counts"]["tasks"] == 1

    # remove manifests to ensure import recreates them
    tool_manifest.unlink()
    skill_manifest.unlink()
    task_manifest.unlink()
    job_manifest.unlink()

    result = import_asset_package_bytes(
        package_bytes,
        signing_key="secret",
        config_manager=config,
        rationale="Test package import",
    )
    assert result["success"] is True
    assert len(result.get("assets", [])) == 4

    assert tool_manifest.exists()
    assert skill_manifest.exists()
    assert task_manifest.exists()
    assert job_manifest.exists()
