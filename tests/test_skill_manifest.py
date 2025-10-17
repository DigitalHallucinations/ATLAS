import json
from pathlib import Path

import pytest

from modules.Skills.manifest_loader import SkillMetadata, load_skill_metadata


def test_load_skill_metadata_includes_shared_and_persona():
    entries = load_skill_metadata()
    assert entries, "Expected at least one skill entry"

    ordering = [(entry.persona or "", entry.name.lower()) for entry in entries]
    assert ordering == sorted(ordering), "Entries should be sorted by persona then name"

    shared_names = {entry.name for entry in entries if entry.persona is None}
    persona_names = {entry.name for entry in entries if entry.persona == "ATLAS"}

    assert {"ContextualSummarizer", "SafetyScout"}.issubset(shared_names)
    assert "AtlasReporter" in persona_names


@pytest.mark.parametrize("invalid_entry", [
    {"name": "BrokenSkill", "version": "1.0"},
    "not-a-dict",
])
def test_invalid_skill_entries_are_skipped(tmp_path: Path, monkeypatch, invalid_entry):
    root = tmp_path
    shared_dir = root / "modules" / "Skills"
    shared_dir.mkdir(parents=True)

    valid_entry = {
        "name": "ValidSkill",
        "version": "3.2.1",
        "instruction_prompt": "Execute valid operations only.",
        "required_tools": ["logger"],
        "required_capabilities": ["compliance"],
        "safety_notes": "Always log activity."
    }

    manifest_payload = [valid_entry, invalid_entry]
    (shared_dir / "skills.json").write_text(json.dumps(manifest_payload), encoding="utf-8")

    personas_root = root / "modules" / "Personas"
    personas_root.mkdir(parents=True)

    from modules import Skills as skills_pkg
    monkeypatch.setattr(skills_pkg.manifest_loader, "_resolve_app_root", lambda *_: root)

    entries = skills_pkg.load_skill_metadata()

    assert entries == [
        SkillMetadata(
            name="ValidSkill",
            version="3.2.1",
            instruction_prompt="Execute valid operations only.",
            required_tools=["logger"],
            required_capabilities=["compliance"],
            safety_notes="Always log activity.",
            persona=None,
            source="modules/Skills/skills.json",
        )
    ]
