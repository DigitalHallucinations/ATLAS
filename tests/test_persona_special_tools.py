import asyncio
import importlib.util
from pathlib import Path
from typing import Any, Dict
import sys
import types

import pytest

_server_stub = types.ModuleType("modules.Server")
_server_stub.atlas_server = types.SimpleNamespace(get_task_catalog=lambda **_kwargs: {"count": 0, "tasks": []})
sys.modules.setdefault("modules.Server", _server_stub)


def _load_tool_module(identifier: str, relative_path: Path):
    spec = importlib.util.spec_from_file_location(identifier, relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = Path("modules") / "Personas"
atlas_catalog = _load_tool_module("atlas_catalog", BASE / "ATLAS" / "Toolbox" / "catalog.py")
backstory = _load_tool_module("cleverbot_backstory", BASE / "Cleverbot" / "Toolbox" / "backstory.py")
documentation = _load_tool_module("docgenius_documentation", BASE / "DocGenius" / "Toolbox" / "documentation.py")
thought_experiments = _load_tool_module("einstein_thought_experiments", BASE / "Einstein" / "Toolbox" / "thought_experiments.py")
audits = _load_tool_module("compliance_audits", BASE / "ComplianceOfficer" / "Toolbox" / "audits.py")
habits = _load_tool_module("healthcoach_habits", BASE / "HealthCoach" / "Toolbox" / "habits.py")
workouts = _load_tool_module("fitnesscoach_workouts", BASE / "FitnessCoach" / "Toolbox" / "workouts.py")
drills = _load_tool_module("languagetutor_drills", BASE / "LanguageTutor" / "Toolbox" / "drills.py")
problem_solver = _load_tool_module("mathtutor_problem_solver", BASE / "MathTutor" / "Toolbox" / "problem_solver.py")
knowledge_cards = _load_tool_module("knowledgecurator_cards", BASE / "KnowledgeCurator" / "Toolbox" / "knowledge_cards.py")
metaphors = _load_tool_module("genius_metaphors", BASE / "genius" / "Toolbox" / "metaphors.py")
innovation = _load_tool_module("nikola_innovation", BASE / "Nikola Tesla" / "Toolbox" / "innovation.py")


def test_task_catalog_snapshot_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    async def fake_to_thread(func, *args, **kwargs):
        captured.update(kwargs)
        return {"count": 0, "tasks": []}

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(
        atlas_catalog.task_catalog_snapshot(persona="DocGenius", tags=["priority"])
    )

    assert result == {"count": 0, "tasks": []}
    assert captured["persona"] == "DocGenius"
    assert captured["tags"] == ["priority"]


def test_persona_backstory_sampler_styles() -> None:
    payload = asyncio.run(backstory.persona_backstory_sampler("humor"))
    assert set(payload.keys()) == {"opener", "callback", "signoff"}


def test_generate_doc_outline_python_signature() -> None:
    outline = asyncio.run(documentation.generate_doc_outline("def foo(bar: int) -> int:"))
    assert outline["entity"]["name"] == "foo"
    assert any(param["name"] == "bar" for param in outline["sections"]["Parameters"])


def test_relativity_scenario_topics() -> None:
    scenario = asyncio.run(thought_experiments.relativity_scenario("gravity"))
    assert scenario["title"]
    assert any(prompt for prompt in scenario["discussion_prompts"])


def test_wireless_power_brief_shapes_response() -> None:
    brief = asyncio.run(innovation.wireless_power_brief("Test goal", ["Constraint"]))
    assert brief["goal"] == "Test goal"
    assert brief["constraint_adaptations"]


def test_metaphor_palette_returns_metaphors() -> None:
    palette = asyncio.run(metaphors.metaphor_palette("innovation", "leaders"))
    assert palette["metaphors"]
    assert palette["audience"] == "leaders"


def test_habit_stack_planner_includes_blockers() -> None:
    plan = asyncio.run(habits.habit_stack_planner("Recovery", "Weeknights", ["Screen time"]))
    assert plan["goal"] == "Recovery"
    assert any(item["blocker"] == "Screen time" for item in plan["blocker_strategies"])


def test_microcycle_plan_scales_to_days() -> None:
    plan = asyncio.run(workouts.microcycle_plan("strength", ["barbell"], 5))
    assert len(plan["schedule"]) == 5
    assert plan["equipment_notes"]


def test_dialogue_drill_levels() -> None:
    drill = asyncio.run(drills.dialogue_drill("ordering food", "beginner", "Spanish"))
    assert drill["language"] == "Spanish"
    assert drill["dialogue"]


def test_stepwise_solution_handles_problem() -> None:
    solution = asyncio.run(problem_solver.stepwise_solution("2*x + 3 = 7", "algebra"))
    assert solution["steps"]
    assert solution["focus"] == "algebra"


def test_regulatory_gap_audit_flags_missing() -> None:
    audit = asyncio.run(audits.regulatory_gap_audit("gdpr", ["data_processing_inventory"]))
    assert "breach_response_plan" in audit["missing"]


def test_knowledge_card_builder_defaults_review_cadence() -> None:
    card = asyncio.run(knowledge_cards.knowledge_card_builder("Query", ["Source"]))
    assert card["review_cadence"] == "quarterly"
