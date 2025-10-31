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
muse_tools = _load_tool_module("muse_toolbox", BASE / "Muse" / "Toolbox" / "maps.py")
hermes_helpers = _load_tool_module("hermes_helpers", BASE / "Hermes" / "Toolbox" / "helpers.py")
echo_tools = _load_tool_module("echo_toolbox", BASE / "Echo" / "Toolbox" / "maps.py")


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


def test_muse_storyweaver_generates_beats() -> None:
    story = asyncio.run(
        muse_tools.function_map["storyweaver"](
            theme="Rebirth",
            genre="mythic",
            characters=["Nova"],
            tone="hopeful",
        )
    )
    assert story["theme"] == "Rebirth"
    assert story["beats"] and story["beats"][0]["order"] == 1


def test_muse_lyricist_returns_hook_and_structure() -> None:
    payload = asyncio.run(
        muse_tools.function_map["lyricist"](
            concept="Aurora",
            structure=["verse", "chorus"],
            keywords=["radiant", "dawn"],
        )
    )
    assert payload["hook"].lower().startswith("hold to the aurora")
    assert "chorus_2" in payload["lyrics"]


def test_muse_visual_prompt_and_mood_map_alignment() -> None:
    visual = asyncio.run(
        muse_tools.function_map["visual_prompt"](
            subject="floating lanterns",
            style="watercolor",
            color_palette=["amber", "cobalt"],
            lighting="twilight",
        )
    )
    assert "floating lanterns" in visual["prompt"]
    mood = asyncio.run(
        muse_tools.function_map["mood_map"](
            beats=["spark", "crescendo", "afterglow"],
            start_mood="anticipation",
            target_mood="serenity",
            inflection="wonder",
        )
    )
    assert mood["beats"][0]["dominant_mood"].lower() == "anticipation"
    assert len(mood["palette"]) >= 1


def test_muse_emotive_tagger_defaults_when_no_keywords() -> None:
    tags = asyncio.run(
        muse_tools.function_map["emotive_tagger"](
            text="The canvas waits in quiet balance, ready for the first brushstroke."
        )
    )
    assert tags["tags"]
    assert isinstance(tags["dominant_emotion"], str)


def test_echo_tone_analyzer_highlights_focus() -> None:
    payload = asyncio.run(
        echo_tools.function_map["tone_analyzer"](
            messages=[
                {"speaker": "Pat", "content": "I'm frustrated by the repeated delay."},
                {"speaker": "Riley", "content": "I appreciate the patience but the risk is rising."},
            ],
            focus_topics=["delay", "risk"],
        )
    )

    assert payload["dominant_tone"] in {"concerned", "tense"}
    assert payload["focus_mentions"]["delay"] >= 1


def test_echo_reflective_prompt_deduplicates_entries() -> None:
    prompts = asyncio.run(
        echo_tools.function_map["reflective_prompt"](
            topic="release scope",
            tension_points=["timeline", "timeline"],
            tone_observation="some tension in the room",
            next_review="Friday",
        )
    )

    assert prompts["topic"] == "release scope"
    assert len(prompts["prompts"]) == prompts["count"]
    assert any("Friday" in prompt for prompt in prompts["prompts"])


def test_echo_memory_recall_prioritises_matches() -> None:
    recall = asyncio.run(
        echo_tools.function_map["memory_recall"](
            query="timeline risk",
            memories=[
                {
                    "topic": "timeline",
                    "summary": "Agreed to review launch timeline next Friday.",
                    "tags": ["timeline", "risk"],
                    "timestamp": "2024-05-01T10:00:00Z",
                },
                {
                    "topic": "budget",
                    "summary": "Discussed marketing spend alignment.",
                    "tags": ["budget"],
                    "timestamp": "2024-04-01T10:00:00Z",
                },
            ],
        )
    )

    assert recall["results"]
    assert recall["results"][0]["topic"] == "timeline"


def test_echo_conflict_resolver_builds_steps() -> None:
    plan = asyncio.run(
        echo_tools.function_map["conflict_resolver"](
            positions=[
                {
                    "participant": "Pat",
                    "statement": "We must ship by Friday",
                    "non_negotiables": ["no quality regressions"],
                },
                {
                    "participant": "Riley",
                    "statement": "We need a stable rollout",
                },
            ],
            shared_goals=["Protect customer trust"],
            decision_horizon="Friday",
        )
    )

    assert plan["participants"] == ["Pat", "Riley"]
    assert any(step["theme"] == "Risk review" for step in plan["next_steps"])
    assert "Friday" in plan["next_steps"][-1]["recommended_action"]


def test_hermes_compose_playbook_sequences_connectors() -> None:
    plan = asyncio.run(
        hermes_helpers.compose_ingestion_playbook(
            source="crm_delta",
            objectives=["sync leads", "sync leads"],
            checkpoints=["alerting"],
            stakeholders=["data-stewards"],
        )
    )
    assert plan["objectives"] == ["sync leads"]
    assert plan["sections"]["Connectors"]["steps"][0] == "api_connector"
    assert plan["sections"]["Observability"]["checkpoints"] == ["alerting"]


def test_hermes_stage_pipeline_runs_data_bridge(tmp_path) -> None:
    sample = tmp_path / "payload.csv"
    sample.write_text("id,name\n1,alpha", encoding="utf-8")

    result = asyncio.run(
        hermes_helpers.stage_pipeline(
            source="crm_delta",
            operations=[
                {"tool": "file_ingest", "params": {"path": str(sample)}},
                {"tool": "schema_infer", "params": {"records": [{"id": 1, "name": "alpha"}]}}
            ],
            dry_run=True,
        )
    )

    assert result["source"] == "crm_delta"
    assert result["steps"][0]["tool"] == "file_ingest"
    assert "fields" in result["steps"][1]["payload"]
