from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from modules.orchestration import job_manager
from modules.orchestration.capability_registry import (
    _job_persona_matches,
    _persona_filter_matches,
)
from modules.orchestration.utils import (
    normalize_persona_identifier,
    persona_matches_filter,
)


@dataclass
class DummyTaskMetadata:
    name: str
    summary: str = "summary"
    description: str = "description"
    required_skills: Tuple[str, ...] = ()
    required_tools: Tuple[str, ...] = ()
    acceptance_criteria: Tuple[str, ...] = ()
    escalation_policy: Mapping[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()
    priority: str = ""
    persona: Optional[str] = None
    source: str = "source"


def test_normalize_persona_identifier_lowercases_inputs() -> None:
    assert normalize_persona_identifier(" Atlas ") == "atlas"
    assert normalize_persona_identifier("SHARED") == "shared"
    assert normalize_persona_identifier(None) is None
    assert normalize_persona_identifier("") is None


def test_capability_registry_persona_filters_use_normalized_keys() -> None:
    persona_key = normalize_persona_identifier("Atlas")
    tokens: Sequence[str] = ("atlas",)

    assert persona_key == "atlas"
    assert _persona_filter_matches(persona_key, tokens)
    assert _job_persona_matches(["Atlas", "Beta"], tokens)
    assert not _job_persona_matches(["Beta"], tokens)


def test_persona_matches_filter_shared_tokens() -> None:
    tokens: Sequence[str] = ("atlas", "-shared")

    assert persona_matches_filter("Atlas", tokens)
    assert not persona_matches_filter(None, tokens)
    assert not persona_matches_filter("shared", tokens)


def test_persona_matches_filter_allows_non_shared_when_excluding_shared() -> None:
    tokens: Sequence[str] = ("-shared",)

    assert persona_matches_filter("Atlas", tokens)
    assert persona_matches_filter("Beta", tokens)
    assert persona_matches_filter("atlas", tokens)
    assert persona_matches_filter("beta", tokens)
    assert not persona_matches_filter(None, tokens)


def test_job_manager_task_resolver_normalizes_persona_keys(monkeypatch) -> None:
    persona_task = DummyTaskMetadata(name="TaskAlpha", persona="Atlas")
    shared_task = DummyTaskMetadata(name="TaskShared", persona=None)

    def fake_loader(*, config_manager=None):  # type: ignore[override]
        return [persona_task, shared_task]

    monkeypatch.setattr(job_manager.task_manifest_loader, "load_task_metadata", fake_loader)

    resolver = job_manager.build_task_manifest_resolver()

    manifest_upper = resolver("TaskAlpha", persona="ATLAS")
    manifest_lower = resolver("TaskAlpha", persona="atlas")
    manifest_mixed = resolver("TaskAlpha", persona="AtLaS")

    assert manifest_upper is not None
    assert manifest_lower is not None
    assert manifest_mixed is not None
    assert manifest_upper["name"] == manifest_lower["name"] == manifest_mixed["name"] == "TaskAlpha"

    shared_manifest = resolver("TaskShared", persona="AtLaS")
    assert shared_manifest is not None
    assert shared_manifest["name"] == "TaskShared"
