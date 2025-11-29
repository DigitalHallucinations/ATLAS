"""Tests for modules.Server.routes helper functions."""

from __future__ import annotations

import base64
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional

import pytest


class _DummyValidator:
    def __init__(self, _schema):
        self.schema = _schema

    def iter_errors(self, _payload):
        return []


if "jsonschema" not in sys.modules:
    sys.modules["jsonschema"] = SimpleNamespace(
        Draft202012Validator=_DummyValidator,
        exceptions=SimpleNamespace(SchemaError=Exception),
    )

if "yaml" not in sys.modules:
    sys.modules["yaml"] = SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: {},
        dump=lambda *_args, **_kwargs: None,
    )

from types import MappingProxyType

from modules.Server.conversation_routes import RequestContext
from modules.Server.routes import _as_bool, _parse_query_timestamp, AtlasServer
from modules.Tools.manifest_loader import ToolManifestEntry
from modules.Skills.manifest_loader import SkillMetadata
from modules.orchestration.capability_registry import ToolCapabilityView, SkillCapabilityView


def _make_entry(name: str, *, persona: Optional[str]) -> ToolManifestEntry:
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


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        (
            "2024-05-22T10:00:00Z",
            datetime(2024, 5, 22, 10, 0, tzinfo=timezone.utc),
        ),
        (
            "2024-05-22T10:00:00",
            datetime(2024, 5, 22, 10, 0, tzinfo=timezone.utc),
        ),
        ("not-a-timestamp", None),
    ],
)
def test_parse_query_timestamp_variants(raw_value, expected):
    parsed = _parse_query_timestamp(raw_value)
    assert parsed == expected


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (None, False),
        ("", False),
        ("true", True),
        ("ON", True),
        ("false", False),
        (False, False),
        (True, True),
    ],
)
def test_as_bool_variants(raw_value, expected):
    assert _as_bool(raw_value) is expected


class _RegistryStub:
    def __init__(self, tools: List[ToolCapabilityView], skills: List[SkillCapabilityView]):
        self.tools = tools
        self.skills = skills

    def query_tools(self, **_filters):
        return list(self.tools)

    def query_skills(self, **filters):
        persona_filters = [token.lower() for token in filters.get("persona_filters") or []]
        if not persona_filters:
            return list(self.skills)

        exclude_shared = any(token == "-shared" for token in persona_filters)
        positive = [token for token in persona_filters if not token.startswith("-")]

        results: List[SkillCapabilityView] = []
        for view in self.skills:
            persona = view.manifest.persona or "shared"
            persona_token = persona.lower()
            if exclude_shared and persona_token == "shared":
                continue
            if positive and persona_token not in positive:
                continue
            results.append(view)
        return results


class _DlpEnforcerStub:
    def __init__(self, sanitized_payload: Mapping[str, Any]):
        self.sanitized_payload = dict(sanitized_payload)
        self.calls: list[tuple[Mapping[str, Any], Optional[str]]] = []

    def apply_to_payload(self, payload: Mapping[str, Any], tenant_id: Optional[str]):
        self.calls.append((payload, tenant_id))
        return self.sanitized_payload


class _ConversationRoutesStub:
    def __init__(self) -> None:
        self.created: list[tuple[Mapping[str, Any], RequestContext]] = []
        self.updated: list[tuple[str, Mapping[str, Any], RequestContext]] = []

    def create_message(self, payload: Mapping[str, Any], *, context: RequestContext):
        self.created.append((payload, context))
        return {"action": "create", "payload": payload, "context": context}

    def update_message(self, message_id: str, payload: Mapping[str, Any], *, context: RequestContext):
        self.updated.append((message_id, payload, context))
        return {"action": "update", "id": message_id, "payload": payload, "context": context}


class _TaskRoutesStub:
    def __init__(self) -> None:
        self.created: list[tuple[Mapping[str, Any], RequestContext]] = []
        self.updated: list[tuple[str, Mapping[str, Any], RequestContext]] = []

    def create_task(self, payload: Mapping[str, Any], *, context: RequestContext):
        self.created.append((payload, context))
        return payload

    def update_task(self, task_id: str, payload: Mapping[str, Any], *, context: RequestContext):
        self.updated.append((task_id, payload, context))
        return payload


class _JobRoutesStub:
    def __init__(self) -> None:
        self.created: list[tuple[Mapping[str, Any], RequestContext]] = []
        self.updated: list[tuple[str, Mapping[str, Any], RequestContext]] = []

    def create_job(self, payload: Mapping[str, Any], *, context: RequestContext):
        self.created.append((payload, context))
        return payload

    def update_job(self, job_id: str, payload: Mapping[str, Any], *, context: RequestContext):
        self.updated.append((job_id, payload, context))
        return payload


def test_get_tools_includes_shared_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_entry = _make_entry("shared_tool", persona=None)
    atlas_entry = _make_entry("atlas_only", persona="Atlas")
    other_entry = _make_entry("other_tool", persona="Other")

    registry = _RegistryStub(
        [_make_tool_view(shared_entry), _make_tool_view(atlas_entry), _make_tool_view(other_entry)],
        [],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_tools(persona="Atlas")

    assert response["count"] == 2
    names = {tool["name"] for tool in response["tools"]}
    assert names == {"shared_tool", "atlas_only"}


def test_get_tools_can_exclude_shared_persona(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_entry = _make_entry("shared_tool", persona=None)
    atlas_entry = _make_entry("atlas_only", persona="Atlas")

    registry = _RegistryStub(
        [_make_tool_view(shared_entry), _make_tool_view(atlas_entry)],
        [],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_tools(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    names = {tool["name"] for tool in response["tools"]}
    assert names == {"atlas_only"}


def test_get_skills_returns_serialized_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    server = AtlasServer()
    shared_skill = SkillMetadata(
        name="summarize",
        version="1.0.0",
        instruction_prompt="Do summary",
        required_tools=["web_search"],
        required_capabilities=["summaries"],
        safety_notes="",
        summary="Quick summary",
        category="Context",
        capability_tags=["summaries"],
        persona=None,
        source="tests",
        collaboration=None,
    )
    atlas_skill = SkillMetadata(
        name="atlas_only",
        version="2.0.0",
        instruction_prompt="Do atlas",
        required_tools=[],
        required_capabilities=[],
        safety_notes="",
        summary="Atlas details",
        category="Atlas",
        capability_tags=[],
        persona="Atlas",
        source="tests",
        collaboration=None,
    )

    registry = _RegistryStub(
        [_make_tool_view(_make_entry("shared_tool", persona=None))],
        [_make_skill_view(shared_skill), _make_skill_view(atlas_skill)],
    )
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    response = server.get_skills(persona=["Atlas", "-shared"])

    assert response["count"] == 1
    assert response["skills"][0]["name"] == "atlas_only"
    assert response["skills"][0]["summary"] == "Atlas details"

    routed = server.handle_request("/skills", method="GET")
    assert routed["count"] == 2
    assert {entry["category"] for entry in routed["skills"]} == {"Context", "Atlas"}


def test_handle_request_skills_details_preserves_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = AtlasServer()
    skill = SkillMetadata(
        name="custom_skill",
        version="1.0.0",
        instruction_prompt="Do the custom task",
        required_tools=[],
        required_capabilities=[],
        safety_notes="",
        summary="Custom skill summary",
        category="Custom",
        capability_tags=[],
        persona=None,
        source="tests",
        collaboration=None,
    )

    registry = _RegistryStub([], [_make_skill_view(skill)])
    monkeypatch.setattr(
        "modules.Server.routes.get_capability_registry",
        lambda config_manager=None: registry,
    )

    context = RequestContext(tenant_id="tenant-123", user_id="user-456")

    response = server.handle_request(
        "/skills/custom_skill",
        method="GET",
        context=context,
    )

    assert response["success"] is True
    assert response["skill"]["name"] == "custom_skill"
    assert response["tenant_id"] == context.tenant_id


def test_create_message_applies_dlp_and_forwards_sanitized_payload() -> None:
    server = AtlasServer()
    sanitized_payload = {"text": "clean"}
    dlp_enforcer = _DlpEnforcerStub(sanitized_payload)
    conversation_routes = _ConversationRoutesStub()

    server._dlp_enforcer = dlp_enforcer
    server._get_conversation_routes = lambda: conversation_routes

    raw_payload = {"text": "raw secret"}
    response = server.create_message(raw_payload, context={"tenant_id": "tenant-007"})

    expected_context = RequestContext(
        tenant_id="tenant-007",
        user_id=None,
        session_id=None,
        roles=(),
        metadata=None,
    )

    assert dlp_enforcer.calls == [(raw_payload, "tenant-007")]
    assert conversation_routes.created == [(sanitized_payload, expected_context)]
    assert response["payload"] == sanitized_payload
    assert response["context"] == expected_context


def test_update_message_applies_dlp_and_forwards_sanitized_payload() -> None:
    server = AtlasServer()
    sanitized_payload = {"content": "cleaned"}
    dlp_enforcer = _DlpEnforcerStub(sanitized_payload)
    conversation_routes = _ConversationRoutesStub()

    server._dlp_enforcer = dlp_enforcer
    server._get_conversation_routes = lambda: conversation_routes

    raw_payload = {"content": "raw"}
    request_context = RequestContext.from_authenticated_claims(
        tenant_id="tenant-abc", roles=("member",)
    )

    response = server.update_message(
        "message-1",
        raw_payload,
        context=request_context,
    )

    expected_context = request_context

    assert dlp_enforcer.calls == [(raw_payload, "tenant-abc")]
    assert conversation_routes.updated == [("message-1", sanitized_payload, expected_context)]
    assert response["id"] == "message-1"
    assert response["payload"] == sanitized_payload
    assert response["context"] == expected_context


def test_create_task_redacts_pii_before_forwarding() -> None:
    config = SimpleNamespace(get_dlp_policy=lambda _tenant: {"enabled": True})
    server = AtlasServer(config_manager=config)
    task_routes = _TaskRoutesStub()
    server._require_task_routes = lambda: task_routes

    raw_payload = {"summary": "Email user@example.com"}
    response = server.create_task(raw_payload, context={"tenant_id": "tenant-1"})

    assert task_routes.created[0][0]["summary"] == "Email <redacted>"
    assert response["summary"] == "Email <redacted>"


def test_update_job_redacts_pii_before_forwarding() -> None:
    config = SimpleNamespace(get_dlp_policy=lambda _tenant: {"enabled": True})
    server = AtlasServer(config_manager=config)
    job_routes = _JobRoutesStub()
    server._require_job_routes = lambda: job_routes

    raw_payload = {"notes": "Call 123-45-6789"}
    context = RequestContext.from_authenticated_claims(tenant_id="tenant-9", roles=("admin",))

    response = server.update_job("job-007", raw_payload, context=context)

    assert job_routes.updated[0][0] == "job-007"
    assert job_routes.updated[0][1]["notes"] == "Call <redacted>"
    assert response["notes"] == "Call <redacted>"


def test_update_persona_tools_sanitizes_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(get_dlp_policy=lambda _tenant: {"enabled": True})
    server = AtlasServer(config_manager=config)

    monkeypatch.setattr(
        "modules.Server.routes.load_persona_definition",
        lambda persona_name, config_manager=None: {"name": persona_name, "allowed_tools": []},
    )
    monkeypatch.setattr(
        "modules.Server.routes.load_tool_metadata",
        lambda config_manager=None: (["search"], {"search": {}}),
    )
    monkeypatch.setattr(
        "modules.Server.routes.load_skill_catalog",
        lambda config_manager=None: ([], {}),
    )
    monkeypatch.setattr(
        "modules.Server.routes.normalize_allowed_tools", lambda tools, metadata_order=None: list(tools)
    )
    monkeypatch.setattr(
        "modules.Server.routes._validate_persona_payload", lambda *args, **kwargs: None
    )

    persisted: dict[str, Any] = {}

    def _persist(name: str, payload: Mapping[str, Any], *, config_manager=None, rationale=None):
        persisted["name"] = name
        persisted["payload"] = payload
        persisted["rationale"] = rationale

    monkeypatch.setattr("modules.Server.routes.persist_persona_definition", _persist)

    response = server.update_persona_tools(
        "Example",
        tools=["ssn 123-45-6789"],
        rationale="See 123-45-6789",
        context=RequestContext.from_authenticated_claims(
            tenant_id="tenant-2", roles=("admin",)
        ),
    )

    assert persisted["payload"]["allowed_tools"] == ["<redacted>"]
    assert persisted["rationale"] == "See <redacted>"
    assert response["persona"]["allowed_tools"] == ["<redacted>"]


def test_set_skill_metadata_redacts_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_calls: list[Mapping[str, Any]] = []
    persisted_metadata: dict[str, Any] = {}

    def _set_skill_metadata(name: str, payload: Mapping[str, Any]) -> None:
        metadata_calls.append(payload)
        persisted_metadata.update(payload)

    def _get_skill_metadata(_name: str) -> Mapping[str, Any]:
        return dict(persisted_metadata)

    config = SimpleNamespace(
        get_dlp_policy=lambda _tenant: {"enabled": True},
        set_skill_metadata=_set_skill_metadata,
        get_skill_metadata=_get_skill_metadata,
    )

    server = AtlasServer(config_manager=config)
    server._find_skill_view = (
        lambda identifier, persona_token=None: SimpleNamespace(
            manifest=SimpleNamespace(persona=None)
        )
    )
    monkeypatch.setattr("modules.Server.routes._serialize_skill", lambda _view: {"name": "demo"})
    monkeypatch.setattr(
        "modules.Server.routes.get_skill_audit_logger",
        lambda: SimpleNamespace(record_change=lambda *args, **kwargs: None),
    )

    response = server.set_skill_metadata(
        skill_name="demo",
        metadata={"tester_notes": "Reach me at user@example.com"},
        context={"tenant_id": "tenant-x"},
    )

    assert metadata_calls[0]["tester_notes"] == "Reach me at <redacted>"
    assert response["metadata"]["tester_notes"] == "Reach me at <redacted>"


def test_import_tool_bundle_redacts_rationale(monkeypatch: pytest.MonkeyPatch) -> None:
    config = SimpleNamespace(get_dlp_policy=lambda _tenant: {"enabled": True})
    server = AtlasServer(config_manager=config)
    server._resolve_signing_key = lambda *_args, **_kwargs: "signing-key"
    server._parse_bundle_metadata = lambda _data: {}
    server._resolve_resource_tenant_from_metadata = lambda _metadata: None
    server._authorize_mutation = lambda *args, **kwargs: None

    captured: dict[str, Any] = {}

    def _import_tool_bundle_bytes(bundle_bytes, *, signing_key, config_manager, rationale):
        captured["rationale"] = rationale
        return {"imported": True}

    monkeypatch.setattr(
        "modules.Server.routes.import_tool_bundle_bytes", _import_tool_bundle_bytes
    )

    bundle = base64.b64encode(b"{}").decode("ascii")
    response = server.import_tool_bundle(
        bundle_base64=bundle,
        rationale="Contact user@example.com",
        context={"tenant_id": "tenant-import"},
    )

    assert captured["rationale"] == "Contact <redacted>"
    assert response["success"] is True
