import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

# Check if GTK4 is available - skip entire module if not
try:
    import gi
    gi.require_version('Gtk', '4.0')
    from gi.repository import Gtk, Gdk, GLib
    _GTK_AVAILABLE = True
except (ImportError, ValueError):
    _GTK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _GTK_AVAILABLE, reason="GTK4 not available")


def _get_gtk4_children(widget: "Gtk.Widget") -> List["Gtk.Widget"]:
    """Get children of a GTK4 widget using the proper API."""
    children = []
    child = widget.get_first_child()
    while child is not None:
        children.append(child)
        child = child.get_next_sibling()
    return children


if "jsonschema" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    jsonschema_stub = types.ModuleType("jsonschema")

    class _Validator:
        def __init__(self, _schema=None):
            self.schema = _schema

        def iter_errors(self, _payload):  # pragma: no cover - simplified stub
            return []

    class _ValidationError(Exception):
        pass

    jsonschema_stub.Draft7Validator = _Validator
    jsonschema_stub.Draft202012Validator = _Validator
    jsonschema_stub.ValidationError = _ValidationError
    jsonschema_stub.exceptions = types.SimpleNamespace(SchemaError=_ValidationError)
    sys.modules["jsonschema"] = jsonschema_stub

# If GTK is available, do the imports - otherwise module is skipped via pytestmark
if _GTK_AVAILABLE:
    from modules.Personas import (
        PersonaValidationError,
        load_persona_definition,
        persist_persona_definition,
    )
    from modules.logging import audit
    from modules.logging.audit import PersonaAuditLogger, SkillAuditLogger
    from modules.Server.routes import AtlasServer
    from modules.Server.conversation_routes import RequestContext
    from GTKUI.Persona_manager.persona_management import PersonaManagement


def _admin_context() -> "RequestContext":
    """Create an authenticated admin context for testing."""
    return RequestContext.from_authenticated_claims(
        tenant_id="default",
        user_id="test_admin",
        roles=("admin",),
    )


class _ConfigStub:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def get_app_root(self) -> str:
        return str(self._root)


class _UserServiceStub:
    def __init__(self, username: str) -> None:
        self._username = username

    def get_active_user(self) -> str:
        return self._username


def _write_persona_file(base: Path, persona: Dict[str, Any]) -> None:
    persona_dir = base / "modules" / "Personas" / persona["name"] / "Persona"
    persona_dir.mkdir(parents=True, exist_ok=True)
    payload = {"persona": [persona]}
    (persona_dir / f"{persona['name']}.json").write_text(
        json.dumps(payload, indent=4),
        encoding="utf-8",
    )


def _copy_schema(base: Path) -> None:
    # Schema is at project_root/modules/Personas/schema.json
    project_root = Path(__file__).resolve().parents[2]
    schema_src = project_root / "modules" / "Personas" / "schema.json"
    schema_dst = base / "modules" / "Personas" / "schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")


def _write_tool_metadata(base: Path, tool_names: Iterable[str]) -> None:
    entries = [
        {
            "name": name,
            "description": f"{name} description",
        }
        for name in tool_names
    ]
    manifest = base / "modules" / "Tools" / "tool_maps" / "functions.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _write_skill_metadata(base: Path, skills: Iterable[Dict[str, Any]]) -> None:
    manifest = base / "modules" / "Skills" / "skills.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(list(skills), indent=2), encoding="utf-8")


@pytest.fixture
def persona_payload() -> Dict[str, Any]:
    return {
        "name": "Atlas",
        "meaning": "",
        "content": {
            "start_locked": "start",
            "editable_content": "body",
            "end_locked": "end",
        },
        "provider": "openai",
        "model": "gpt-4",
        "allowed_tools": ["alpha_tool"],
        "allowed_skills": [],
        "sys_info_enabled": "False",
        "user_profile_enabled": "False",
        "type": {},
        "Speech_provider": "11labs",
        "voice": "jack",
    }


@pytest.fixture
def audit_logger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> "PersonaAuditLogger":
    logger = PersonaAuditLogger(
        log_path=tmp_path / "persona_audit.jsonl",
        user_service_factory=lambda: _UserServiceStub("auditor"),
    )
    logger.clear()
    monkeypatch.setattr(audit, "_default_logger", logger)
    yield logger
    logger.clear()


@pytest.fixture
def skill_audit_logger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> "SkillAuditLogger":
    logger = SkillAuditLogger(
        log_path=tmp_path / "skill_audit.jsonl",
        user_service_factory=lambda: _UserServiceStub("reviewer"),
    )
    logger.clear()
    monkeypatch.setattr(audit, "_default_skill_logger", logger)
    yield logger
    logger.clear()


def test_persist_persona_definition_records_audit_entry(
    tmp_path: Path,
    audit_logger: "PersonaAuditLogger",
    persona_payload: Dict[str, Any],
) -> None:
    _copy_schema(tmp_path)
    config = _ConfigStub(tmp_path)

    original = dict(persona_payload)
    _write_persona_file(tmp_path, original)

    updated = dict(persona_payload)
    updated["allowed_tools"] = ["beta_tool"]

    persist_persona_definition(
        "Atlas",
        updated,
        config_manager=config,
        rationale="Unit test update",
    )

    entries, total = audit_logger.get_history(persona_name="Atlas")
    assert total == 1
    entry = entries[0]
    assert entry.persona_name == "Atlas"
    assert entry.username == "auditor"
    assert entry.old_tools == ["alpha_tool"]
    assert entry.new_tools == ["beta_tool"]
    assert entry.rationale == "Unit test update"


def test_server_route_logs_persona_update(
    tmp_path: Path,
    audit_logger: "PersonaAuditLogger",
    persona_payload: Dict[str, Any],
) -> None:
    _copy_schema(tmp_path)
    _write_tool_metadata(tmp_path, ["alpha_tool", "beta_tool", "gamma_tool"])
    config = _ConfigStub(tmp_path)

    _write_persona_file(tmp_path, dict(persona_payload))

    server = AtlasServer(config_manager=config)
    response = server.handle_request(
        "/personas/Atlas/tools",
        method="POST",
        query={"tools": ["beta_tool", "gamma_tool"], "rationale": "Server update"},
        context=_admin_context(),
    )

    assert response["success"] is True
    assert response["persona"]["allowed_tools"] == ["beta_tool", "gamma_tool"]

    entries, total = audit_logger.get_history(persona_name="Atlas")
    assert total == 1
    entry = entries[0]
    assert entry.new_tools == ["beta_tool", "gamma_tool"]
    assert entry.rationale == "Server update"


def test_server_route_updates_persona_skills(
    tmp_path: Path,
    persona_payload: Dict[str, Any],
) -> None:
    _copy_schema(tmp_path)
    _write_tool_metadata(tmp_path, ["alpha_tool", "beta_tool"])
    _write_skill_metadata(
        tmp_path,
        [
            {
                "name": "shared_skill",
                "version": "1.0",
                "instruction_prompt": "Shared instructions",
                "required_tools": ["alpha_tool"],
                "required_capabilities": ["analysis"],
                "safety_notes": "Review output.",
            },
                {
                    "name": "atlas_insight",
                    "version": "1.0",
                    "instruction_prompt": "Atlas-only insight",
                    "required_tools": ["beta_tool"],
                    "required_capabilities": ["expertise"],
                    "safety_notes": "Use responsibly.",
                },
            ],
        )
    config = _ConfigStub(tmp_path)

    persona_data = dict(persona_payload)
    persona_data["allowed_skills"] = []
    persona_data["allowed_tools"] = ["alpha_tool", "beta_tool"]
    _write_persona_file(tmp_path, persona_data)

    server = AtlasServer(config_manager=config)
    response = server.handle_request(
        "/personas/Atlas/skills",
        method="POST",
        query={
            "skills": ["shared_skill", "atlas_insight"],
            "rationale": "Server skill update",
        },
        context=_admin_context(),
    )

    assert response["success"] is True
    assert response["persona"]["allowed_skills"] == [
        "atlas_insight",
        "shared_skill",
    ]

    persisted = load_persona_definition("Atlas", config_manager=config)
    assert persisted is not None
    assert persisted.get("allowed_skills") == ["atlas_insight", "shared_skill"]


def test_list_tool_changes_supports_filters(
    audit_logger: "PersonaAuditLogger",
) -> None:
    audit_logger.clear()
    audit_logger.record_change(
        "Atlas",
        ["alpha_tool"],
        ["beta_tool"],
        username="alice",
        timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    audit_logger.record_change(
        "Zephyr",
        ["delta_tool"],
        ["epsilon_tool"],
        username="zeus",
        timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )
    audit_logger.record_change(
        "Atlas",
        ["beta_tool"],
        ["gamma_tool"],
        username="bob",
        timestamp=datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc),
    )

    server = AtlasServer()

    filtered = server.list_tool_changes(persona="Atlas", limit=5)
    assert filtered["total"] == 2
    assert filtered["count"] == 2
    authors = [entry["author"] for entry in filtered["changes"]]
    assert authors == ["bob", "alice"]
    first_entry = filtered["changes"][0]
    assert first_entry["persona"] == "Atlas"
    assert first_entry["added"] == ["gamma_tool"]
    assert first_entry["removed"] == ["beta_tool"]
    assert first_entry["summary"] == "Tool configuration updated."
    assert first_entry["timestamp"].endswith("Z")

    paginated = server.list_tool_changes(limit=1, offset=1)
    assert paginated["count"] == 1
    assert paginated["total"] == 3
    assert paginated["changes"][0]["persona"] == "Zephyr"

    empty = server.list_tool_changes(limit=0)
    assert empty["limit"] == 0
    assert empty["total"] == 3
    assert empty["changes"] == []


def test_list_skill_changes_supports_filters(
    skill_audit_logger: "SkillAuditLogger",
) -> None:
    skill_audit_logger.clear()
    skill_audit_logger.record_change(
        "Atlas",
        "insight",
        old_review_status="pending",
        new_review_status="reviewed",
        old_tester_notes="",
        new_tester_notes="Looks good",
        username="reviewer1",
        timestamp=datetime(2024, 2, 1, 9, 0, tzinfo=timezone.utc),
    )
    skill_audit_logger.record_change(
        "Zephyr",
        "calibrate",
        old_review_status="unreviewed",
        new_review_status="reviewed",
        old_tester_notes="",
        new_tester_notes="Escalate edge cases",
        username="reviewer2",
        timestamp=datetime(2024, 2, 2, 9, 0, tzinfo=timezone.utc),
    )
    skill_audit_logger.record_change(
        "Atlas",
        "insight",
        old_review_status="reviewed",
        new_review_status="reviewed",
        old_tester_notes="Looks good",
        new_tester_notes="Updated tester notes",
        username="reviewer3",
        timestamp=datetime(2024, 2, 3, 9, 0, tzinfo=timezone.utc),
    )

    server = AtlasServer()

    filtered = server.list_skill_changes(persona="Atlas", limit=10)
    assert filtered["total"] == 2
    assert filtered["count"] == 2
    first_entry = filtered["changes"][0]
    assert first_entry["skill"] == "insight"
    assert first_entry["author"] == "reviewer3"
    assert first_entry["details"]["tester_notes"]["changed"] is True
    assert first_entry["details"]["review_status"]["changed"] is False
    assert first_entry["summary"]
    assert first_entry["timestamp"].endswith("Z")

    skill_filtered = server.list_skill_changes(skill="calibrate", limit=5)
    assert skill_filtered["total"] == 1
    assert skill_filtered["changes"][0]["persona"] == "Zephyr"

    empty = server.list_skill_changes(limit=0)
    assert empty["limit"] == 0
    assert empty["total"] == 3
    assert empty["changes"] == []


def test_server_route_rejects_invalid_tool_update(
    tmp_path: Path,
    audit_logger: "PersonaAuditLogger",
    persona_payload: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_schema(tmp_path)
    _write_tool_metadata(tmp_path, ["alpha_tool", "beta_tool"])
    config = _ConfigStub(tmp_path)

    _write_persona_file(tmp_path, dict(persona_payload))

    def _fake_validate(
        payload,
        *,
        persona_name: str,
        tool_ids,
        skill_ids=None,
        skill_catalog=None,
        config_manager=None,
    ):
        personas = payload.get("persona") if isinstance(payload, dict) else None
        if not personas:
            return
        allowed = personas[0].get("allowed_tools") or []
        invalid = [name for name in allowed if name not in set(tool_ids)]
        if invalid:
            raise PersonaValidationError(
                "Persona '{name}' failed schema validation: invalid tools {tools}".format(
                    name=persona_name,
                    tools=", ".join(invalid),
                )
            )

    monkeypatch.setattr(
        "modules.Server.routes._validate_persona_payload",
        _fake_validate,
    )

    server = AtlasServer(config_manager=config)
    response = server.handle_request(
        "/personas/Atlas/tools",
        method="POST",
        query={"tools": ["alpha_tool", "invalid_tool"], "rationale": "Bad update"},
        context=_admin_context(),
    )

    assert response["success"] is False
    error_text = response.get("error", "")
    assert "invalid_tool" in error_text
    assert "failed schema validation" in error_text
    assert response.get("errors") == [error_text]

    persona_path = (
        tmp_path / "modules" / "Personas" / "Atlas" / "Persona" / "Atlas.json"
    )
    saved = json.loads(persona_path.read_text(encoding="utf-8"))
    tools = saved["persona"][0].get("allowed_tools")
    assert tools == ["alpha_tool"]


class _AtlasStub:
    def __init__(self) -> None:
        self.messages = []
        self.review_status = {
            "success": True,
            "persona_name": "",
            "last_change": None,
            "last_review": None,
            "reviewer": None,
            "expires_at": None,
            "overdue": False,
            "pending_task": False,
            "next_due": None,
            "policy_days": 90,
            "notes": None,
        }
        self.review_attestations: list[tuple[str, Dict[str, Any]]] = []
        self.audit_history: list[Dict[str, Any]] = []

    def register_message_dispatcher(self, handler) -> None:  # pragma: no cover - stored for completeness
        self.dispatcher = handler

    def get_persona_names(self):  # pragma: no cover - used by other UI paths
        return ["Atlas"]

    def show_persona_message(self, role: str, message: str) -> None:
        self.messages.append((role, message))

    def get_persona_review_status(self, persona_name: str) -> Dict[str, Any]:
        status = dict(self.review_status)
        status["persona_name"] = persona_name
        return status

    def attest_persona_review(
        self,
        persona_name: str,
        *,
        reviewer: str,
        expires_in_days: Optional[int] = None,
        expires_at: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        attestation = {
            "timestamp": "2024-01-01T00:00:00Z",
            "persona_name": persona_name,
            "reviewer": reviewer,
            "expires_at": expires_at or "2024-04-01T00:00:00Z",
            "notes": notes or "",
        }
        self.review_attestations.append((persona_name, attestation))
        status = dict(self.review_status)
        status.update(
            {
                "persona_name": persona_name,
                "last_review": attestation["timestamp"],
                "reviewer": reviewer,
                "expires_at": attestation["expires_at"],
                "overdue": False,
                "pending_task": False,
                "next_due": attestation["expires_at"],
            }
        )
        self.review_status = status
        return {"success": True, "attestation": attestation, "status": status}

    def get_persona_audit_history(
        self,
        _persona_name: str,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Dict[str, Any]], int]:
        try:
            offset_value = int(offset)
        except (TypeError, ValueError):
            offset_value = 0
        if offset_value < 0:
            offset_value = 0

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 20
        if limit_value <= 0:
            return [], len(self.audit_history)

        window = self.audit_history[offset_value : offset_value + limit_value]
        return [dict(entry) for entry in window], len(self.audit_history)


def test_history_view_renders_entries(
    audit_logger: "PersonaAuditLogger",
) -> None:
    audit_logger.clear()
    audit_logger.record_change(
        "Atlas",
        ["alpha"],
        ["beta"],
        rationale="Initial change",
        username="auditor",
        timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )
    audit_logger.record_change(
        "Atlas",
        ["beta"],
        ["gamma"],
        rationale="Follow-up change",
        username="auditor",
        timestamp=datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc),
    )

    atlas = _AtlasStub()
    entries, _total = audit_logger.get_history(persona_name="Atlas")
    atlas.audit_history = [
        {
            "timestamp": entry.timestamp,
            "persona_name": entry.persona_name,
            "username": entry.username,
            "old_tools": list(entry.old_tools),
            "new_tools": list(entry.new_tools),
            "rationale": entry.rationale,
        }
        for entry in entries
    ]
    parent_window = Gtk.Window()
    manager = PersonaManagement(atlas, parent_window)

    history_tab = manager.create_history_tab({"general": {"name": "Atlas"}})
    assert isinstance(history_tab, Gtk.Box)

    list_box = manager.history_list_box
    assert list_box is not None
    rows = _get_gtk4_children(list_box)
    assert len(rows) == 2

    first_row = rows[0]
    # In GTK4, use get_child() to get the child of a ListBoxRow
    container = first_row.get_child()
    children = _get_gtk4_children(container)
    summary_label = children[0]
    tools_label = children[1]
    rationale_label = children[2]

    assert summary_label.get_label() == "2024-01-02 12:00 UTC — auditor"
    assert tools_label.get_label() == "Tools: beta → gamma"
    assert rationale_label.get_label() == "Rationale: Follow-up change"

    assert manager._history_load_more_button is not None
    assert manager._history_load_more_button.get_visible() is False
