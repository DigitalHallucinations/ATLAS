import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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

import pytest

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are available
sys.modules.setdefault("gi.repository.Pango", types.ModuleType("Pango"))
sys.modules.setdefault("Pango", sys.modules["gi.repository.Pango"])
from gi.repository import Gtk

if not hasattr(Gtk, "Switch"):
    class _DummySwitch:
        def __init__(self, *args, **kwargs):
            self._active = False

        def set_active(self, value: bool) -> None:
            self._active = bool(value)

        def get_active(self) -> bool:  # pragma: no cover - compatibility helper
            return self._active

    Gtk.Switch = _DummySwitch

if not hasattr(Gtk, "Dialog"):
    class _DummyDialog:
        def __init__(self, *args, **kwargs):
            self.visible = True

        def close(self):  # pragma: no cover - compatibility helper
            self.visible = False

    Gtk.Dialog = _DummyDialog

if not hasattr(Gtk, "SelectionMode"):
    Gtk.SelectionMode = types.SimpleNamespace(NONE=0, SINGLE=1, MULTIPLE=2)

if not hasattr(Gtk, "ListBoxRow"):
    class _DummyListBoxRow:
        def __init__(self, *args, **kwargs):
            self.children = []

        def set_child(self, child):
            self.children = [child]

    Gtk.ListBoxRow = _DummyListBoxRow

Gtk.ListBoxRow = type(
    "ListBoxRow",
    (),
    {
        "__init__": lambda self, *args, **kwargs: setattr(self, "children", []),
        "set_child": lambda self, child: setattr(self, "children", [child]),
    },
)

from modules.Personas import persist_persona_definition
from modules.logging import audit
from modules.logging.audit import PersonaAuditLogger
from modules.Server.routes import AtlasServer
from GTKUI.Persona_manager.persona_management import PersonaManagement


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
    schema_src = Path(__file__).resolve().parents[1] / "modules" / "Personas" / "schema.json"
    schema_dst = base / "modules" / "Personas" / "schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")


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
        "sys_info_enabled": "False",
        "user_profile_enabled": "False",
        "type": {},
        "Speech_provider": "11labs",
        "voice": "jack",
    }


@pytest.fixture
def audit_logger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PersonaAuditLogger:
    logger = PersonaAuditLogger(
        log_path=tmp_path / "persona_audit.jsonl",
        user_service_factory=lambda: _UserServiceStub("auditor"),
    )
    logger.clear()
    monkeypatch.setattr(audit, "_default_logger", logger)
    yield logger
    logger.clear()


def test_persist_persona_definition_records_audit_entry(
    tmp_path: Path,
    audit_logger: PersonaAuditLogger,
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
    audit_logger: PersonaAuditLogger,
    persona_payload: Dict[str, Any],
) -> None:
    _copy_schema(tmp_path)
    config = _ConfigStub(tmp_path)

    _write_persona_file(tmp_path, dict(persona_payload))

    server = AtlasServer(config_manager=config)
    response = server.handle_request(
        "/personas/Atlas/tools",
        method="POST",
        query={"tools": ["beta_tool", "gamma_tool"], "rationale": "Server update"},
    )

    assert response["success"] is True
    assert response["persona"]["allowed_tools"] == ["beta_tool", "gamma_tool"]

    entries, total = audit_logger.get_history(persona_name="Atlas")
    assert total == 1
    entry = entries[0]
    assert entry.new_tools == ["beta_tool", "gamma_tool"]
    assert entry.rationale == "Server update"


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


def test_history_view_renders_entries(
    audit_logger: PersonaAuditLogger,
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
    parent_window = Gtk.Window()
    manager = PersonaManagement(atlas, parent_window)

    history_tab = manager.create_history_tab({"general": {"name": "Atlas"}})
    assert isinstance(history_tab, Gtk.Box)

    list_box = manager.history_list_box
    assert list_box is not None
    rows = list(getattr(list_box, "children", []))
    assert len(rows) == 2

    first_row = rows[0]
    container = first_row.children[0]
    summary_label = container.children[0]
    tools_label = container.children[1]
    rationale_label = container.children[2]

    assert summary_label.label == "2024-01-02 12:00 UTC — auditor"
    assert tools_label.label == "Tools: beta → gamma"
    assert rationale_label.label == "Rationale: Follow-up change"

    assert manager._history_load_more_button is not None
    assert manager._history_load_more_button.visible is False
