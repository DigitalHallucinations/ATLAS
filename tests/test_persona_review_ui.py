from __future__ import annotations

import sys
import types
from typing import Any, Dict, Optional

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub

if "jsonschema" not in sys.modules:
    jsonschema_stub = types.ModuleType("jsonschema")

    class _Validator:
        def __init__(self, _schema=None):
            self.schema = _schema

        def iter_errors(self, _payload):  # pragma: no cover - stub helper
            return []

    class _ValidationError(Exception):
        pass

    jsonschema_stub.Draft7Validator = _Validator
    jsonschema_stub.Draft202012Validator = _Validator
    jsonschema_stub.ValidationError = _ValidationError
    jsonschema_stub.exceptions = types.SimpleNamespace(
        ValidationError=_ValidationError,
        SchemaError=_ValidationError,
    )
    sys.modules["jsonschema"] = jsonschema_stub

gtk_module = sys.modules.setdefault("gi.repository.Gtk", types.ModuleType("Gtk"))
sys.modules.setdefault("Gtk", gtk_module)
pango_module = sys.modules.setdefault("gi.repository.Pango", types.ModuleType("Pango"))
sys.modules.setdefault("Pango", pango_module)

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are initialised
from gi.repository import Gtk

if not hasattr(Gtk, "Switch"):
    class _DummySwitch:
        def __init__(self, *args, **kwargs):
            self._active = False

        def set_active(self, value: bool) -> None:
            self._active = bool(value)

        def get_active(self) -> bool:
            return self._active

    Gtk.Switch = _DummySwitch

if not hasattr(Gtk, "Dialog"):
    class _DummyDialog:
        def __init__(self, *args, **kwargs):
            self.visible = True

        def destroy(self):  # pragma: no cover - stub helper
            self.visible = False

        def present(self):  # pragma: no cover - stub helper
            self.visible = True

    Gtk.Dialog = _DummyDialog

if not hasattr(Gtk, "StackTransitionType"):
    Gtk.StackTransitionType = types.SimpleNamespace(SLIDE_LEFT_RIGHT=0)
else:
    setattr(
        Gtk.StackTransitionType,
        "SLIDE_LEFT_RIGHT",
        getattr(Gtk.StackTransitionType, "SLIDE_LEFT_RIGHT", 0),
    )

class _TextBuffer:
    def __init__(self) -> None:
        self._text = ""

    def connect(self, *_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    def set_text(self, text: str, *_args, **_kwargs) -> None:  # pragma: no cover - stub helper
        self._text = text

    def get_text(self, *_args, **_kwargs) -> str:  # pragma: no cover - stub helper
        return self._text

    def get_start_iter(self, *_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    def get_end_iter(self, *_args, **_kwargs):  # pragma: no cover - stub helper
        return None


class _TextView:
    def __init__(self, *args, **kwargs):
        self._editable = True
        self._buffer = _TextBuffer()
        self._tooltip = None
        self._css_classes: set[str] = set()
        self._controllers: list[Any] = []

    def set_wrap_mode(self, *_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    def set_editable(self, value: bool) -> None:
        self._editable = bool(value)

    def set_cursor_visible(self, value: bool) -> None:  # pragma: no cover - stub helper
        self._cursor_visible = bool(value)

    def set_monospace(self, value: bool) -> None:  # pragma: no cover - stub helper
        self._monospace = bool(value)

    def set_tooltip_text(self, value: str) -> None:  # pragma: no cover - stub helper
        self._tooltip = value

    def set_size_request(self, *_args, **_kwargs):  # pragma: no cover - stub helper
        return None

    def get_buffer(self):  # pragma: no cover - stub helper
        return self._buffer

    def get_style_context(self):  # pragma: no cover - stub helper
        return self

    def add_class(self, name: str) -> None:  # pragma: no cover - stub helper
        self._css_classes.add(name)

    def add_controller(self, controller: Any) -> None:  # pragma: no cover - stub helper
        self._controllers.append(controller)
        return None


Gtk.TextView = _TextView
Gtk.TextBuffer = _TextBuffer

class _Frame:
    def __init__(self, *args, **kwargs):
        self.child = None
        self._css_classes: set[str] = set()

    def set_child(self, child) -> None:
        self.child = child

    def get_style_context(self):  # pragma: no cover - stub helper
        return self

    def add_class(self, name: str) -> None:  # pragma: no cover - stub helper
        self._css_classes.add(name)


Gtk.Frame = _Frame

from GTKUI.Persona_manager import persona_management as persona_mgmt
from GTKUI.Persona_manager.Persona_Type_Tab import persona_type_tab as persona_type_module
from GTKUI.Persona_manager.persona_management import PersonaManagement


if not hasattr(Gtk, "EventControllerFocus"):
    class _EventControllerFocus:
        def __init__(self, *args, **kwargs):  # pragma: no cover - stub helper
            self._signals: list[tuple[str, Any]] = []

        def connect(self, signal: str, callback):  # pragma: no cover - stub helper
            self._signals.append((signal, callback))
            return len(self._signals)

    Gtk.EventControllerFocus = _EventControllerFocus


class _GeneralTabStub:
    def __init__(self, persona_state: Dict[str, Any], atlas: Any) -> None:
        self.persona_state = persona_state
        self.atlas = atlas
        self._widget = Gtk.Box()

    def get_widget(self) -> Any:
        return self._widget


class _PersonaTypeTabStub:
    def __init__(self, persona_state: Dict[str, Any], general_tab: Any) -> None:
        self.persona_state = persona_state
        self.general_tab = general_tab
        self._widget = Gtk.Box()

    def get_widget(self) -> Any:
        return self._widget


persona_mgmt.GeneralTab = _GeneralTabStub
persona_mgmt.PersonaTypeTab = _PersonaTypeTabStub
persona_type_module.PersonaTypeTab = _PersonaTypeTabStub


def _persona_state() -> Dict[str, Any]:
    return {
        "original_name": "Atlas",
        "general": {
            "name": "Atlas",
            "meaning": "",
            "content": {
                "start_locked": "start",
                "editable_content": "body",
                "end_locked": "end",
            },
        },
        "provider": {"provider": "openai", "model": "gpt-4"},
        "speech": {"Speech_provider": "11labs", "voice": "jack"},
        "tools": {"allowed": [], "available": []},
    }


class _AtlasStub:
    def __init__(self, status: Dict[str, Any]) -> None:
        self.review_status = dict(status)
        self.messages: list[tuple[str, str]] = []
        self.attest_calls: list[Dict[str, Any]] = []

    def register_message_dispatcher(self, handler) -> None:  # pragma: no cover - stored
        self.dispatcher = handler

    def show_persona_message(self, role: str, message: str) -> None:
        self.messages.append((role, message))

    def get_persona_review_status(self, persona_name: str) -> Dict[str, Any]:
        status = dict(self.review_status)
        status.setdefault("persona_name", persona_name)
        return status

    def attest_persona_review(
        self,
        persona_name: str,
        *,
        reviewer: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        expires_at: Optional[str] = None,
        notes: Optional[str] = None,
        **_extra: Any,
    ) -> Dict[str, Any]:
        attestation = {
            "timestamp": "2024-05-15T00:00:00Z",
            "persona_name": persona_name,
            "reviewer": reviewer or "auditor",
            "expires_at": expires_at or "2024-08-15T00:00:00Z",
            "notes": notes or "",
        }
        self.attest_calls.append(attestation)
        self.review_status = {
            "success": True,
            "persona_name": persona_name,
            "last_change": self.review_status.get("last_change"),
            "last_review": attestation["timestamp"],
            "reviewer": reviewer,
            "expires_at": attestation["expires_at"],
            "overdue": False,
            "pending_task": False,
            "next_due": attestation["expires_at"],
            "policy_days": self.review_status.get("policy_days", 90),
            "notes": notes,
        }
        return {"success": True, "attestation": attestation, "status": dict(self.review_status)}


def test_overdue_review_shows_banner_and_button_enabled():
    status = {
        "success": True,
        "persona_name": "Atlas",
        "last_change": "2024-03-01T00:00:00Z",
        "last_review": "2024-03-15T00:00:00Z",
        "reviewer": "auditor",
        "expires_at": "2024-04-15T00:00:00Z",
        "overdue": True,
        "pending_task": False,
        "next_due": "2024-04-15T00:00:00Z",
        "policy_days": 45,
        "notes": None,
    }

    atlas = _AtlasStub(status)
    manager = PersonaManagement(atlas, Gtk.Window())
    settings_window = Gtk.Window()

    manager.show_persona_settings(_persona_state(), settings_window)

    banner = manager._review_banner_box
    assert banner is not None
    assert getattr(banner, "visible", False) is True

    label = manager._review_banner_label
    assert label is not None
    text = getattr(label, "label", "")
    assert "Review overdue" in text

    button = manager._review_mark_complete_button
    assert button is not None
    assert getattr(button, "_sensitive", True) is True


def test_pending_task_disables_review_button():
    status = {
        "success": True,
        "persona_name": "Atlas",
        "last_change": "2024-01-01T00:00:00Z",
        "last_review": None,
        "reviewer": None,
        "expires_at": "2024-02-01T00:00:00Z",
        "overdue": True,
        "pending_task": True,
        "next_due": "2024-02-01T00:00:00Z",
        "policy_days": 30,
        "notes": None,
    }

    atlas = _AtlasStub(status)
    manager = PersonaManagement(atlas, Gtk.Window())
    settings_window = Gtk.Window()

    manager.show_persona_settings(_persona_state(), settings_window)

    button = manager._review_mark_complete_button
    assert button is not None
    assert getattr(button, "_sensitive", True) is False


def test_attesting_review_updates_banner_message():
    status = {
        "success": True,
        "persona_name": "Atlas",
        "last_change": "2024-03-01T00:00:00Z",
        "last_review": None,
        "reviewer": None,
        "expires_at": "2024-04-01T00:00:00Z",
        "overdue": True,
        "pending_task": False,
        "next_due": "2024-04-01T00:00:00Z",
        "policy_days": 30,
        "notes": None,
    }

    atlas = _AtlasStub(status)
    manager = PersonaManagement(atlas, Gtk.Window())
    settings_window = Gtk.Window()
    manager.show_persona_settings(_persona_state(), settings_window)

    manager._on_mark_review_complete()

    assert atlas.attest_calls, "attest_persona_review should be invoked"
    label = manager._review_banner_label
    assert label is not None
    text = getattr(label, "label", "")
    assert "Next review due" in text
    assert any(role == "system" for role, _ in atlas.messages)
