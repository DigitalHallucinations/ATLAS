from __future__ import annotations

import sys
import types
from typing import Any, Dict, Optional

if "jsonschema" not in sys.modules:  # pragma: no cover - lightweight stub for tests
    jsonschema_stub = types.ModuleType("jsonschema")

    class _Validator:
        def __init__(self, _schema=None):
            self.schema = _schema

        def validate(self, _payload):
            return None

    class _ValidationError(Exception):
        pass

    jsonschema_stub.Draft7Validator = _Validator
    jsonschema_stub.Draft202012Validator = _Validator
    jsonschema_stub.ValidationError = _ValidationError
    sys.modules["jsonschema"] = jsonschema_stub

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are active
_chat_helper = sys.modules["tests.test_chat_async_helper"]
pango_module = sys.modules.setdefault("gi.repository.Pango", types.ModuleType("Pango"))
sys.modules.setdefault("Pango", pango_module)
gtk_module = sys.modules.setdefault("gi.repository.Gtk", types.ModuleType("Gtk"))
sys.modules.setdefault("Gtk", gtk_module)
class _DummyWidget:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):  # pragma: no cover
        return self

    def __getattr__(self, _name):  # pragma: no cover
        return lambda *_a, **_kw: None

if not hasattr(gtk_module, "__getattr__"):
    def _gtk_getattr(name):  # pragma: no cover
        cls = type(name, (_DummyWidget,), {})
        setattr(gtk_module, name, cls)
        return cls

    gtk_module.__getattr__ = _gtk_getattr
if not hasattr(gtk_module, "Switch"):
    class _DummySwitch:
        def __init__(self, *args, **kwargs):
            pass

        def set_active(self, *_args, **_kwargs):
            return None

    gtk_module.Switch = _DummySwitch
if not hasattr(gtk_module, "Dialog"):
    gtk_module.Dialog = type("Dialog", (_DummyWidget,), {})
gtk_module.SelectionMode = types.SimpleNamespace(NONE=0, SINGLE=1, MULTIPLE=2)
gtk_module.Widget = _chat_helper._DummyWidget
if not hasattr(gtk_module, "Align"):
    gtk_module.Align = types.SimpleNamespace(FILL=0, START=1, END=2, CENTER=3)
from gi.repository import Gtk

if not hasattr(Gtk.Align, "FILL"):
    setattr(Gtk.Align, "FILL", 0)

if not hasattr(Gtk, "Notebook"):
    Gtk.Notebook = type("Notebook", (_chat_helper._DummyWidget,), {})
if not hasattr(Gtk.Notebook, "set_tab_pos"):
    Gtk.Notebook.set_tab_pos = lambda self, *args, **kwargs: None
if not hasattr(Gtk.Notebook, "set_scrollable"):
    Gtk.Notebook.set_scrollable = lambda self, *args, **kwargs: None
if not hasattr(Gtk.Notebook, "set_tooltip_text"):
    Gtk.Notebook.set_tooltip_text = lambda self, *args, **kwargs: None
if not hasattr(Gtk.Notebook, "connect"):
    Gtk.Notebook.connect = lambda self, *args, **kwargs: None
if not hasattr(Gtk.Notebook, "show"):
    Gtk.Notebook.show = lambda self, *args, **kwargs: None
if not hasattr(Gtk, "PositionType"):
    Gtk.PositionType = types.SimpleNamespace(TOP=0)
elif not hasattr(Gtk.PositionType, "TOP"):
    Gtk.PositionType.TOP = 0

from GTKUI.Persona_manager.persona_management import PersonaManagement
from GTKUI.Persona_manager.Persona_Type_Tab.persona_type_tab import PersonaTypeTab




class _AtlasStub:
    def __init__(self) -> None:
        self.editor_state = None
        self.update_payload = None
        self.messages = []
        self.export_requests = []
        self.import_payload = None
        self.import_result = {"success": True, "persona": {"name": "Imported"}, "warnings": []}
        self.persona_manager = None
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
        self.review_attestations: list[tuple[str, dict[str, Any]]] = []
        self.audit_history: list[Dict[str, Any]] = []

    def register_message_dispatcher(self, handler) -> None:  # pragma: no cover - stored for completeness
        self.dispatcher = handler

    def get_persona_editor_state(self, _name: str):
        return self.editor_state

    def update_persona_from_editor(self, *args, **kwargs):
        self.update_payload = args + (kwargs,)
        return {"success": True, "persona": {"name": args[1]["name"]}}

    def show_persona_message(self, role: str, message: str) -> None:
        self.messages.append((role, message))

    def export_persona_bundle(self, persona_name: str, *, signing_key: str):
        self.export_requests.append((persona_name, signing_key))
        return {
            "success": True,
            "bundle_bytes": b"bundle-data",
            "warnings": ["Export warning"],
        }

    def import_persona_bundle(self, *, bundle_bytes: bytes, signing_key: str, rationale: str = "Imported via UI"):
        self.import_payload = {
            "bundle_bytes": bundle_bytes,
            "signing_key": signing_key,
            "rationale": rationale,
        }
        return dict(self.import_result)

    def get_persona_names(self) -> list[str]:  # pragma: no cover - UI helper
        return ["Specialist"]

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
        return [], len(self.audit_history)


class _GeneralStub:
    def get_name(self) -> str:
        return "Specialist"

    def get_meaning(self) -> str:
        return ""

    def get_start_locked(self) -> str:
        return "start"

    def get_editable_content(self) -> str:
        return "body"

    def get_end_locked(self) -> str:
        return "end"


class _PersonaTypeStub:
    def get_values(self) -> dict:
        return {}


class _EntryStub:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakeWindow:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


def _persona_state() -> dict:
    return {
        "original_name": "Specialist",
        "general": {"name": "Specialist"},
        "provider": {"provider": "openai", "model": "gpt-4o"},
        "speech": {"Speech_provider": "11labs", "voice": "jack"},
        "tools": {
            "allowed": ["beta_tool", "custom_tool"],
            "available": [
                {
                    "name": "beta_tool",
                    "enabled": True,
                    "order": 0,
                    "metadata": {
                        "name": "beta_tool",
                        "description": "Beta",
                        "safety_level": "medium",
                        "cost_per_call": 0.0,
                        "cost_unit": "USD",
                    },
                },
                {
                    "name": "custom_tool",
                    "enabled": True,
                    "order": 1,
                    "metadata": {
                        "name": "custom_tool",
                        "description": "Custom",
                        "safety_level": "high",
                        "cost_per_call": 1.0,
                        "cost_unit": "USD",
                    },
                },
                {
                    "name": "alpha_tool",
                    "enabled": False,
                    "order": 2,
                    "metadata": {
                        "name": "alpha_tool",
                        "description": "Alpha",
                        "safety_level": "low",
                        "cost_per_call": 0.2,
                        "cost_unit": "USD",
                    },
                },
            ],
        },
        "skills": {
            "available": [
                {
                    "name": "beta_skill",
                    "enabled": True,
                    "order": 0,
                    "metadata": {
                        "name": "Beta Skill",
                        "instruction_prompt": "Use beta skill wisely.",
                    },
                },
                {
                    "name": "gamma_skill",
                    "enabled": False,
                    "order": 1,
                    "metadata": {
                        "name": "Gamma Skill",
                        "instruction_prompt": "Gamma skill adds depth.",
                    },
                },
            ]
        },
    }


def test_tools_tab_collects_selection_order():
    atlas = _AtlasStub()
    persona_state = _persona_state()
    atlas.editor_state = persona_state

    parent_window = Gtk.Window()
    manager = PersonaManagement(atlas, parent_window)
    manager.general_tab = _GeneralStub()
    manager.persona_type_tab = _PersonaTypeStub()
    manager.provider_entry = _EntryStub("openai")
    manager.model_entry = _EntryStub("gpt-4o")
    manager.speech_provider_entry = _EntryStub("11labs")
    manager.voice_entry = _EntryStub("jack")
    manager._current_editor_state = {"original_name": "Specialist"}

    tools_widget = manager.create_tools_tab(persona_state)
    assert isinstance(tools_widget, Gtk.Widget)

    # Simulate user edits: enable alpha, disable custom, move alpha to front
    manager.tool_rows["alpha_tool"]["check"].set_active(True)
    manager.tool_rows["custom_tool"]["check"].set_active(False)
    manager._move_tool_row("alpha_tool", 0)

    fake_window = _FakeWindow()
    manager.save_persona_settings(fake_window)

    assert fake_window.destroyed is True
    assert atlas.update_payload is not None
    tools_arg = atlas.update_payload[5]
    assert tools_arg == ["alpha_tool", "beta_tool"]

    # ensure refresh pulled new state
    assert manager._current_editor_state.get("original_name") == "Specialist"


def test_skills_tab_collects_selection_order():
    atlas = _AtlasStub()
    persona_state = _persona_state()
    atlas.editor_state = persona_state

    parent_window = Gtk.Window()
    manager = PersonaManagement(atlas, parent_window)
    manager.general_tab = _GeneralStub()
    manager.persona_type_tab = _PersonaTypeStub()
    manager.provider_entry = _EntryStub("openai")
    manager.model_entry = _EntryStub("gpt-4o")
    manager.speech_provider_entry = _EntryStub("11labs")
    manager.voice_entry = _EntryStub("jack")
    manager._current_editor_state = {"original_name": "Specialist"}

    skills_widget = manager.create_skills_tab(persona_state)
    assert isinstance(skills_widget, Gtk.Widget)

    # Enable gamma skill and move it to the top of the list
    manager.skill_rows["gamma_skill"]["check"].set_active(True)
    manager._move_skill_row("gamma_skill", 0)

    fake_window = _FakeWindow()
    manager.save_persona_settings(fake_window)

    assert fake_window.destroyed is True
    assert atlas.update_payload is not None
    skills_arg = atlas.update_payload[6]
    assert skills_arg == ["gamma_skill", "beta_skill"]


def test_export_persona_bundle_writes_file(tmp_path):
    atlas = _AtlasStub()
    atlas.editor_state = _persona_state()
    parent_window = Gtk.Window()
    manager = PersonaManagement(atlas, parent_window)
    manager._current_editor_state = {"original_name": "Specialist"}

    export_path = tmp_path / "Specialist.atlasbundle"
    manager._choose_file_path = lambda **_kwargs: str(export_path)
    manager._prompt_signing_key = lambda _title: "secret"

    manager._on_export_persona_clicked(None)

    assert atlas.export_requests == [("Specialist", "secret")]
    assert export_path.read_bytes() == b"bundle-data"
    assert any("Export warning" in message for _role, message in atlas.messages)


def test_import_persona_bundle_triggers_backend(tmp_path):
    atlas = _AtlasStub()
    atlas.import_result = {
        "success": True,
        "persona": {"name": "Imported"},
        "warnings": ["Missing tools pruned"],
    }

    parent_window = Gtk.Window()
    manager = PersonaManagement(atlas, parent_window)

    bundle_path = tmp_path / "bundle.atlasbundle"
    bundle_path.write_bytes(b"bundle-bytes")

    manager._choose_file_path = lambda **_kwargs: str(bundle_path)
    manager._prompt_signing_key = lambda _title: "import-secret"

    manager._on_import_persona_clicked(None)

    assert atlas.import_payload["bundle_bytes"] == b"bundle-bytes"
    assert atlas.import_payload["signing_key"] == "import-secret"
    assert any("Missing tools pruned" in message for _role, message in atlas.messages)


def test_personal_assistant_calendar_write_toggle_updates_state():
    persona_state = {
        'flags': {
            'sys_info_enabled': False,
            'user_profile_enabled': False,
            'type': {
                'personal_assistant': {
                    'enabled': True,
                    'access_to_calendar': True,
                    'calendar_write_enabled': False,
                    'terminal_read_enabled': False,
                    'terminal_write_enabled': False,
                }
            },
        }
    }

    class _GeneralPersonaStub:
        def __init__(self, state: Dict[str, Any]) -> None:
            self.state = state
            self.calls: list[tuple[bool, Optional[Dict[str, Any]]]] = []

        def set_personal_assistant_enabled(
            self,
            enabled: bool,
            extras: Optional[Dict[str, Any]] = None,
        ) -> bool:
            self.calls.append((enabled, extras))
            flags = self.state.setdefault('flags', {})
            persona_type = flags.setdefault('type', {})
            entry = persona_type.setdefault('personal_assistant', {})
            entry['enabled'] = enabled
            if extras:
                entry['access_to_calendar'] = extras.get('access_to_calendar', False)
                entry['calendar_write_enabled'] = extras.get('calendar_write_enabled', False)
                entry['terminal_read_enabled'] = extras.get('terminal_read_enabled', False)
                entry['terminal_write_enabled'] = extras.get('terminal_write_enabled', False)
            return True

    general_stub = _GeneralPersonaStub(persona_state)
    tab = PersonaTypeTab(persona_state, general_stub)

    assert tab.personal_assistant_calendar_switch.get_active() is True
    assert tab.personal_assistant_calendar_write_switch.get_active() is False
    assert tab.personal_assistant_terminal_read_switch.get_active() is False
    assert tab.personal_assistant_terminal_write_switch.get_active() is False

    tab.personal_assistant_calendar_write_switch.set_active(True)
    tab.on_personal_assistant_calendar_write_toggled(tab.personal_assistant_calendar_write_switch, None)

    assert general_stub.calls[-1][0] is True
    assert general_stub.calls[-1][1]['calendar_write_enabled'] is True
    assert persona_state['flags']['type']['personal_assistant']['calendar_write_enabled'] is True

    tab.personal_assistant_calendar_switch.set_active(False)
    tab.on_personal_assistant_calendar_toggled(tab.personal_assistant_calendar_switch, None)

    latest_extras = general_stub.calls[-1][1]
    assert latest_extras['access_to_calendar'] is False
    assert latest_extras['calendar_write_enabled'] is False
    assert tab.personal_assistant_calendar_write_switch.get_active() is False
    assert latest_extras['terminal_read_enabled'] is False
    assert latest_extras['terminal_write_enabled'] is False

    tab.personal_assistant_terminal_read_switch.set_active(True)
    tab.on_personal_assistant_terminal_read_toggled(tab.personal_assistant_terminal_read_switch, None)

    latest_extras = general_stub.calls[-1][1]
    assert latest_extras['terminal_read_enabled'] is True
    assert latest_extras['terminal_write_enabled'] is False
    assert getattr(tab.personal_assistant_terminal_write_switch, '_sensitive', None) is True

    tab.personal_assistant_terminal_write_switch.set_active(True)
    tab.on_personal_assistant_terminal_write_toggled(tab.personal_assistant_terminal_write_switch, None)

    latest_extras = general_stub.calls[-1][1]
    assert latest_extras['terminal_write_enabled'] is True
    assert persona_state['flags']['type']['personal_assistant']['terminal_write_enabled'] is True

    tab.personal_assistant_terminal_read_switch.set_active(False)
    tab.on_personal_assistant_terminal_read_toggled(tab.personal_assistant_terminal_read_switch, None)

    latest_extras = general_stub.calls[-1][1]
    assert latest_extras['terminal_read_enabled'] is False
    assert latest_extras['terminal_write_enabled'] is False
    assert tab.personal_assistant_terminal_write_switch.get_active() is False
    assert getattr(tab.personal_assistant_terminal_write_switch, '_sensitive', None) is False
