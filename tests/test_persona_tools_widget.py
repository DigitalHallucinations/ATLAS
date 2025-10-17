from __future__ import annotations

import sys
import types

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
from gi.repository import Gtk

from GTKUI.Persona_manager.persona_management import PersonaManagement




class _AtlasStub:
    def __init__(self) -> None:
        self.editor_state = None
        self.update_payload = None
        self.messages = []

    def register_message_dispatcher(self, handler) -> None:  # pragma: no cover - stored for completeness
        self.dispatcher = handler

    def get_persona_editor_state(self, _name: str):
        return self.editor_state

    def update_persona_from_editor(self, *args, **kwargs):
        self.update_payload = args + (kwargs,)
        return {"success": True, "persona": {"name": args[1]["name"]}}

    def show_persona_message(self, role: str, message: str) -> None:
        self.messages.append((role, message))


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
    manager._move_tool_row("alpha_tool", -1)
    manager._move_tool_row("alpha_tool", -1)

    fake_window = _FakeWindow()
    manager.save_persona_settings(fake_window)

    assert fake_window.destroyed is True
    assert atlas.update_payload is not None
    tools_arg = atlas.update_payload[5]
    assert tools_arg == ["alpha_tool", "beta_tool"]

    # ensure refresh pulled new state
    assert manager._current_editor_state.get("original_name") == "Specialist"
