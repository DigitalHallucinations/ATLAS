import sys
import types

jsonschema_stub = sys.modules.setdefault("jsonschema", types.ModuleType("jsonschema"))


class _Validator:
    def __init__(self, _schema=None):
        self.schema = _schema

    def validate(self, _payload):  # pragma: no cover - interface shim
        return None


class _ValidationError(Exception):
    pass


jsonschema_stub.Draft7Validator = _Validator
jsonschema_stub.ValidationError = _ValidationError
jsonschema_stub.Draft202012Validator = _Validator
jsonschema_stub.exceptions = types.SimpleNamespace(
    ValidationError=_ValidationError,
    SchemaError=_ValidationError,
)

import tests.test_chat_async_helper  # noqa: E402,F401 - ensure GTK stubs are loaded

gtk_module = sys.modules.setdefault("gi.repository.Gtk", types.ModuleType("Gtk"))
sys.modules.setdefault("Gtk", gtk_module)
pango_module = sys.modules.setdefault("gi.repository.Pango", types.ModuleType("Pango"))
sys.modules.setdefault("Pango", pango_module)


if not hasattr(gtk_module, "Switch"):
    class _DummySwitch:
        def __init__(self, *args, **kwargs):
            pass

        def set_active(self, *_args, **_kwargs):  # pragma: no cover - stub method
            return None


    gtk_module.Switch = _DummySwitch

if not hasattr(gtk_module, "Dialog"):
    gtk_module.Dialog = type("Dialog", (), {})

if not hasattr(gtk_module, "SelectionMode"):
    gtk_module.SelectionMode = types.SimpleNamespace(NONE=0, SINGLE=1, MULTIPLE=2)

if not hasattr(gtk_module, "ListBoxRow"):
    class _DummyListBoxRow:
        def __init__(self, *args, **kwargs):
            self.child = None

        def set_child(self, child):  # pragma: no cover - widget helper
            self.child = child


    gtk_module.ListBoxRow = _DummyListBoxRow

from gi.repository import Gtk

from GTKUI.Persona_manager.persona_management import PersonaManagement


class _AtlasStub:
    def __init__(self) -> None:
        self.dispatcher = None

    def register_message_dispatcher(self, handler) -> None:
        self.dispatcher = handler


def _persona_state(reason: str = "Policy restriction") -> dict:
    return {
        "general": {"name": "Specialist"},
        "tools": {
            "allowed": ["allowed_tool"],
            "available": [
                {
                    "name": "restricted_tool",
                    "enabled": False,
                    "order": 0,
                    "metadata": {
                        "name": "restricted_tool",
                        "description": "Restricted tool",
                    },
                    "disabled": True,
                    "disabled_reason": reason,
                },
                {
                    "name": "allowed_tool",
                    "enabled": True,
                    "order": 1,
                    "metadata": {
                        "name": "allowed_tool",
                        "description": "Allowed tool",
                    },
                },
            ],
        },
    }


def test_policy_disabled_tool_renders_badge_and_is_readonly():
    atlas = _AtlasStub()
    manager = PersonaManagement(atlas, Gtk.Window())
    manager.create_tools_tab(_persona_state())

    row_info = manager.tool_rows.get("restricted_tool")
    assert row_info is not None, "Restricted tool should be present in tool rows"

    check = row_info.get("check")
    assert isinstance(check, Gtk.CheckButton)
    assert not check.get_active(), "Policy disabled tools should not be active"
    assert not getattr(check, "_sensitive", True), "Policy disabled tools should be read-only"

    tooltip = getattr(check, "_tooltip", "") or ""
    assert "Policy restriction" in tooltip

    badge = row_info.get("badge")
    assert badge is not None, "Badge should be rendered for policy disabled tools"
    assert getattr(badge, "label", "") == "Policy restriction"


def test_policy_enforced_tool_not_collected_even_if_marked_enabled():
    persona = _persona_state()
    persona["tools"]["available"][0]["enabled"] = True

    atlas = _AtlasStub()
    manager = PersonaManagement(atlas, Gtk.Window())
    manager.create_tools_tab(persona)

    allowed = manager._collect_tool_payload()
    assert "restricted_tool" not in allowed
    assert "allowed_tool" in allowed
