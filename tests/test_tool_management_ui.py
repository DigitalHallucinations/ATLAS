import sys
import types
from typing import Any, Dict

import pytest

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are loaded

from gi.repository import Gtk

pango_module = sys.modules.setdefault("gi.repository.Pango", types.ModuleType("Pango"))
sys.modules.setdefault("Pango", pango_module)

align = getattr(Gtk, "Align", None)
if align is not None and not hasattr(align, "FILL"):
    setattr(align, "FILL", 0)

_chat_helper = sys.modules.get("tests.test_chat_async_helper")
if _chat_helper is not None:
    dummy_base = getattr(_chat_helper, "_DummyWidget", object)
else:  # pragma: no cover - fallback when helper is unavailable
    dummy_base = object

if not hasattr(Gtk, "Separator"):
    Gtk.Separator = type("Separator", (dummy_base,), {})

if not hasattr(Gtk, "PolicyType"):
    Gtk.PolicyType = types.SimpleNamespace(NEVER=0, AUTOMATIC=1)

notebook_cls = getattr(Gtk, "Notebook", None)
if notebook_cls is not None and not hasattr(notebook_cls, "set_scrollable"):
    notebook_cls.set_scrollable = lambda self, *args, **kwargs: None
if notebook_cls is not None and not hasattr(notebook_cls, "set_tab_reorderable"):
    notebook_cls.set_tab_reorderable = lambda self, *args, **kwargs: None
if notebook_cls is not None and not hasattr(notebook_cls, "set_current_page"):
    notebook_cls.set_current_page = lambda self, *args, **kwargs: None
if notebook_cls is not None and not hasattr(notebook_cls, "append_page"):
    notebook_cls.append_page = lambda self, child, label: 0

if not hasattr(Gtk, "SelectionMode"):
    Gtk.SelectionMode = types.SimpleNamespace(NONE=0, SINGLE=1, MULTIPLE=2)

accessible_role = getattr(Gtk, "AccessibleRole", None)
if accessible_role is None:
    Gtk.AccessibleRole = types.SimpleNamespace(LIST=0, LIST_ITEM=1)
else:
    setattr(accessible_role, "LIST", getattr(accessible_role, "LIST", 0))
    setattr(accessible_role, "LIST_ITEM", getattr(accessible_role, "LIST_ITEM", 1))

if not hasattr(Gtk, "ListBoxRow"):
    Gtk.ListBoxRow = type("ListBoxRow", (dummy_base,), {})


class _DummyPersonaManagement:
    def __init__(self, atlas, parent):
        self._widget = Gtk.Box()

    def get_embeddable_widget(self):
        return self._widget


class _DummyProviderManagement:
    def __init__(self, atlas, parent):
        self._widget = Gtk.Box()

    def get_embeddable_widget(self):
        return self._widget


class _ParentWindowStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def show_error_dialog(self, message: str) -> None:
        self.errors.append(message)


class _PersonaManagerStub:
    def __init__(self) -> None:
        self.allowed = ["google_search"]
        self.saved_calls: list[Dict[str, Any]] = []
        self.get_calls = 0

    def get_persona(self, persona_name: str) -> Dict[str, Any]:
        self.get_calls += 1
        return {"name": persona_name, "allowed_tools": list(self.allowed)}

    def set_allowed_tools(self, persona_name: str, tools: list[str]) -> Dict[str, Any]:
        record = {"persona": persona_name, "tools": list(tools)}
        self.saved_calls.append(record)
        self.allowed = list(tools)
        return {"success": True, "persona": record}


class _AtlasStub:
    def __init__(self) -> None:
        self.tool_fetches = 0
        self.skill_fetches = 0
        self.tool_requests: list[Dict[str, Any]] = []
        self.skill_requests: list[Dict[str, Any]] = []
        self.persona_manager = _PersonaManagerStub()
        self.server = types.SimpleNamespace(get_tools=self._get_tools, get_skills=self._get_skills)

    def is_initialized(self) -> bool:
        return True

    def get_active_persona_name(self) -> str:
        return "Atlas"

    def _get_tools(self, **kwargs: Any) -> Dict[str, Any]:
        self.tool_fetches += 1
        self.tool_requests.append(dict(kwargs))
        return {
            "count": 2,
            "tools": [
                {
                    "name": "google_search",
                    "title": "Google Search",
                    "summary": "Search the web for up-to-date information.",
                    "capabilities": ["web_search", "news"],
                    "auth": {"required": True, "provider": "Google", "status": "Linked"},
                },
                {
                    "name": "terminal_command",
                    "title": "Terminal",
                    "summary": "Execute safe shell commands.",
                    "capabilities": ["system"],
                    "auth": {"required": False, "status": "Optional"},
                },
            ],
        }

    def _get_skills(self, **kwargs: Any) -> Dict[str, Any]:
        self.skill_fetches += 1
        self.skill_requests.append(dict(kwargs))
        return {
            "count": 2,
            "skills": [
                {
                    "name": "ResearchBrief",
                    "summary": "Compose a short research summary using trusted sources.",
                    "version": "1.2.0",
                    "persona": None,
                    "category": "Research",
                    "required_tools": ["google_search"],
                    "required_capabilities": ["web_search"],
                    "capability_tags": ["analysis", "writing"],
                    "safety_notes": "Review generated content for accuracy.",
                    "source": "modules/Skills/skills.json",
                },
                {
                    "name": "FollowUpPlanner",
                    "summary": "Suggest follow-up questions based on the current chat context.",
                    "version": "0.4.1",
                    "persona": "Atlas",
                    "category": "Conversation",
                    "required_tools": [],
                    "required_capabilities": ["conversation"],
                    "capability_tags": ["conversation", "planning"],
                    "safety_notes": "No elevated permissions required.",
                    "source": "modules/Personas/Atlas/Skills/skills.json",
                },
            ],
        }


def _walk(widget: Any):
    yield widget
    children = getattr(widget, "children", []) or []
    for child in children:
        yield from _walk(child)


def test_sidebar_adds_tools_button_and_workspace(monkeypatch):
    persona_stub = types.ModuleType("GTKUI.Persona_manager.persona_management")
    persona_stub.PersonaManagement = _DummyPersonaManagement
    provider_stub = types.ModuleType("GTKUI.Provider_manager.provider_management")
    provider_stub.ProviderManagement = _DummyProviderManagement

    monkeypatch.setitem(sys.modules, "GTKUI.Persona_manager.persona_management", persona_stub)
    monkeypatch.setitem(sys.modules, "GTKUI.Provider_manager.provider_management", provider_stub)

    from GTKUI import sidebar

    monkeypatch.setattr(sidebar, "apply_css", lambda: None)

    atlas = _AtlasStub()
    window = sidebar.MainWindow(atlas)

    buttons = [
        widget
        for widget in _walk(window.sidebar)
        if getattr(widget, "_tooltip", "") == "Tools"
    ]
    assert buttons, "Sidebar should include a Tools navigation button"

    window.show_tools_menu()
    assert "tools" in window._pages, "Tools page should be registered after opening"
    assert atlas.tool_fetches >= 1, "Opening tools workspace should query the backend"
    tools_widget = window.tool_management.get_embeddable_widget()
    assert window._pages["tools"] is tools_widget


def test_sidebar_adds_skills_button_and_workspace(monkeypatch):
    persona_stub = types.ModuleType("GTKUI.Persona_manager.persona_management")
    persona_stub.PersonaManagement = _DummyPersonaManagement
    provider_stub = types.ModuleType("GTKUI.Provider_manager.provider_management")
    provider_stub.ProviderManagement = _DummyProviderManagement

    monkeypatch.setitem(sys.modules, "GTKUI.Persona_manager.persona_management", persona_stub)
    monkeypatch.setitem(sys.modules, "GTKUI.Provider_manager.provider_management", provider_stub)

    from GTKUI import sidebar

    monkeypatch.setattr(sidebar, "apply_css", lambda: None)

    atlas = _AtlasStub()
    window = sidebar.MainWindow(atlas)

    buttons = [
        widget
        for widget in _walk(window.sidebar)
        if getattr(widget, "_tooltip", "") == "Skills"
    ]
    assert buttons, "Sidebar should include a Skills navigation button"

    window.show_skills_menu()
    assert "skills" in window._pages, "Skills page should be registered after opening"
    assert atlas.skill_fetches >= 1, "Opening skills workspace should query the backend"
    skills_widget = window.skill_management.get_embeddable_widget()
    assert window._pages["skills"] is skills_widget


def test_tool_management_save_and_reset():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.tool_fetches == 1

    manager._select_tool("terminal_command")
    manager._on_switch_state_set(manager._switch, True)
    manager._on_save_clicked(manager._save_button)

    assert atlas.persona_manager.saved_calls, "Save action should persist via the persona manager"
    record = atlas.persona_manager.saved_calls[-1]
    assert record["persona"] == "Atlas"
    assert "terminal_command" in record["tools"]

    # Simulate the persona being reverted on disk and ensure reset reloads state.
    atlas.persona_manager.allowed = ["google_search"]
    manager._on_reset_clicked(manager._reset_button)

    assert atlas.tool_fetches >= 2, "Reset should trigger a fresh backend fetch"
    assert manager._enabled_tools == {"google_search"}
    assert not parent.errors, "Successful actions should not surface errors"


def test_tool_management_filter_modes():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.tool_requests
    assert atlas.tool_requests[-1] == {"persona": "Atlas"}

    scope_widget = manager._scope_selector
    assert scope_widget is not None
    assert scope_widget.get_active_text() == "Persona tools"

    scope_widget.set_active(1)
    manager._on_scope_changed(scope_widget)

    assert manager._tool_scope == "all"
    assert atlas.tool_requests[-1] == {}
    assert scope_widget.get_active_text() == "All tools"
    assert manager._entries, "Entries should remain populated when showing all tools"
    assert not getattr(manager._switch, "_sensitive", True)

    scope_widget.set_active(0)
    manager._on_scope_changed(scope_widget)

    assert manager._tool_scope == "persona"
    assert atlas.tool_requests[-1] == {"persona": "Atlas"}
    assert scope_widget.get_active_text() == "Persona tools"
    assert getattr(manager._switch, "_sensitive", False)


def test_skill_management_renders_payloads():
    from GTKUI.Skill_manager.skill_management import SkillManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = SkillManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.skill_fetches == 1
    assert manager._entries, "Skill entries should be populated from backend payload"
    assert manager._active_skill is not None
    assert not parent.errors


def test_skill_management_scope_modes():
    from GTKUI.Skill_manager.skill_management import SkillManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = SkillManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.skill_requests
    assert atlas.skill_requests[-1] == {"persona": "Atlas"}

    scope_widget = manager._scope_selector
    assert scope_widget is not None
    assert scope_widget.get_active_text() == "Persona skills"

    initial_fetches = atlas.skill_fetches

    scope_widget.set_active(1)
    manager._on_scope_changed(scope_widget)

    assert manager._skill_scope == "all"
    assert atlas.skill_requests[-1] == {}
    assert atlas.skill_fetches > initial_fetches
    assert scope_widget.get_active_text() == "All skills"
    assert manager._entries, "Entries should remain populated when showing all skills"

    scope_widget.set_active(0)
    manager._on_scope_changed(scope_widget)

    assert manager._skill_scope == "persona"
    assert atlas.skill_requests[-1] == {"persona": "Atlas"}
    assert scope_widget.get_active_text() == "Persona skills"
