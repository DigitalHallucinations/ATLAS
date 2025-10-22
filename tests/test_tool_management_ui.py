import copy
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
        self.toasts: list[str] = []

    def show_error_dialog(self, message: str) -> None:
        self.errors.append(message)

    def show_success_toast(self, message: str) -> None:
        self.toasts.append(message)


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
        self.task_fetches = 0
        self.tool_requests: list[Dict[str, Any]] = []
        self.skill_requests: list[Dict[str, Any]] = []
        self.task_requests: list[Dict[str, Any]] = []
        self.task_transitions: list[Dict[str, Any]] = []
        self.task_contexts: list[Any] = []
        self.settings_updates: list[Dict[str, Any]] = []
        self.credential_updates: list[Dict[str, Any]] = []
        self.persona_manager = _PersonaManagerStub()
        self._tool_catalog: list[Dict[str, Any]] = [
            {
                "name": "google_search",
                "title": "Google Search",
                "summary": "Search the web for up-to-date information.",
                "capabilities": ["web_search", "news"],
                "persona": None,
                "auth": {"required": True, "provider": "Google", "status": "Linked"},
                "settings": {"enabled": True, "providers": ["primary"]},
                "credentials": {
                    "GOOGLE_API_KEY": {"configured": False, "hint": "MASKED", "required": True}
                },
            },
            {
                "name": "terminal_command",
                "title": "Terminal",
                "summary": "Execute safe shell commands.",
                "capabilities": ["system"],
                "persona": "Atlas",
                "auth": {"required": False, "status": "Optional"},
                "settings": {"enabled": True, "shell": "bash"},
                "credentials": {},
            },
            {
                "name": "atlas_curated_search",
                "title": "Atlas Curated Search",
                "summary": "Search curated internal resources.",
                "capabilities": ["curated_search"],
                "persona": None,
                "persona_allowlist": ["Atlas", "Researcher"],
                "auth": {"required": True, "provider": "Atlas", "status": "Linked"},
                "settings": {"enabled": False},
                "credentials": {},
            },
            {
                "name": "restricted_calculator",
                "title": "Restricted Calculator",
                "summary": "Persona restricted tool.",
                "capabilities": ["math"],
                "persona": None,
                "persona_allowlist": ["Researcher"],
                "auth": {"required": False},
                "settings": {"enabled": False},
                "credentials": {"TOKEN": {"configured": True}},
            },
        ]
        self._task_catalog: list[Dict[str, Any]] = [
            {
                "id": "task-1",
                "title": "Draft proposal",
                "description": "Prepare an initial proposal for review.",
                "status": "draft",
                "priority": 1,
                "owner_id": "user-1",
                "session_id": "session-1",
                "conversation_id": "conv-1",
                "tenant_id": "default",
                "metadata": {
                    "persona": "Atlas",
                    "required_skills": ["analysis"],
                    "required_tools": ["google_search"],
                    "acceptance_criteria": ["Proposal outline ready"],
                    "dependencies": [
                        {"id": "support-1", "title": "Collect references", "status": "done"}
                    ],
                },
                "created_at": "2024-01-01T10:00:00+00:00",
                "updated_at": "2024-01-01T10:00:00+00:00",
                "due_at": "2024-01-05T12:00:00+00:00",
            },
            {
                "id": "task-2",
                "title": "Deep dive research",
                "description": "Collect supporting data for the proposal.",
                "status": "ready",
                "priority": 2,
                "owner_id": "user-1",
                "session_id": "session-1",
                "conversation_id": "conv-2",
                "tenant_id": "default",
                "metadata": {
                    "persona": "Researcher",
                    "required_skills": ["research"],
                    "required_tools": ["terminal_command"],
                    "acceptance_criteria": ["Data summarized for review"],
                    "dependencies": [
                        {"id": "task-1", "title": "Draft proposal", "status": "draft"}
                    ],
                },
                "created_at": "2024-01-02T09:00:00+00:00",
                "updated_at": "2024-01-02T09:00:00+00:00",
                "due_at": None,
            },
            {
                "id": "task-3",
                "title": "QA review",
                "description": "Review deliverables for completeness.",
                "status": "in_progress",
                "priority": 3,
                "owner_id": None,
                "session_id": None,
                "conversation_id": "conv-3",
                "tenant_id": "default",
                "metadata": {
                    "required_skills": ["quality assurance"],
                    "required_tools": [],
                    "acceptance_criteria": ["Checklist completed"],
                    "dependencies": [],
                },
                "created_at": "2024-01-03T08:00:00+00:00",
                "updated_at": "2024-01-03T08:30:00+00:00",
                "due_at": None,
            },
        ]
        self.server = types.SimpleNamespace(
            get_tools=self._get_tools,
            get_skills=self._get_skills,
            list_tasks=self._list_tasks,
            get_task=self._get_task,
            transition_task=self._transition_task,
        )

    def is_initialized(self) -> bool:
        return True

    def get_active_persona_name(self) -> str:
        return "Atlas"

    def list_tools(self) -> list[Dict[str, Any]]:
        self.tool_fetches += 1
        return [copy.deepcopy(entry) for entry in self._tool_catalog]

    def update_tool_settings(self, tool_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool": tool_name, "settings": copy.deepcopy(settings)}
        self.settings_updates.append(payload)
        for entry in self._tool_catalog:
            if entry.get("name") == tool_name:
                entry_settings = entry.setdefault("settings", {})
                entry_settings.update(copy.deepcopy(settings))
                break
        return {"success": True, "settings": copy.deepcopy(settings)}

    def update_tool_credentials(self, tool_name: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"tool": tool_name, "credentials": copy.deepcopy(credentials)}
        self.credential_updates.append(payload)
        for entry in self._tool_catalog:
            if entry.get("name") == tool_name:
                entry_credentials = entry.setdefault("credentials", {})
                for key, value in credentials.items():
                    block = entry_credentials.setdefault(key, {})
                    block["configured"] = True
                    block["hint"] = "MASKED"
                break
        return {"success": True}

    def _get_tools(self, **kwargs: Any) -> Dict[str, Any]:
        self.tool_requests.append(dict(kwargs))
        self.tool_fetches += 1
        persona = kwargs.get("persona")

        def _matches(entry: Dict[str, Any]) -> bool:
            if not persona:
                return True
            scope = entry.get("persona")
            if scope:
                return scope == persona
            allowlist = entry.get("persona_allowlist") or []
            if allowlist:
                return persona in allowlist
            return True

        tools = [copy.deepcopy(entry) for entry in self._tool_catalog if _matches(entry)]
        return {"count": len(tools), "tools": tools}

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

    def _list_tasks(self, params: Dict[str, Any] | None = None, *, context: Any | None = None) -> Dict[str, Any]:
        params = dict(params or {})
        self.task_requests.append(dict(params))
        if context is not None:
            self.task_contexts.append(context)
        self.task_fetches += 1
        status_filter = params.get("status")
        if isinstance(status_filter, str):
            statuses = {status_filter}
        elif isinstance(status_filter, (list, tuple, set)):
            statuses = {str(value) for value in status_filter}
        else:
            statuses = None
        items = []
        for entry in self._task_catalog:
            if statuses and entry.get("status") not in statuses:
                continue
            items.append(copy.deepcopy(entry))
        return {
            "items": items,
            "page": {"next_cursor": None, "page_size": len(items), "count": len(items)},
        }

    def _get_task(
        self,
        task_id: str,
        *,
        context: Any | None = None,
        include_events: bool = False,
    ) -> Dict[str, Any]:
        if context is not None:
            self.task_contexts.append(context)
        for entry in self._task_catalog:
            if entry.get("id") == task_id:
                payload = copy.deepcopy(entry)
                if include_events:
                    payload["events"] = []
                return payload
        return {"id": task_id, "status": "unknown", "metadata": {}}

    def _transition_task(
        self,
        task_id: str,
        target_status: str,
        *,
        context: Any | None = None,
        expected_updated_at: Any | None = None,
    ) -> Dict[str, Any]:
        record = {
            "task_id": task_id,
            "target": str(target_status),
            "context": context,
            "expected": expected_updated_at,
        }
        self.task_transitions.append(record)
        for entry in self._task_catalog:
            if entry.get("id") == task_id:
                entry["status"] = str(target_status)
                entry["updated_at"] = f"2024-02-{len(self.task_transitions):02d}T12:00:00+00:00"
                return copy.deepcopy(entry)
        return {"id": task_id, "status": str(target_status)}


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

    google_entry = manager._entry_lookup.get("google_search")
    assert google_entry is not None
    assert google_entry.raw_metadata["settings"] == {"enabled": True, "providers": ["primary"]}
    assert google_entry.raw_metadata["credentials"]["GOOGLE_API_KEY"]["configured"] is False

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


def test_tool_management_settings_editor_updates_backend():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()
    manager._select_tool("google_search")

    field = manager._settings_inputs.get("enabled")
    assert field is not None
    field.set_text("false")

    apply_button = manager._settings_apply_button
    assert apply_button is not None
    manager._on_settings_apply_clicked(apply_button)

    assert atlas.settings_updates, "Settings updates should be sent to the backend stub"
    record = atlas.settings_updates[-1]
    assert record["tool"] == "google_search"
    assert record["settings"]["enabled"] is False
    assert atlas.tool_fetches >= 2, "Applying settings should refresh tool metadata"
    assert parent.toasts and parent.toasts[-1] == "Settings saved."


def test_tool_management_credentials_validation_and_refresh():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    manager.get_embeddable_widget()
    manager._select_tool("google_search")

    field = manager._credential_inputs.get("GOOGLE_API_KEY")
    assert field is not None
    placeholder = getattr(field, "placeholder", getattr(field, "_placeholder_text", ""))
    assert placeholder
    assert "MASKED" not in placeholder
    assert all(char == "•" for char in placeholder.strip()), "Credentials placeholder should be masked"

    apply_button = manager._credentials_apply_button
    assert apply_button is not None
    manager._on_credentials_apply_clicked(apply_button)

    assert not atlas.credential_updates, "Empty credentials should not be submitted when required"
    assert "error" in getattr(field, "_css_classes", set()), "Missing required credential should be highlighted"

    field.set_text("secret-token")
    manager._on_credentials_apply_clicked(apply_button)

    assert atlas.credential_updates, "Credential updates should reach the backend stub"
    cred_record = atlas.credential_updates[-1]
    assert cred_record["tool"] == "google_search"
    assert cred_record["credentials"]["GOOGLE_API_KEY"] == "secret-token"
    assert atlas.tool_fetches >= 2
    assert parent.toasts and parent.toasts[-1] == "Credentials saved."
    assert "error" not in getattr(field, "_css_classes", set())


def test_tool_management_filter_modes():
    from GTKUI.Tool_manager.tool_management import ToolManagement

    parent = _ParentWindowStub()
    atlas = _AtlasStub()
    manager = ToolManagement(atlas, parent)

    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.tool_fetches == 1

    persona_entries = {entry.name for entry in manager._entries}
    assert persona_entries == {"google_search", "terminal_command", "atlas_curated_search"}

    scope_widget = manager._scope_selector
    assert scope_widget is not None
    assert scope_widget.get_active_text() == "Persona tools"

    scope_widget.set_active(1)
    manager._on_scope_changed(scope_widget)

    assert manager._tool_scope == "all"
    assert atlas.tool_fetches >= 2
    assert scope_widget.get_active_text() == "All tools"
    assert manager._entries, "Entries should remain populated when showing all tools"
    assert not getattr(manager._switch, "_sensitive", True)
    assert not getattr(manager._settings_apply_button, "_sensitive", True)
    assert not getattr(manager._credentials_apply_button, "_sensitive", True)

    all_entries = {entry.name for entry in manager._entries}
    assert "restricted_calculator" in all_entries

    scope_widget.set_active(0)
    manager._on_scope_changed(scope_widget)

    assert manager._tool_scope == "persona"
    assert atlas.tool_fetches >= 3
    assert scope_widget.get_active_text() == "Persona tools"
    manager._select_tool("google_search")
    assert getattr(manager._switch, "_sensitive", False)
    assert getattr(manager._settings_apply_button, "_sensitive", False)
    assert getattr(manager._credentials_apply_button, "_sensitive", False)
    persona_entries_after = {entry.name for entry in manager._entries}
    assert persona_entries_after == {"google_search", "terminal_command", "atlas_curated_search"}


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
