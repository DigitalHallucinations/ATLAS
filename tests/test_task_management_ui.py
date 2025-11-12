from typing import Any

import types
import sys

if "ATLAS.ToolManager" not in sys.modules:
    tool_manager_stub = types.ModuleType("ATLAS.ToolManager")
    tool_manager_stub.load_function_map_from_current_persona = lambda *_args, **_kwargs: {}
    tool_manager_stub.load_functions_from_json = lambda *_args, **_kwargs: {}
    tool_manager_stub.compute_tool_policy_snapshot = lambda *_args, **_kwargs: {}
    tool_manager_stub._resolve_function_callable = (
        lambda *_args, **_kwargs: lambda *_inner_args, **_inner_kwargs: None
    )
    tool_manager_stub.get_tool_activity_log = lambda *_args, **_kwargs: []
    tool_manager_stub.ToolPolicyDecision = type("ToolPolicyDecision", (), {})
    tool_manager_stub.SandboxedToolRunner = type("SandboxedToolRunner", (), {})
    tool_manager_stub.use_tool = lambda *_args, **_kwargs: None
    tool_manager_stub.call_model_with_new_prompt = lambda *_args, **_kwargs: None
    sys.modules["ATLAS.ToolManager"] = tool_manager_stub

sys.modules.setdefault("tests.test_provider_manager", types.ModuleType("tests.test_provider_manager"))
sys.modules.setdefault("tests.test_speech_settings_facade", types.ModuleType("tests.test_speech_settings_facade"))
gi_stub = sys.modules.setdefault("gi", types.ModuleType("gi"))
if not hasattr(gi_stub, "require_version"):
    gi_stub.require_version = lambda *args, **kwargs: None
repository_stub = sys.modules.setdefault("gi.repository", types.ModuleType("gi.repository"))
gi_stub.repository = repository_stub

sqlalchemy_stub = types.ModuleType("sqlalchemy")
sqlalchemy_exc = types.ModuleType("sqlalchemy.exc")
setattr(sqlalchemy_exc, "IntegrityError", Exception)
sqlalchemy_stub.exc = sqlalchemy_exc
sys.modules.setdefault("sqlalchemy", sqlalchemy_stub)
sys.modules.setdefault("sqlalchemy.exc", sqlalchemy_exc)

jsonschema_stub = types.ModuleType("jsonschema")

class _DummyValidator:
    def __init__(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):  # pragma: no cover - stub helper
        return None

    def iter_errors(self, *args, **kwargs):  # pragma: no cover - stub helper
        return []


class _DummyValidationError(Exception):
    def __init__(self, message: str = "", path=None):
        super().__init__(message)
        self.message = message
        self.absolute_path = list(path or [])


jsonschema_stub.Draft7Validator = _DummyValidator
jsonschema_stub.Draft202012Validator = _DummyValidator
jsonschema_stub.ValidationError = _DummyValidationError
jsonschema_stub.exceptions = types.SimpleNamespace(ValidationError=_DummyValidationError)
sys.modules["jsonschema"] = jsonschema_stub
sys.modules["jsonschema.exceptions"] = jsonschema_stub.exceptions

import asyncio
import pytest


import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are loaded

from GTKUI.Task_manager.task_management import TaskManagement
from tests.test_tool_management_ui import _AtlasStub, _ParentWindowStub


class _SubscriptionStub:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def _register_bus_stub(monkeypatch):
    subscriptions: list[_SubscriptionStub] = []

    def fake_subscribe(self, event_name: str, callback: Any, **_kwargs: Any) -> _SubscriptionStub:
        subscription = _SubscriptionStub(callback)
        subscriptions.append(subscription)
        return subscription

    monkeypatch.setattr(_AtlasStub, "subscribe_event", fake_subscribe, raising=False)
    return subscriptions


def _click(button):
    for signal, callback in getattr(button, "_callbacks", []):
        if signal == "clicked":
            callback(button)


def test_task_management_workspace_loads_and_filters(monkeypatch):
    subscriptions = _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub()

    manager = TaskManagement(atlas, parent)
    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.task_fetches >= 1
    assert atlas.task_catalog_fetches >= 1
    assert subscriptions, "Workspace should subscribe to task events"

    persona_combo = manager._persona_filter_combo
    assert persona_combo is not None
    items = list(getattr(persona_combo, "_items", []))
    assert "Atlas" in items
    assert "Researcher" in items
    assert "Unassigned" in items

    atlas_index = items.index("Atlas")
    persona_combo.set_active(atlas_index)
    manager._on_persona_filter_changed(persona_combo)
    assert {entry.task_id for entry in manager._display_entries} == {"task-1"}
    assert atlas.task_catalog_fetches > 1
    assert all(
        template.get("persona") in ("Atlas", None)
        for template in manager._catalog_entries
    )

    researcher_index = items.index("Researcher")
    persona_combo.set_active(researcher_index)
    manager._on_persona_filter_changed(persona_combo)
    assert {entry.task_id for entry in manager._display_entries} == {"task-2"}

    unassigned_index = items.index("Unassigned")
    persona_combo.set_active(unassigned_index)
    manager._on_persona_filter_changed(persona_combo)
    assert {entry.task_id for entry in manager._display_entries} == {"task-3"}
    assert all(
        template.get("persona") is None for template in manager._catalog_entries
    )


def test_task_management_search_filters(monkeypatch):
    _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub()

    manager = TaskManagement(atlas, parent)
    manager.get_embeddable_widget()

    search_entry = manager._search_entry
    assert search_entry is not None
    search_entry.set_text("deep")
    manager._apply_search_filters()

    assert {entry.task_id for entry in manager._display_entries} == {"task-2"}
    assert atlas.task_search_requests
    payload = atlas.task_search_requests[-1]
    assert payload.get("text") == "deep"

    # Apply metadata filter
    search_entry.set_text("")
    rows = manager._metadata_filter_rows
    assert rows
    key_entry = rows[0].get("key")
    value_entry = rows[0].get("value")
    assert key_entry is not None and value_entry is not None
    key_entry.set_text("persona")
    value_entry.set_text("Atlas")
    manager._apply_search_filters()

    assert {entry.task_id for entry in manager._display_entries} == {"task-1"}
    metadata_payload = atlas.task_search_requests[-1]
    assert metadata_payload.get("metadata") == {"persona": "Atlas"}

    clear_button = manager._search_clear_button
    if clear_button is not None:
        _click(clear_button)

    owner_combo = manager._owner_filter_combo
    assert owner_combo is not None
    owner_items = list(getattr(owner_combo, "_items", []))
    assert "user-1" in owner_items
    owner_combo.set_active(owner_items.index("user-1"))
    manager._on_owner_filter_changed(owner_combo)
    assert all(entry.owner_id == "user-1" for entry in manager._display_entries)
    owner_payload = atlas.task_search_requests[-1]
    assert owner_payload.get("owner_id") == "user-1"

    conversation_combo = manager._conversation_filter_combo
    assert conversation_combo is not None
    conversation_items = list(getattr(conversation_combo, "_items", []))
    assert "conv-2" in conversation_items
    conversation_combo.set_active(conversation_items.index("conv-2"))
    manager._on_conversation_filter_changed(conversation_combo)
    assert {entry.conversation_id for entry in manager._display_entries} == {"conv-2"}
    conversation_payload = atlas.task_search_requests[-1]
    assert conversation_payload.get("conversation_id") == "conv-2"


def test_task_management_transition_updates_state(monkeypatch):
    _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub()

    manager = TaskManagement(atlas, parent)
    manager.get_embeddable_widget()

    active = manager._entry_lookup.get("task-1")
    assert active is not None
    assert active.status == "draft"

    button = manager._primary_action_button
    assert button is not None
    assert getattr(button, "label", "") == "Mark ready"

    _click(button)

    assert atlas.task_transitions
    assert atlas.task_transitions[-1]["target"] == "ready"
    updated = manager._entry_lookup.get("task-1")
    assert updated is not None
    assert updated.status == "ready"
    assert parent.toasts, "Successful transitions should trigger a toast"


def test_task_management_bus_refresh(monkeypatch):
    subscriptions = _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub()

    manager = TaskManagement(atlas, parent)
    manager.get_embeddable_widget()
    initial_fetches = atlas.task_fetches

    assert subscriptions, "Bus subscription should be registered"
    callback = subscriptions[0].callback
    payload = {"task_id": "task-2"}

    if asyncio.iscoroutinefunction(callback):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(callback(payload))
        finally:
            loop.close()
    else:
        result = callback(payload)
        if asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(result)
            finally:
                loop.close()

    assert atlas.task_fetches > initial_fetches, "Task events should trigger a refresh"

    manager._on_close_request()
    assert all(sub.cancelled for sub in subscriptions)
