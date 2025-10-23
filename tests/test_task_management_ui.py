from typing import Any

import types
import sys

sys.modules.setdefault("tests.test_provider_manager", types.ModuleType("tests.test_provider_manager"))
sys.modules.setdefault("tests.test_speech_settings_facade", types.ModuleType("tests.test_speech_settings_facade"))
gi_stub = sys.modules.setdefault("gi", types.ModuleType("gi"))
if not hasattr(gi_stub, "require_version"):
    gi_stub.require_version = lambda *args, **kwargs: None
repository_stub = sys.modules.setdefault("gi.repository", types.ModuleType("gi.repository"))
gi_stub.repository = repository_stub

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

    def fake_subscribe(event_name: str, callback: Any, **_kwargs: Any) -> _SubscriptionStub:
        subscription = _SubscriptionStub(callback)
        subscriptions.append(subscription)
        return subscription

    monkeypatch.setattr(
        "GTKUI.Task_manager.task_management.subscribe_bus_event",
        fake_subscribe,
    )
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
    assert atlas.task_fetches == 1
    assert atlas.task_catalog_fetches == 1
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
