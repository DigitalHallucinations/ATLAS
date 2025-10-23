import asyncio
import copy
import sys
import types

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

import pytest

import tests.test_chat_async_helper  # noqa: F401 - ensure GTK stubs are loaded

from GTKUI.Job_manager.job_management import JobManagement


class _JobServerStub:
    def __init__(self, atlas) -> None:
        self._atlas = atlas
        self.jobs = [
            {
                "id": "job-1",
                "name": "Weekly Summary",
                "description": "Compile weekly report",
                "status": "draft",
                "owner_id": "owner-1",
                "conversation_id": "conversation-1",
                "tenant_id": atlas.tenant_id,
                "metadata": {
                    "personas": ["Atlas"],
                    "recurrence": {"frequency": "weekly", "interval": "1"},
                    "escalation_policy": {
                        "level": "Ops Lead",
                        "contact": "ops@example.com",
                        "timeframe": "48h",
                    },
                },
                "created_at": "2024-01-01T09:00:00Z",
                "updated_at": "2024-01-01T09:00:00Z",
            },
            {
                "id": "job-2",
                "name": "Launch Checklist",
                "description": "Prepare launch assets",
                "status": "running",
                "owner_id": "owner-2",
                "conversation_id": None,
                "tenant_id": atlas.tenant_id,
                "metadata": {"persona": "Researcher"},
                "created_at": "2024-01-02T10:00:00Z",
                "updated_at": "2024-01-02T10:00:00Z",
            },
            {
                "id": "job-3",
                "name": "Ad-hoc Review",
                "description": "One-off governance review",
                "status": "scheduled",
                "owner_id": None,
                "conversation_id": None,
                "tenant_id": atlas.tenant_id,
                "metadata": {},
                "created_at": "2024-01-03T11:00:00Z",
                "updated_at": "2024-01-03T11:00:00Z",
            },
        ]
        self.schedules = {
            "job-1": {
                "schedule_type": "cron",
                "expression": "0 9 * * 1",
                "timezone": "UTC",
                "next_run_at": "2024-01-08T09:00:00Z",
            }
        }
        self.linked_tasks = {
            "job-1": [
                {
                    "id": "link-1",
                    "task_id": "task-1",
                    "relationship_type": "blocks",
                    "metadata": {"summary": "Discovery"},
                    "task": {
                        "id": "task-1",
                        "title": "Initial research",
                        "status": "ready",
                    },
                }
            ]
        }
        self.transitions: list[dict[str, object]] = []

    def list_jobs(self, params=None, *, context):
        self._atlas.job_fetches += 1
        params = params or {}
        status = params.get("status")
        if status:
            items = [job for job in self.jobs if job["status"] == status]
        else:
            items = list(self.jobs)
        return {"items": [copy.deepcopy(job) for job in items]}

    def get_job(
        self,
        job_id: str,
        *,
        context,
        include_schedule: bool = False,
        include_runs: bool = False,
        include_events: bool = False,
    ):
        self._atlas.job_detail_fetches += 1
        for job in self.jobs:
            if job["id"] == job_id:
                payload = copy.deepcopy(job)
                if include_schedule and job_id in self.schedules:
                    payload["schedule"] = copy.deepcopy(self.schedules[job_id])
                return payload
        raise RuntimeError("Job not found")

    def list_job_tasks(self, job_id: str, *, context):
        tasks = self.linked_tasks.get(job_id, [])
        return [copy.deepcopy(entry) for entry in tasks]

    def transition_job(
        self,
        job_id: str,
        target_status: str,
        *,
        context,
        expected_updated_at: str | None = None,
    ):
        for job in self.jobs:
            if job["id"] == job_id:
                job["status"] = target_status
                job["updated_at"] = f"{target_status}-timestamp"
                record = {
                    "job_id": job_id,
                    "target": target_status,
                    "expected": expected_updated_at,
                }
                self.transitions.append(record)
                return copy.deepcopy(job)
        raise RuntimeError("Job not found")


class _AtlasStub:
    def __init__(self) -> None:
        self.tenant_id = "tenant-1"
        self.job_fetches = 0
        self.job_detail_fetches = 0
        self.server = _JobServerStub(self)


class _ParentWindowStub:
    def __init__(self, atlas: _AtlasStub) -> None:
        self.ATLAS = atlas
        self.errors: list[str] = []
        self.toasts: list[str] = []

    def show_error_dialog(self, message: str) -> None:
        self.errors.append(message)

    def show_success_toast(self, message: str) -> None:
        self.toasts.append(message)

    def _transition_job(self, job_id: str, target_status: str, updated_at: str | None):
        return self.ATLAS.server.transition_job(
            job_id,
            target_status,
            context={"tenant_id": self.ATLAS.tenant_id},
            expected_updated_at=updated_at,
        )

    def start_job(self, job_id: str, current_status: str, updated_at: str | None):
        target = "scheduled" if (current_status or "").lower() == "draft" else "running"
        payload = self._transition_job(job_id, target, updated_at)
        self.show_success_toast(f"Job moved to {target.title()}")
        return payload

    def pause_job(self, job_id: str, current_status: str, updated_at: str | None):
        payload = self._transition_job(job_id, "cancelled", updated_at)
        self.show_success_toast("Job moved to Cancelled")
        return payload

    def rerun_job(self, job_id: str, current_status: str, updated_at: str | None):
        payload = self._transition_job(job_id, "running", updated_at)
        self.show_success_toast("Job moved to Running")
        return payload


class _SubscriptionStub:
    def __init__(self, callback):
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def _register_bus_stub(monkeypatch):
    subscriptions: list[_SubscriptionStub] = []

    def fake_subscribe(event_name: str, callback, **_kwargs):
        subscription = _SubscriptionStub(callback)
        subscriptions.append(subscription)
        return subscription

    monkeypatch.setattr(
        "GTKUI.Job_manager.job_management.subscribe_bus_event",
        fake_subscribe,
    )
    return subscriptions


def _click(button):
    for signal, callback in getattr(button, "_callbacks", []):
        if signal == "clicked":
            callback(button)


def test_job_management_filters_and_actions(monkeypatch):
    subscriptions = _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)

    manager = JobManagement(atlas, parent)
    widget = manager.get_embeddable_widget()
    assert widget is not None
    assert atlas.job_fetches == 1
    assert subscriptions, "Workspace should subscribe to job events"

    persona_combo = manager._persona_filter_combo
    assert persona_combo is not None
    persona_items = list(getattr(persona_combo, "_items", []))
    assert "Atlas" in persona_items
    assert "Unassigned" in persona_items

    atlas_index = persona_items.index("Atlas")
    persona_combo.set_active(atlas_index)
    manager._on_persona_filter_changed(persona_combo)
    assert {entry.job_id for entry in manager._display_entries} == {"job-1"}

    unassigned_index = persona_items.index("Unassigned")
    persona_combo.set_active(unassigned_index)
    manager._on_persona_filter_changed(persona_combo)
    assert {entry.job_id for entry in manager._display_entries} == {"job-3"}

    status_combo = manager._status_filter_combo
    assert status_combo is not None
    status_items = list(getattr(status_combo, "_items", []))
    running_index = status_items.index("Running")
    status_combo.set_active(running_index)
    manager._on_status_filter_changed(status_combo)
    assert {entry.job_id for entry in manager._display_entries} == {"job-2"}

    recurrence_combo = manager._recurrence_filter_combo
    assert recurrence_combo is not None
    recurrence_items = list(getattr(recurrence_combo, "_items", []))
    assert "Recurring" in recurrence_items
    recurring_index = recurrence_items.index("Recurring")
    recurrence_combo.set_active(recurring_index)
    manager._on_recurrence_filter_changed(recurrence_combo)
    assert {entry.job_id for entry in manager._display_entries} == {"job-1"}

    manager._select_job("job-1")
    assert atlas.job_detail_fetches > 0
    assert any("Type: cron" == badge for badge in manager._current_schedule_badges)
    assert any("Next run: 2024-01-08T09:00:00Z" == badge for badge in manager._current_schedule_badges)
    assert any("Initial research (Ready) – blocks – Discovery" == badge for badge in manager._current_linked_task_badges)
    assert any("Level: Ops Lead" == badge for badge in manager._current_escalation_badges)

    start_button = manager._start_button
    assert start_button is not None and start_button.visible
    _click(start_button)
    assert atlas.server.transitions[-1]["target"] == "scheduled"
    assert parent.toasts, "Successful transitions should trigger a toast"

    manager._select_job("job-1")
    pause_button = manager._pause_button
    assert pause_button is not None and pause_button.visible
    _click(pause_button)
    assert atlas.server.transitions[-1]["target"] == "cancelled"

    manager._select_job("job-1")
    rerun_button = manager._rerun_button
    assert rerun_button is not None and rerun_button.visible
    _click(rerun_button)
    assert atlas.server.transitions[-1]["target"] == "running"


def test_job_management_bus_refresh(monkeypatch):
    subscriptions = _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)

    manager = JobManagement(atlas, parent)
    manager.get_embeddable_widget()
    initial_fetches = atlas.job_fetches

    assert subscriptions, "Bus subscription should be registered"
    callback = subscriptions[0].callback
    payload = {"job_id": "job-2"}

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

    assert atlas.job_fetches > initial_fetches, "Job events should trigger a refresh"

    manager._on_close_request()
    assert all(sub.cancelled for sub in subscriptions)
