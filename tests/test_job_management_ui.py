import asyncio
import copy
import sys
import types
from typing import Iterable, Mapping

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

execution_module = sys.modules.get("ATLAS.tools.execution")
if execution_module is None:
    execution_module = types.ModuleType("ATLAS.tools.execution")
    execution_module.ToolPolicyDecision = type("ToolPolicyDecision", (), {})
    execution_module.SandboxedToolRunner = type("SandboxedToolRunner", (), {})

    def _stub_function(*_args, **_kwargs):  # pragma: no cover - placeholder
        return None

    for name in (
        "compute_tool_policy_snapshot",
        "use_tool",
        "call_model_with_new_prompt",
        "_freeze_generation_settings",
        "_extract_text_and_audio",
        "_store_assistant_message",
        "_proxy_streaming_response",
        "_resolve_provider_manager",
        "get_required_args",
        "_freeze_metadata",
        "_extract_persona_name",
        "_normalize_persona_allowlist",
        "_join_with_and",
        "_normalize_requires_flags",
        "_coerce_persona_flag_value",
        "_persona_flag_enabled",
        "_collect_missing_flag_requirements",
        "_format_operation_flag_reason",
        "_format_denied_operations_summary",
        "_build_persona_context_snapshot",
        "_has_tool_consent",
        "_request_tool_consent",
        "_evaluate_tool_policy",
        "_get_sandbox_runner",
        "_resolve_tool_timeout_seconds",
        "_generate_idempotency_key",
        "_is_tool_idempotent",
        "_apply_idempotent_retry_backoff",
        "_run_with_timeout",
        "_resolve_function_callable",
    ):
        setattr(execution_module, name, _stub_function)
    sys.modules["ATLAS.tools.execution"] = execution_module


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
                "metadata": {
                    "schedule": {"metadata": {"state": "scheduled"}},
                    "schedule_state": "scheduled",
                },
                "created_at": "2024-01-03T11:00:00Z",
                "updated_at": "2024-01-03T11:00:00Z",
            },
            {
                "id": "job-4",
                "name": "Incident Review",
                "description": "Post-incident analysis",
                "status": "succeeded",
                "owner_id": "owner-3",
                "conversation_id": None,
                "tenant_id": atlas.tenant_id,
                "metadata": {
                    "personas": ["Atlas"],
                    "manifest": {"name": "Incident Review", "persona": "Atlas"},
                },
                "created_at": "2024-01-04T12:00:00Z",
                "updated_at": "2024-01-04T12:00:00Z",
            },
        ]
        self.schedules = {
            "job-1": {
                "schedule_type": "cron",
                "expression": "0 9 * * 1",
                "timezone": "UTC",
                "next_run_at": "2024-01-08T09:00:00Z",
                "metadata": {"state": "scheduled"},
            }
        }
        self.schedule_actions: list[tuple[str, str]] = []
        self.reruns: list[dict[str, object]] = []
        self.run_now_requests: list[dict[str, object]] = []
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
        self.link_events: list[dict[str, object]] = []
        self.unlink_events: list[dict[str, object]] = []
        self.transitions: list[dict[str, object]] = []
        self.created_jobs: list[dict[str, object]] = []

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

    def link_job_task(self, job_id: str, payload: Mapping[str, object], *, context):
        record = {
            "id": f"link-{len(self.link_events) + len(self.linked_tasks.get(job_id, [])) + 1}",
            "task_id": payload.get("task_id"),
            "relationship_type": payload.get("relationship_type"),
            "metadata": copy.deepcopy(
                payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            ),
            "task": {
                "id": payload.get("task_id"),
                "title": f"Task {payload.get('task_id')}",
                "status": "ready",
            },
        }
        bucket = self.linked_tasks.setdefault(job_id, [])
        bucket.append(record)
        event = {
            "job_id": job_id,
            "payload": copy.deepcopy(payload),
            "tenant": context.get("tenant_id") if isinstance(context, Mapping) else None,
        }
        self.link_events.append(event)
        return copy.deepcopy(record)

    def unlink_job_task(
        self,
        job_id: str,
        *,
        context,
        link_id: object | None = None,
        task_id: object | None = None,
    ):
        records = self.linked_tasks.get(job_id, [])
        removed = None
        for item in list(records):
            if link_id is not None and item.get("id") == link_id:
                records.remove(item)
                removed = item
                break
            if task_id is not None and item.get("task_id") == task_id:
                records.remove(item)
                removed = item
                break
        event = {
            "job_id": job_id,
            "link_id": link_id,
            "task_id": task_id,
            "tenant": context.get("tenant_id") if isinstance(context, Mapping) else None,
        }
        self.unlink_events.append(event)
        return copy.deepcopy(removed) if removed is not None else {}

    def pause_job_schedule(self, job_id: str, *, context, expected_updated_at: str | None = None):
        for job in self.jobs:
            if job["id"] == job_id:
                metadata = job.setdefault("metadata", {})
                schedule = metadata.setdefault("schedule", {"metadata": {}})
                schedule_meta = schedule.setdefault("metadata", {})
                schedule_meta["state"] = "paused"
                metadata["schedule_state"] = "paused"
                job["updated_at"] = f"{job_id}-paused"
                schedule_record = self.schedules.setdefault(job_id, {"metadata": {}})
                schedule_record = dict(schedule_record)
                schedule_record.setdefault("schedule_type", "cron")
                schedule_record.setdefault("expression", "0 9 * * 1")
                schedule_record.setdefault("timezone", "UTC")
                schedule_record.setdefault("next_run_at", "2024-01-08T09:00:00Z")
                schedule_record.setdefault("metadata", {})["state"] = "paused"
                self.schedules[job_id] = schedule_record
                self.schedule_actions.append(("pause", job_id))
                payload = copy.deepcopy(job)
                payload["schedule"] = copy.deepcopy(schedule_record)
                return payload
        raise RuntimeError("Job not found")

    def resume_job_schedule(self, job_id: str, *, context, expected_updated_at: str | None = None):
        for job in self.jobs:
            if job["id"] == job_id:
                metadata = job.setdefault("metadata", {})
                schedule = metadata.setdefault("schedule", {"metadata": {}})
                schedule_meta = schedule.setdefault("metadata", {})
                schedule_meta["state"] = "scheduled"
                metadata["schedule_state"] = "scheduled"
                job["status"] = "scheduled"
                job["updated_at"] = f"{job_id}-resumed"
                schedule_record = self.schedules.setdefault(job_id, {"metadata": {}})
                schedule_record = dict(schedule_record)
                schedule_record.setdefault("schedule_type", "cron")
                schedule_record.setdefault("expression", "0 9 * * 1")
                schedule_record.setdefault("timezone", "UTC")
                schedule_record.setdefault("next_run_at", "2024-01-08T09:00:00Z")
                schedule_record.setdefault("metadata", {})["state"] = "scheduled"
                self.schedules[job_id] = schedule_record
                self.schedule_actions.append(("resume", job_id))
                payload = copy.deepcopy(job)
                payload["schedule"] = copy.deepcopy(schedule_record)
                return payload
        raise RuntimeError("Job not found")

    def run_job_now(
        self,
        job_id: str,
        *,
        context,
        expected_updated_at: str | None = None,
    ):
        for job in self.jobs:
            if job["id"] == job_id:
                metadata = job.setdefault("metadata", {})
                schedule = metadata.setdefault("schedule", {"metadata": {}})
                schedule_meta = schedule.setdefault("metadata", {})
                queue_status = {
                    "job_id": f"queue-{len(self.run_now_requests) + 1}",
                    "state": "queued",
                }
                schedule_meta["last_run"] = dict(queue_status)
                job["updated_at"] = f"{job_id}-run-now"
                record = {
                    "job_id": job_id,
                    "expected": expected_updated_at,
                    "tenant": context.get("tenant_id") if isinstance(context, Mapping) else None,
                }
                self.run_now_requests.append(record)
                payload = copy.deepcopy(job)
                payload.setdefault("metadata", {})
                payload["metadata"].setdefault("schedule", copy.deepcopy(schedule))
                payload["queue_status"] = dict(queue_status)
                return payload
        raise RuntimeError("Job not found")

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

    def create_job(self, payload: Mapping[str, object], *, context):
        record = dict(payload)
        record.setdefault("metadata", {})
        if not isinstance(record["metadata"], Mapping):
            record["metadata"] = {}
        job_index = len(self.jobs) + 1
        record.setdefault("id", f"job-{job_index}")
        record.setdefault("status", "draft")
        record["tenant_id"] = context.get("tenant_id")
        record["metadata"] = copy.deepcopy(record["metadata"])
        self.jobs.append(copy.deepcopy(record))
        self.created_jobs.append(copy.deepcopy(record))
        return copy.deepcopy(record)

    def rerun_job(
        self,
        job_id: str,
        *,
        context,
        expected_updated_at: str | None = None,
    ):
        for job in self.jobs:
            if job["id"] == job_id:
                job["status"] = "running"
                job["updated_at"] = f"{job_id}-rerun"
                runs = job.setdefault("runs", [])
                run_record = {
                    "id": f"run-{len(self.reruns) + 1}",
                    "status": "running",
                }
                runs.append(run_record)
                record = {
                    "job_id": job_id,
                    "expected": expected_updated_at,
                }
                self.reruns.append(record)
                return copy.deepcopy(job)
        raise RuntimeError("Job not found")


class _AtlasStub:
    def __init__(self) -> None:
        self.tenant_id = "tenant-1"
        self.job_fetches = 0
        self.job_detail_fetches = 0
        self.server = _JobServerStub(self)

    def link_job_task(
        self,
        job_id: str,
        task_id: str,
        *,
        relationship_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ):
        payload = {"task_id": task_id}
        if relationship_type:
            payload["relationship_type"] = relationship_type
        if metadata:
            payload["metadata"] = dict(metadata)
        return self.server.link_job_task(
            job_id,
            payload,
            context={"tenant_id": self.tenant_id},
        )

    def unlink_job_task(
        self,
        job_id: str,
        *,
        link_id: object | None = None,
        task_id: object | None = None,
    ):
        return self.server.unlink_job_task(
            job_id,
            context={"tenant_id": self.tenant_id},
            link_id=link_id,
            task_id=task_id,
        )

    def create_job(
        self,
        name: str,
        *,
        description: str | None = None,
        personas: Iterable[str] | None = None,
        schedule: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
    ):
        payload: dict[str, object] = {"name": name}
        if description:
            payload["description"] = description
        metadata_payload: dict[str, object] = {}
        if metadata:
            metadata_payload.update(metadata)
        if personas:
            metadata_payload.setdefault("personas", list(personas))
        if schedule:
            metadata_payload["schedule"] = dict(schedule)
        if metadata_payload:
            payload["metadata"] = metadata_payload
        return self.server.create_job(payload, context={"tenant_id": self.tenant_id})


class _ParentWindowStub:
    def __init__(self, atlas: _AtlasStub) -> None:
        self.ATLAS = atlas
        self.errors: list[str] = []
        self.toasts: list[str] = []
        self.prompt_result: Mapping[str, object] | None = None
        self.new_job_details: Mapping[str, object] | None = None

    def show_error_dialog(self, message: str) -> None:
        self.errors.append(message)

    def show_success_toast(self, message: str) -> None:
        self.toasts.append(message)

    def prompt_link_job_task(self, _job_id: str):
        return copy.deepcopy(self.prompt_result)

    def prompt_new_job_details(self):
        return copy.deepcopy(self.new_job_details)

    def _transition_job(self, job_id: str, target_status: str, updated_at: str | None):
        return self.ATLAS.server.transition_job(
            job_id,
            target_status,
            context={"tenant_id": self.ATLAS.tenant_id},
            expected_updated_at=updated_at,
        )

    def start_job(
        self,
        job_id: str,
        current_status: str,
        updated_at: str | None,
        *,
        mode: str = "auto",
    ):
        status = (current_status or "").lower()
        normalized_mode = (mode or "auto").lower()
        if normalized_mode == "run_now":
            scheduled_payload = None
            if status == "draft":
                scheduled_payload = self._transition_job(job_id, "scheduled", updated_at)
                if isinstance(scheduled_payload, Mapping):
                    updated_at = scheduled_payload.get("updated_at")
            run_now = getattr(self.ATLAS.server, "run_job_now", None)
            if callable(run_now):
                payload = run_now(
                    job_id,
                    context={"tenant_id": self.ATLAS.tenant_id},
                    expected_updated_at=updated_at,
                )
                self.show_success_toast("Job run queued")
                return payload
            if scheduled_payload is not None:
                return scheduled_payload
            target = "running"
        elif status == "draft":
            target = "scheduled"
        elif normalized_mode == "resume":
            return self.resume_job(job_id, current_status, updated_at)
        else:
            target = "running"
        payload = self._transition_job(job_id, target, updated_at)
        self.show_success_toast(f"Job moved to {target.title()}")
        return payload

    def resume_job(self, job_id: str, current_status: str, updated_at: str | None):
        resume_schedule = getattr(self.ATLAS.server, "resume_job_schedule", None)
        if callable(resume_schedule):
            payload = resume_schedule(
                job_id,
                context={"tenant_id": self.ATLAS.tenant_id},
                expected_updated_at=updated_at,
            )
            self.show_success_toast("Job schedule resumed")
            return payload
        payload = self._transition_job(job_id, "scheduled", updated_at)
        self.show_success_toast("Job moved to Scheduled")
        return payload

    def pause_job(self, job_id: str, current_status: str, updated_at: str | None):
        pause_schedule = getattr(self.ATLAS.server, "pause_job_schedule", None)
        if callable(pause_schedule):
            payload = pause_schedule(
                job_id,
                context={"tenant_id": self.ATLAS.tenant_id},
                expected_updated_at=updated_at,
            )
            self.show_success_toast("Job schedule paused")
            return payload
        payload = self._transition_job(job_id, "cancelled", updated_at)
        self.show_success_toast("Job moved to Cancelled")
        return payload

    def rerun_job(self, job_id: str, current_status: str, updated_at: str | None):
        rerun_job = getattr(self.ATLAS.server, "rerun_job", None)
        if callable(rerun_job):
            payload = rerun_job(
                job_id,
                context={"tenant_id": self.ATLAS.tenant_id},
                expected_updated_at=updated_at,
            )
        else:
            payload = self._transition_job(job_id, "running", updated_at)
        self.show_success_toast("Job rerun queued")
        return payload


class _SubscriptionStub:
    def __init__(self, event_name: str, callback):
        self.event_name = event_name
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _SubscriptionStore(list[_SubscriptionStub]):
    def __init__(self) -> None:
        super().__init__()
        self.requested_events: list[str] = []


def _register_bus_stub(monkeypatch):
    subscriptions = _SubscriptionStore()

    def fake_subscribe(self, event_name: str, callback, **_kwargs):
        subscriptions.requested_events.append(event_name)
        subscription = _SubscriptionStub(event_name, callback)
        subscriptions.append(subscription)
        return subscription

    monkeypatch.setattr(_AtlasStub, "subscribe_event", fake_subscribe, raising=False)
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
    assert {
        "job.created",
        "job.updated",
        "job.status_changed",
    }.issubset(set(subscriptions.requested_events))

    persona_combo = manager._persona_filter_combo
    assert persona_combo is not None
    persona_items = list(getattr(persona_combo, "_items", []))
    assert "Atlas" in persona_items
    assert "Unassigned" in persona_items

    atlas_index = persona_items.index("Atlas")
    persona_combo.set_active(atlas_index)
    manager._on_persona_filter_changed(persona_combo)
    assert {entry.job_id for entry in manager._display_entries} == {"job-1", "job-4"}

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
    assert atlas.server.schedule_actions[-1] == ("pause", "job-1")
    paused_job = next(job for job in atlas.server.jobs if job["id"] == "job-1")
    assert paused_job["metadata"].get("schedule_state") == "paused"

    manager._select_job("job-1")
    rerun_button = manager._rerun_button
    assert rerun_button is not None and not rerun_button.visible

    manager._select_job("job-4")
    rerun_button = manager._rerun_button
    assert rerun_button is not None and rerun_button.visible
    rerun_before = list(atlas.server.reruns)
    _click(rerun_button)
    assert atlas.server.reruns != rerun_before
    assert atlas.server.reruns[-1]["job_id"] == "job-4"
    assert any("Job rerun queued" in toast for toast in parent.toasts)

    manager._select_job("job-3")
    start_button = manager._start_button
    assert start_button is not None and start_button.visible
    before = list(atlas.server.transitions)
    _click(start_button)
    assert atlas.server.transitions == before, "No transition should occur without confirmation"
    choices = dict(manager._start_confirmation_choices)
    assert "run_now" in choices and "resume" in choices
    choices["run_now"]()
    assert atlas.server.run_now_requests and atlas.server.run_now_requests[-1]["job_id"] == "job-3"
    assert atlas.server.transitions == before
    entry = manager._entry_lookup["job-3"]
    assert entry.metadata.get("schedule", {}).get("metadata", {}).get("last_run", {}).get("state") == "queued"
    assert not manager._start_confirmation_choices

    atlas.server.pause_job_schedule(
        "job-3", context={"tenant_id": atlas.tenant_id}, expected_updated_at=None
    )
    manager._refresh_state()
    manager._select_job("job-3")
    _click(start_button)
    paused_choices = dict(manager._start_confirmation_choices)
    assert "resume" in paused_choices
    paused_choices["resume"]()
    assert atlas.server.schedule_actions[-1] == ("resume", "job-3")
    resumed_job = next(job for job in atlas.server.jobs if job["id"] == "job-3")
    assert resumed_job["metadata"].get("schedule_state") == "scheduled"


def test_status_filter_set_active_triggers_single_refresh(monkeypatch):
    _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)

    manager = JobManagement(atlas, parent)
    manager.get_embeddable_widget()

    status_combo = manager._status_filter_combo
    assert status_combo is not None
    status_items = list(getattr(status_combo, "_items", []))
    assert "Running" in status_items
    running_index = status_items.index("Running")

    initial_fetches = atlas.job_fetches
    status_combo.set_active(running_index)

    assert atlas.job_fetches == initial_fetches + 1
    assert manager._status_filter == "running"
    assert {entry.job_id for entry in manager._display_entries} == {"job-2"}


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


def test_job_management_job_created_event(monkeypatch):
    subscriptions = _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)

    manager = JobManagement(atlas, parent)
    manager.get_embeddable_widget()

    initial_fetches = atlas.job_fetches
    job_created = [sub for sub in subscriptions if sub.event_name == "job.created"]
    assert job_created, "job.created subscription should be registered"

    callback = job_created[0].callback
    payload = {"id": "job-new"}

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

    assert atlas.job_fetches > initial_fetches, "job.created should trigger refresh"


def test_job_management_link_and_unlink_tasks(monkeypatch):
    _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)

    manager = JobManagement(atlas, parent)
    manager.get_embeddable_widget()
    manager._select_job("job-1")

    parent.prompt_result = {
        "task_id": "task-9",
        "relationship_type": "relates",
        "metadata": {"summary": "Draft brief"},
    }

    link_button = manager._link_task_button
    assert link_button is not None
    _click(link_button)

    assert any("Task task-9" in badge for badge in manager._current_linked_task_badges)
    assert parent.toasts and parent.toasts[-1] == "Task linked to job."

    actions = list(manager._linked_task_action_lookup)
    assert actions, "Linked task actions should be tracked"
    _, unlink_button = actions[-1]
    _click(unlink_button)

    assert all("task-9" not in badge for badge in manager._current_linked_task_badges)
    assert parent.toasts and parent.toasts[-1] == "Task unlinked from job."
    assert atlas.server.link_events, "Link route should be invoked"
    assert atlas.server.unlink_events, "Unlink route should be invoked"


def test_job_management_link_task_error(monkeypatch):
    _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)

    def failing_link(*_args, **_kwargs):
        raise RuntimeError("backend failure")

    atlas.link_job_task = failing_link  # type: ignore[assignment]

    manager = JobManagement(atlas, parent)
    manager.get_embeddable_widget()
    manager._select_job("job-1")

    parent.prompt_result = {"task_id": "task-err"}

    link_button = manager._link_task_button
    assert link_button is not None
    _click(link_button)

    assert parent.errors, "Errors should surface to the user"
    assert all("task-err" not in badge for badge in manager._current_linked_task_badges)


def test_job_management_create_job(monkeypatch):
    _register_bus_stub(monkeypatch)
    atlas = _AtlasStub()
    parent = _ParentWindowStub(atlas)
    parent.new_job_details = {
        "name": "Nightly sync",
        "description": "Sync CRM data",
        "personas": ["Atlas", "Researcher"],
        "schedule": {"frequency": "daily", "interval": "1"},
    }

    manager = JobManagement(atlas, parent)
    manager.get_embeddable_widget()

    new_button = manager._new_job_button
    assert new_button is not None
    _click(new_button)

    assert atlas.server.created_jobs, "Create job route should be invoked"
    created = atlas.server.created_jobs[-1]
    assert created["name"] == "Nightly sync"
    assert created["metadata"].get("schedule", {}).get("frequency") == "daily"
    assert created["metadata"].get("personas") == ["Atlas", "Researcher"]
    assert manager._active_job == created["id"], "New job should be focused"
    assert parent.toasts and parent.toasts[-1] == "Job created."
