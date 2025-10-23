"""Tests for persona analytics storage and aggregation."""

from datetime import datetime, timedelta, timezone
import threading

import pytest

from modules.analytics.persona_metrics import (
    PersonaMetricEvent,
    PersonaMetricsStore,
    TaskLifecycleEvent,
    JobLifecycleEvent,
    get_task_lifecycle_metrics,
    get_job_lifecycle_metrics,
    record_task_lifecycle_event,
    record_job_lifecycle_event,
)


class DummyConfig:
    def __init__(self, app_root):
        self._app_root = str(app_root)

    def get_app_root(self):
        return self._app_root


@pytest.fixture
def metrics_store(tmp_path):
    storage_path = tmp_path / "metrics.json"
    store = PersonaMetricsStore(storage_path=str(storage_path))
    store.reset()
    return store


def test_persona_metrics_aggregation(metrics_store):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="web_search",
            success=True,
            latency_ms=120.0,
            timestamp=base,
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="web_search",
            success=False,
            latency_ms=250.0,
            timestamp=base + timedelta(minutes=5),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="calculator",
            success=True,
            latency_ms=75.0,
            timestamp=base + timedelta(minutes=10),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="summarize",
            success=True,
            latency_ms=180.0,
            timestamp=base + timedelta(minutes=12),
            category="skill",
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="summarize",
            success=False,
            latency_ms=220.0,
            timestamp=base + timedelta(minutes=14),
            category="skill",
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Other",
            tool="web_search",
            success=True,
            latency_ms=15.0,
            timestamp=base,
        )
    )

    metrics = metrics_store.get_metrics("Atlas")
    assert metrics["totals"]["calls"] == 3
    assert metrics["totals"]["success"] == 2
    assert metrics["totals"]["failure"] == 1
    assert metrics["success_rate"] == pytest.approx(2 / 3, rel=1e-6)
    assert metrics["average_latency_ms"] == pytest.approx(
        (120.0 + 250.0 + 75.0) / 3, rel=1e-6
    )

    tool_breakdown = {entry["tool"]: entry for entry in metrics["totals_by_tool"]}
    assert tool_breakdown["web_search"]["calls"] == 2
    assert tool_breakdown["web_search"]["success"] == 1
    assert tool_breakdown["calculator"]["success_rate"] == pytest.approx(1.0)

    recent_tools = [entry["tool"] for entry in metrics["recent"]]
    assert recent_tools[0] == "calculator"
    assert recent_tools[-1] == "web_search"

    skill_metrics = metrics["skills"]
    assert skill_metrics["totals"]["calls"] == 2
    assert skill_metrics["totals"]["success"] == 1
    assert skill_metrics["totals"]["failure"] == 1
    assert skill_metrics["success_rate"] == pytest.approx(0.5)
    skill_breakdown = {
        entry["skill"]: entry for entry in skill_metrics["totals_by_skill"]
    }
    assert skill_breakdown["summarize"]["calls"] == 2
    assert skill_metrics["recent"][0]["skill"] == "summarize"
    assert skill_metrics["recent"][0]["success"] is False

    filtered = metrics_store.get_metrics(
        "Atlas",
        start=base + timedelta(minutes=6),
    )
    assert filtered["totals"]["calls"] == 1
    assert filtered["recent"][0]["tool"] == "calculator"


def test_record_event_concurrent_persistence(metrics_store):
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    events = [
        PersonaMetricEvent(
            persona="Atlas",
            tool="alpha",
            success=True,
            latency_ms=50.0,
            timestamp=base,
        ),
        PersonaMetricEvent(
            persona="Atlas",
            tool="beta",
            success=False,
            latency_ms=75.0,
            timestamp=base + timedelta(seconds=1),
        ),
    ]

    barrier = threading.Barrier(len(events) + 1)

    threads = []

    def worker(event):
        barrier.wait()
        metrics_store.record_event(event)

    for event in events:
        thread = threading.Thread(target=worker, args=(event,))
        threads.append(thread)
        thread.start()

    barrier.wait()

    for thread in threads:
        thread.join()

    metrics = metrics_store.get_metrics("Atlas")
    assert metrics["totals"]["calls"] == 2
    assert metrics["totals"]["success"] == 1
    tools = {entry["tool"] for entry in metrics["recent"]}
    assert tools == {"alpha", "beta"}


def test_task_lifecycle_metrics_aggregation(metrics_store):
    base = datetime(2024, 3, 1, 8, 0, tzinfo=timezone.utc)
    metrics_store.record_task_event(
        TaskLifecycleEvent(
            task_id="task-1",
            event="created",
            persona="Atlas",
            tenant_id="tenant-a",
            to_status="ready",
            success=None,
            latency_ms=None,
            reassignments=0,
            timestamp=base,
            metadata={"source": "unit"},
        )
    )
    metrics_store.record_task_event(
        TaskLifecycleEvent(
            task_id="task-1",
            event="completed",
            persona="Atlas",
            tenant_id="tenant-a",
            from_status="review",
            to_status="done",
            success=True,
            latency_ms=540.0,
            reassignments=0,
            timestamp=base + timedelta(minutes=30),
        )
    )
    metrics_store.record_task_event(
        TaskLifecycleEvent(
            task_id="task-1",
            event="reassigned",
            persona="Atlas",
            tenant_id="tenant-a",
            from_status="done",
            to_status="done",
            success=None,
            latency_ms=None,
            reassignments=1,
            timestamp=base + timedelta(hours=1),
        )
    )
    metrics_store.record_task_event(
        TaskLifecycleEvent(
            task_id="other",
            event="cancelled",
            persona="Other",
            tenant_id="tenant-b",
            from_status="review",
            to_status="cancelled",
            success=False,
            latency_ms=200.0,
            reassignments=0,
            timestamp=base + timedelta(hours=2),
        )
    )

    metrics = metrics_store.get_task_metrics(persona="Atlas")
    assert metrics["totals"]["events"] == 3
    assert metrics["totals"]["reassignments"] == 1
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["average_latency_ms"] == pytest.approx(540.0)

    status_totals = {entry["status"]: entry for entry in metrics["status_totals"]}
    assert status_totals["done"]["events"] == 2

    task_summary = {entry["task_id"]: entry for entry in metrics["tasks"]}
    assert task_summary["task-1"]["reassignments"] == 1
    assert task_summary["task-1"]["last_status"] == "done"

    recent_first = metrics["recent"][0]
    assert recent_first["event"] == "reassigned"
    assert recent_first["task_id"] == "task-1"


def test_record_task_lifecycle_event_publishes(monkeypatch, metrics_store):
    published = []

    def _fake_publish(topic, payload, **kwargs):
        published.append((topic, payload, kwargs))

    monkeypatch.setattr(
        "modules.analytics.persona_metrics.publish_bus_event",
        _fake_publish,
    )
    monkeypatch.setattr(
        "modules.analytics.persona_metrics._get_store",
        lambda config_manager=None: metrics_store,
    )

    timestamp = datetime(2024, 4, 5, 9, 30, tzinfo=timezone.utc)
    record_task_lifecycle_event(
        task_id="task-123",
        event="completed",
        persona="Atlas",
        tenant_id="tenant-x",
        from_status="review",
        to_status="done",
        success=True,
        latency_ms=1250.0,
        reassignments=0,
        timestamp=timestamp,
        metadata={"source": "test"},
    )

    assert published and published[0][0] == "task_metrics.lifecycle"
    payload = published[0][1]
    assert payload["task_id"] == "task-123"
    assert payload["success"] is True

    metrics = get_task_lifecycle_metrics(persona="Atlas")
    assert metrics["totals"]["events"] == 1
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["recent"][0]["metadata"]["source"] == "test"


def test_job_lifecycle_metrics_aggregation(metrics_store):
    base = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    metrics_store.record_job_event(
        JobLifecycleEvent(
            job_id="job-1",
            event="created",
            persona="Atlas",
            tenant_id="tenant-a",
            to_status="draft",
            success=None,
            timestamp=base,
            metadata={"owner_id": "owner-1"},
        )
    )
    metrics_store.record_job_event(
        JobLifecycleEvent(
            job_id="job-1",
            event="completed",
            persona="Atlas",
            tenant_id="tenant-a",
            from_status="running",
            to_status="succeeded",
            success=True,
            latency_ms=1200.0,
            timestamp=base + timedelta(hours=1),
            metadata={"sla_met": True},
        )
    )
    metrics_store.record_job_event(
        JobLifecycleEvent(
            job_id="job-2",
            event="failed",
            persona="Atlas",
            tenant_id="tenant-a",
            from_status="running",
            to_status="failed",
            success=False,
            latency_ms=1800.0,
            timestamp=base + timedelta(hours=2),
            metadata={"sla_breached": True},
        )
    )
    metrics_store.record_job_event(
        JobLifecycleEvent(
            job_id="other",
            event="completed",
            persona="Other",
            tenant_id="tenant-b",
            to_status="succeeded",
            success=True,
            timestamp=base + timedelta(hours=3),
        )
    )

    metrics = metrics_store.get_job_metrics(persona="Atlas")
    assert metrics["totals"]["events"] == 3
    assert metrics["totals"]["success_events"] == 1
    assert metrics["totals"]["failure_events"] == 1
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["throughput_per_hour"] == pytest.approx(2.0)

    status_totals = {entry["status"]: entry for entry in metrics["status_totals"]}
    assert status_totals["succeeded"]["success"] == 1
    assert status_totals["failed"]["failure"] == 1

    job_summary = {entry["job_id"]: entry for entry in metrics["jobs"]}
    assert job_summary["job-1"]["last_status"] == "succeeded"
    assert job_summary["job-2"]["failure"] == 1

    sla_summary = metrics["sla"]
    assert sla_summary["checks"] == 2
    assert sla_summary["breaches"] == 1
    assert sla_summary["adherence_rate"] == pytest.approx(0.5)

    recent_first = metrics["recent"][0]
    assert recent_first["event"] == "failed"
    assert recent_first["job_id"] == "job-2"


def test_record_job_lifecycle_event_publishes(monkeypatch, metrics_store):
    published = []

    def _fake_publish(topic, payload, **kwargs):
        published.append((topic, payload, kwargs))

    monkeypatch.setattr(
        "modules.analytics.persona_metrics.publish_bus_event",
        _fake_publish,
    )
    monkeypatch.setattr(
        "modules.analytics.persona_metrics._get_store",
        lambda config_manager=None: metrics_store,
    )

    timestamp = datetime(2024, 6, 10, 15, 45, tzinfo=timezone.utc)
    record_job_lifecycle_event(
        job_id="job-456",
        event="completed",
        persona="Atlas",
        tenant_id="tenant-y",
        from_status="running",
        to_status="succeeded",
        success=True,
        latency_ms=640.0,
        timestamp=timestamp,
        metadata={"sla_met": True},
    )

    assert published and published[0][0] == "jobs.metrics.lifecycle"
    payload = published[0][1]
    assert payload["job_id"] == "job-456"
    assert payload["success"] is True

    metrics = get_job_lifecycle_metrics(persona="Atlas")
    assert metrics["totals"]["events"] == 1
    assert metrics["success_rate"] == pytest.approx(1.0)
    assert metrics["recent"][0]["metadata"]["sla_met"] is True

def test_persona_metrics_server_endpoint(tmp_path):
    pytest.importorskip("jsonschema")
    pytest.importorskip("sqlalchemy")
    from modules.Server.routes import AtlasServer

    app_root = tmp_path / "app"
    analytics_dir = app_root / "modules" / "analytics"
    analytics_dir.mkdir(parents=True)

    store = PersonaMetricsStore(app_root=str(app_root))
    store.reset()
    timestamp = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="web_search",
            success=True,
            latency_ms=90.0,
            timestamp=timestamp,
        )
    )
    store.record_event(
        PersonaMetricEvent(
            persona="Atlas",
            tool="fact_check",
            success=False,
            latency_ms=140.0,
            timestamp=timestamp + timedelta(minutes=1),
            category="skill",
        )
    )

    config = DummyConfig(app_root)
    server = AtlasServer(config_manager=config)

    payload = server.get_persona_metrics("Atlas")
    assert payload["totals"]["calls"] == 1
    assert payload["recent"][0]["tool"] == "web_search"
    assert payload["skills"]["totals"]["calls"] == 1
    assert payload["skills"]["recent"][0]["skill"] == "fact_check"

    skill_payload = server.get_persona_metrics("Atlas", metric_type="skill")
    assert skill_payload["category"] == "skill"
    assert skill_payload["totals"]["calls"] == 1
    assert skill_payload["recent"][0]["skill"] == "fact_check"

    response = server.handle_request(
        "/personas/Atlas/analytics",
        method="GET",
        query={"start": timestamp.isoformat(), "limit": "5"},
    )
    assert response["totals"]["calls"] == 1
    assert response["recent"][0]["tool"] == "web_search"

    skill_response = server.handle_request(
        "/personas/Atlas/analytics",
        method="GET",
        query={"type": "skill", "limit": "5"},
    )
    assert skill_response["category"] == "skill"
    assert skill_response["totals"]["calls"] == 1
    assert skill_response["recent"][0]["skill"] == "fact_check"
