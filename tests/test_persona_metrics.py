"""Tests for persona analytics storage and aggregation."""

from datetime import datetime, timedelta, timezone
import threading
from typing import Optional

import pytest

from modules.analytics.persona_metrics import (
    LifecycleEvent,
    PersonaMetricEvent,
    PersonaMetricsStore,
    get_job_lifecycle_metrics,
    get_task_lifecycle_metrics,
    record_job_lifecycle_event,
    record_task_lifecycle_event,
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


def test_persona_comparison_summary_rankings(metrics_store):
    base = datetime(2024, 4, 1, tzinfo=timezone.utc)

    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Alpha",
            tool="insight",
            success=True,
            latency_ms=50.0,
            timestamp=base,
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Alpha",
            tool="insight",
            success=True,
            latency_ms=40.0,
            timestamp=base + timedelta(minutes=1),
        )
    )

    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Beta",
            tool="insight",
            success=True,
            latency_ms=200.0,
            timestamp=base + timedelta(minutes=2),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Beta",
            tool="insight",
            success=True,
            latency_ms=180.0,
            timestamp=base + timedelta(minutes=3),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Beta",
            tool="insight",
            success=False,
            latency_ms=220.0,
            timestamp=base + timedelta(minutes=4),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Beta",
            tool="insight",
            success=True,
            latency_ms=190.0,
            timestamp=base + timedelta(minutes=5),
        )
    )

    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Gamma",
            tool="insight",
            success=False,
            latency_ms=80.0,
            timestamp=base + timedelta(minutes=6),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Gamma",
            tool="insight",
            success=False,
            latency_ms=90.0,
            timestamp=base + timedelta(minutes=7),
        )
    )
    metrics_store.record_event(
        PersonaMetricEvent(
            persona="Gamma",
            tool="insight",
            success=True,
            latency_ms=100.0,
            timestamp=base + timedelta(minutes=8),
        )
    )

    summary = metrics_store.get_persona_comparison_summary()
    assert summary["pagination"]["total"] == 3

    rankings = summary.get("rankings", {})
    assert rankings["top_performers"][0]["persona"] == "Alpha"
    assert rankings["worst_failure_rates"][0]["persona"] == "Gamma"
    assert rankings["fastest_latency"][0]["persona"] == "Alpha"
    assert rankings["slowest_latency"][0]["persona"] == "Beta"

    results = summary["results"]
    personas = [entry["persona"] for entry in results]
    assert personas == ["Alpha", "Beta", "Gamma"]
    assert results[0]["totals"]["calls"] == 2


def test_persona_comparison_summary_filters(metrics_store):
    base = datetime(2024, 5, 1, tzinfo=timezone.utc)
    personas = ["Alpha", "Beta", "Gamma"]
    for offset, persona in enumerate(personas):
        metrics_store.record_event(
            PersonaMetricEvent(
                persona=persona,
                tool="insight",
                success=persona != "Gamma",
                latency_ms=100.0 + offset,
                timestamp=base + timedelta(minutes=offset),
            )
        )

    summary = metrics_store.get_persona_comparison_summary(personas=["gamma"])
    assert summary["pagination"]["total"] == 1
    assert summary["results"][0]["persona"] == "Gamma"

    summary = metrics_store.get_persona_comparison_summary(search="alp")
    assert summary["pagination"]["total"] == 1
    assert summary["results"][0]["persona"] == "Alpha"

    paged = metrics_store.get_persona_comparison_summary(page=2, page_size=1)
    assert paged["pagination"]["page"] == 2
    assert len(paged["results"]) == 1
    assert paged["results"][0]["persona"] == "Beta"


def test_task_lifecycle_metrics_aggregation(metrics_store):
    base = datetime(2024, 3, 1, 8, 0, tzinfo=timezone.utc)
    metrics_store.record_task_event(
        LifecycleEvent(
            entity_id="task-1",
            entity_key="task_id",
            event="created",
            persona="Atlas",
            tenant_id="tenant-a",
            to_status="ready",
            success=None,
            latency_ms=None,
            timestamp=base,
            metadata={"source": "unit"},
            extra={"reassignments": 0},
        )
    )
    metrics_store.record_task_event(
        LifecycleEvent(
            entity_id="task-1",
            entity_key="task_id",
            event="completed",
            persona="Atlas",
            tenant_id="tenant-a",
            from_status="review",
            to_status="done",
            success=True,
            latency_ms=540.0,
            timestamp=base + timedelta(minutes=30),
            extra={"reassignments": 0},
        )
    )
    metrics_store.record_task_event(
        LifecycleEvent(
            entity_id="task-1",
            entity_key="task_id",
            event="reassigned",
            persona="Atlas",
            tenant_id="tenant-a",
            from_status="done",
            to_status="done",
            success=None,
            latency_ms=None,
            timestamp=base + timedelta(hours=1),
            extra={"reassignments": 1},
        )
    )
    metrics_store.record_task_event(
        LifecycleEvent(
            entity_id="other",
            entity_key="task_id",
            event="cancelled",
            persona="Other",
            tenant_id="tenant-b",
            from_status="review",
            to_status="cancelled",
            success=False,
            latency_ms=200.0,
            timestamp=base + timedelta(hours=2),
            extra={"reassignments": 0},
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

    funnel = metrics.get("funnel") or {}
    stages = funnel.get("stages") or []
    stage_lookup = {stage["status"]: stage for stage in stages}
    ready_stage = stage_lookup["ready"]
    assert ready_stage["conversion_rate"] == pytest.approx(1.0)
    assert ready_stage["average_time_ms"] == pytest.approx(1_800_000.0)
    done_stage = stage_lookup["done"]
    assert done_stage["conversion_rate"] == pytest.approx(1.0)
    assert done_stage["abandonment_rate"] == pytest.approx(0.0)
    assert done_stage["average_time_ms"] is None

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


def test_task_funnel_conversion_and_dwell(metrics_store):
    base = datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc)

    def record(
        task_id: str,
        *,
        event: str,
        from_status: Optional[str],
        to_status: str,
        minutes: int,
        success: Optional[bool],
    ) -> None:
        metrics_store.record_task_event(
            LifecycleEvent(
                entity_id=task_id,
                entity_key="task_id",
                event=event,
                persona="Atlas",
                tenant_id="tenant-a",
                from_status=from_status,
                to_status=to_status,
                success=success,
                timestamp=base + timedelta(minutes=minutes),
            )
        )

    record("task-a", event="created", from_status=None, to_status="new", minutes=0, success=None)
    record(
        "task-a",
        event="started",
        from_status="new",
        to_status="in_progress",
        minutes=10,
        success=None,
    )
    record(
        "task-a",
        event="review",
        from_status="in_progress",
        to_status="review",
        minutes=30,
        success=None,
    )
    record(
        "task-a",
        event="completed",
        from_status="review",
        to_status="done",
        minutes=45,
        success=True,
    )

    record("task-b", event="created", from_status=None, to_status="new", minutes=5, success=None)
    record(
        "task-b",
        event="started",
        from_status="new",
        to_status="in_progress",
        minutes=20,
        success=None,
    )
    record(
        "task-b",
        event="failed",
        from_status="in_progress",
        to_status="failed",
        minutes=40,
        success=False,
    )

    record("task-c", event="created", from_status=None, to_status="new", minutes=12, success=None)

    metrics = metrics_store.get_task_metrics(persona="Atlas")
    funnel = metrics["funnel"]
    stages = funnel["stages"]
    stage_lookup = {stage["status"]: stage for stage in stages}

    new_stage = stage_lookup["new"]
    assert new_stage["entered"] == 3
    assert new_stage["converted"] == 2
    assert new_stage["abandoned"] == 1
    assert new_stage["conversion_rate"] == pytest.approx(2 / 3, rel=1e-6)
    assert new_stage["abandonment_rate"] == pytest.approx(1 / 3, rel=1e-6)
    assert new_stage["average_time_ms"] == pytest.approx(750_000.0)
    assert new_stage["samples"] == 2

    in_progress_stage = stage_lookup["in_progress"]
    assert in_progress_stage["converted"] == 2
    assert in_progress_stage["abandonment_rate"] == pytest.approx(0.0)
    assert in_progress_stage["average_time_ms"] == pytest.approx(1_200_000.0)

    review_stage = stage_lookup["review"]
    assert review_stage["converted"] == 1
    assert review_stage["abandonment_rate"] == pytest.approx(0.0)
    assert review_stage["average_time_ms"] == pytest.approx(900_000.0)

    done_stage = stage_lookup["done"]
    assert done_stage["converted"] == 1
    assert done_stage["abandonment_rate"] == pytest.approx(0.0)
    assert done_stage["average_time_ms"] is None

    failed_stage = stage_lookup["failed"]
    assert failed_stage["converted"] == 0
    assert failed_stage["abandonment_rate"] == pytest.approx(1.0)
    assert failed_stage["average_time_ms"] is None


def test_job_lifecycle_metrics_aggregation(metrics_store):
    base = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    metrics_store.record_job_event(
        LifecycleEvent(
            entity_id="job-1",
            entity_key="job_id",
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
        LifecycleEvent(
            entity_id="job-1",
            entity_key="job_id",
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
        LifecycleEvent(
            entity_id="job-2",
            entity_key="job_id",
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
        LifecycleEvent(
            entity_id="other",
            entity_key="job_id",
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
