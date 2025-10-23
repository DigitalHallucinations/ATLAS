import datetime as dt
from dataclasses import dataclass

import pytest

from modules.Jobs.manifest_loader import JobMetadata
from modules.job_store import JobStatus
from modules.job_store.repository import JobStoreRepository
from modules.orchestration.job_scheduler import JobScheduler
from modules.Tools.Base_Tools.task_queue import RetryPolicy, TaskEvent

try:  # pragma: no cover - SQLAlchemy optional
    import sqlalchemy
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
except Exception:  # pragma: no cover - skip when dependency missing
    pytestmark = pytest.mark.skip("SQLAlchemy is required for job scheduler tests")
else:  # pragma: no cover - skip when stubbed module detected
    if not getattr(sqlalchemy, "__version__", None):
        pytestmark = pytest.mark.skip("SQLAlchemy runtime is required for job scheduler tests")


@pytest.fixture()
def engine():
    engine = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):  # pragma: no cover - event wiring
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    yield engine
    engine.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, future=True)


@pytest.fixture()
def job_repository(session_factory):
    repo = JobStoreRepository(session_factory)
    repo.create_schema()
    return repo


@dataclass
class _Call:
    job_name: str
    persona: str | None
    run_id: str | None


class JobManagerStub:
    def __init__(self) -> None:
        self.calls: list[_Call] = []

    async def run_job(
        self,
        job_name: str,
        *,
        persona: str | None = None,
        run_id: str | None = None,
        **_: object,
    ) -> None:
        self.calls.append(_Call(job_name, persona, run_id))


class FastQueueStub:
    def __init__(self) -> None:
        self.retry_policy = RetryPolicy(max_attempts=2, backoff_seconds=0.01, jitter_seconds=0.0, backoff_multiplier=1.0)
        self.monitors: list = []
        self.jobs: dict[str, dict[str, object]] = {}
        self._counter = 0

    def add_monitor(self, callback) -> None:
        self.monitors.append(callback)

    def get_retry_policy(self) -> RetryPolicy:
        return self.retry_policy

    def schedule_cron(
        self,
        *,
        name: str,
        payload,
        cron_schedule=None,
        cron_fields=None,
        idempotency_key=None,
        retry_policy=None,
    ):
        self._counter += 1
        job_id = f"stub-{self._counter}"
        policy = retry_policy or self.retry_policy
        next_run = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=1)
        self.jobs[job_id] = {
            "name": name,
            "payload": dict(payload),
            "policy": policy,
            "attempt": 0,
            "next_run": next_run,
            "cron": cron_fields or cron_schedule,
        }
        return {
            "job_id": job_id,
            "name": name,
            "state": "scheduled",
            "attempts": 0,
            "next_run_time": next_run,
            "retry_policy": policy.to_dict(),
            "recurring": True,
        }

    def emit(
        self,
        job_id: str,
        state: str,
        *,
        error: str | None = None,
        delay_seconds: float | None = 1.0,
    ) -> None:
        job = self.jobs[job_id]
        job["attempt"] = int(job.get("attempt", 0)) + 1
        attempt = job["attempt"]
        timestamp = dt.datetime.now(dt.timezone.utc)
        next_run = None
        if state in {"retrying", "succeeded"} and delay_seconds is not None:
            next_run = timestamp + dt.timedelta(seconds=delay_seconds)
            job["next_run"] = next_run
        metadata = {
            "policy": job["policy"],
            "last_run_at": timestamp,
        }
        event = TaskEvent(
            job_id=job_id,
            state=state,
            attempt=attempt,
            timestamp=timestamp,
            payload=dict(job["payload"]),
            error=error,
            next_run_time=next_run,
            metadata=metadata,
        )
        for monitor in list(self.monitors):
            monitor(event)


def _build_metadata(
    *,
    name: str = "DailyBriefing",
    persona: str | None = "ATLAS",
    recurrence: dict | None = None,
) -> JobMetadata:
    recurrence_payload = recurrence or {
        "frequency": "daily",
        "timezone": "UTC",
        "start_date": "2024-01-01T09:00:00Z",
    }
    return JobMetadata(
        name=name,
        summary="Summary",
        description="Description",
        personas=(persona,) if persona else tuple(),
        required_skills=tuple(),
        required_tools=tuple(),
        task_graph=tuple(),
        recurrence=recurrence_payload,
        acceptance_criteria=tuple(),
        escalation_policy={},
        persona=persona,
        source="modules/Jobs/jobs.json",
    )


def test_manifest_registration_persists_schedule(job_repository):
    queue = FastQueueStub()
    scheduler = JobScheduler(JobManagerStub(), queue, job_repository, tenant_id="tenant-1")
    metadata = _build_metadata()

    record = scheduler.register_manifest(metadata)
    assert record["schedule_type"] == "cron"
    assert "day" in record["expression"]

    job_name = scheduler._resolve_job_name(metadata)
    job_record = job_repository.get_job_by_name(job_name, tenant_id="tenant-1")
    assert job_record["status"] == JobStatus.SCHEDULED.value

    snapshot = job_repository.get_job(job_record["id"], tenant_id="tenant-1", with_schedule=True)
    schedule = snapshot["schedule"]
    assert schedule["next_run_at"] is not None
    assert schedule["metadata"]["retry_policy"]["max_attempts"] == queue.retry_policy.max_attempts


def test_scheduler_updates_on_retry_and_success(job_repository):
    queue = FastQueueStub()
    scheduler = JobScheduler(JobManagerStub(), queue, job_repository, tenant_id="tenant-1")
    metadata = _build_metadata()

    record = scheduler.register_manifest(metadata)
    job_name = scheduler._resolve_job_name(metadata)
    job_record = job_repository.get_job_by_name(job_name, tenant_id="tenant-1")
    queue_job_id = record["metadata"]["task_queue_job_id"]

    queue.emit(queue_job_id, "retrying", error="boom", delay_seconds=0.5)
    snapshot = job_repository.get_job(job_record["id"], tenant_id="tenant-1", with_schedule=True, with_runs=True)
    schedule = snapshot["schedule"]
    assert schedule["metadata"]["state"] == "retrying"
    assert schedule["metadata"]["attempt"] == 1
    assert schedule["metadata"]["error"] == "boom"
    assert snapshot["runs"][0]["status"] == "failed"

    queue.emit(queue_job_id, "succeeded", delay_seconds=0.25)
    snapshot = job_repository.get_job(job_record["id"], tenant_id="tenant-1", with_schedule=True, with_runs=True)
    schedule = snapshot["schedule"]
    assert schedule["metadata"]["state"] == "succeeded"
    assert schedule["next_run_at"] is not None
    assert snapshot["runs"][0]["status"] == "succeeded"
    assert len(snapshot["runs"]) == 2


def test_manual_override_updates_schedule(job_repository):
    queue = FastQueueStub()
    scheduler = JobScheduler(JobManagerStub(), queue, job_repository, tenant_id="tenant-1")
    metadata = _build_metadata()

    scheduler.register_manifest(metadata)
    next_run = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=4)
    updated = scheduler.apply_override(
        metadata.name,
        persona=metadata.persona,
        next_run_at=next_run,
        retry_policy={"max_attempts": 5},
        state="manual",
    )

    assert updated["next_run_at"] == next_run
    assert updated["metadata"]["retry_policy"]["max_attempts"] == 5
    assert updated["metadata"]["state"] == "manual"


def test_executor_invokes_job_manager(job_repository):
    queue = FastQueueStub()
    manager = JobManagerStub()
    scheduler = JobScheduler(manager, queue, job_repository, tenant_id="tenant-1")
    metadata = _build_metadata()
    scheduler.register_manifest(metadata)

    executor = scheduler.build_executor()
    job_id = "stub-run"
    executor({
        "job_id": job_id,
        "name": metadata.name,
        "payload": {"job_name": metadata.name, "persona": metadata.persona},
        "attempt": 1,
        "recurring": True,
    })

    assert manager.calls
    call = manager.calls[0]
    assert call.job_name == metadata.name
    assert call.persona == metadata.persona
    assert call.run_id == job_id

