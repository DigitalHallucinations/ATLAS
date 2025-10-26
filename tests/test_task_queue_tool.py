"""Tests for the durable task queue base tool."""

from __future__ import annotations

import threading
import datetime as _dt
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import types

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(_stream):  # pragma: no cover - simple stub
        return {}

    yaml_stub.safe_load = _safe_load
    sys.modules["yaml"] = yaml_stub

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")

    def _noop(*_args, **_kwargs):  # pragma: no cover - simple stub
        return None

    dotenv_stub.load_dotenv = _noop
    dotenv_stub.set_key = _noop
    dotenv_stub.find_dotenv = _noop
    sys.modules["dotenv"] = dotenv_stub

if "pytz" not in sys.modules:
    pytz_stub = types.ModuleType("pytz")

    def _timezone(name: str):  # pragma: no cover - simple stub
        if name.upper() == "UTC":
            return _dt.timezone.utc
        raise ValueError(name)

    pytz_stub.UTC = _dt.timezone.utc
    pytz_stub.timezone = _timezone
    sys.modules["pytz"] = pytz_stub

import importlib.util


_TASK_QUEUE_MODULE_PATH = Path(__file__).resolve().parent.parent / "modules" / "Tools" / "Base_Tools" / "task_queue.py"

spec = importlib.util.spec_from_file_location("_task_queue_tool", _TASK_QUEUE_MODULE_PATH)
task_queue = importlib.util.module_from_spec(spec)
assert spec and spec.loader  # pragma: no cover - defensive
sys.modules["_task_queue_tool"] = task_queue
spec.loader.exec_module(task_queue)  # type: ignore[union-attr]

RetryPolicy = task_queue.RetryPolicy
TaskQueueService = task_queue.TaskQueueService
build_task_queue_service = task_queue.build_task_queue_service


class _StubConfigManager:
    """Minimal stand-in for :class:`ATLAS.config.ConfigManager`."""

    def __init__(self, app_root: Path, overrides: Mapping[str, Any] | None = None) -> None:
        self._config = {"APP_ROOT": str(app_root)}
        if overrides:
            self._config.update(overrides)

    def get_config(self, key: str, default: Any = None) -> Any:  # pragma: no cover - simple proxy
        return self._config.get(key, default)


def _build_service(jobstore_url: str, tmp_path: Path, executor) -> TaskQueueService:
    config_manager = _StubConfigManager(tmp_path)
    retry_policy = RetryPolicy(max_attempts=3, backoff_seconds=0.25, jitter_seconds=0.0, backoff_multiplier=1.0)
    return build_task_queue_service(
        config_manager=config_manager,
        jobstore_url=jobstore_url,
        executor=executor,
        retry_policy=retry_policy,
    )


def test_enqueue_executes_task(tmp_path: Path, postgresql) -> None:
    events = deque()
    completed = threading.Event()

    def executor(payload: Mapping[str, Any]) -> None:
        events.append((payload["attempt"], payload["payload"]))
        completed.set()

    service = _build_service(postgresql.dsn(), tmp_path, executor)
    try:
        result = service.enqueue_task(
            name="example",
            payload={"value": 42},
            delay_seconds=0.1,
            idempotency_key="enqueue-example",
        )

        assert result["state"] == "scheduled"
        assert completed.wait(timeout=5), "task did not execute"

        attempt, payload = events.pop()
        assert attempt == 1
        assert payload == {"value": 42}

        status = service.get_status(result["job_id"])
        assert status["state"] == "succeeded"
        assert status["attempts"] == 1
    finally:
        service.shutdown(wait=True)


def test_enqueue_respects_idempotency(tmp_path: Path, postgresql) -> None:
    executed = threading.Event()
    attempts = []

    def executor(payload: Mapping[str, Any]) -> None:
        attempts.append(payload["attempt"])
        executed.set()

    service = _build_service(postgresql.dsn(), tmp_path, executor)
    try:
        key = "same-task"
        first = service.enqueue_task(name="repeat", payload={}, delay_seconds=0.05, idempotency_key=key)
        second = service.enqueue_task(name="repeat", payload={}, delay_seconds=0.05, idempotency_key=key)

        assert first["job_id"] == second["job_id"]
        assert second["state"] in {"scheduled", "succeeded"}

        assert executed.wait(timeout=5)
        assert attempts == [1]
    finally:
        service.shutdown(wait=True)


def test_retry_with_backoff(tmp_path: Path, postgresql) -> None:
    attempts: list[tuple[int, float]] = []
    completion = threading.Event()

    def executor(payload: Mapping[str, Any]) -> None:
        attempts.append((payload["attempt"], time.monotonic()))
        if payload["attempt"] < 2:
            raise RuntimeError("first attempt fails")
        completion.set()

    service = _build_service(postgresql.dsn(), tmp_path, executor)
    try:
        result = service.enqueue_task(name="flaky", payload={}, delay_seconds=0.05, idempotency_key="flaky-task")

        assert completion.wait(timeout=10), "task never completed"
        assert [attempt for attempt, _ in attempts] == [1, 2]

        first_time, second_time = attempts[0][1], attempts[1][1]
        assert second_time - first_time >= 0.24

        status = service.get_status(result["job_id"])
        assert status["state"] == "succeeded"
    finally:
        service.shutdown(wait=True)


def test_cron_schedule_executes_multiple_times(tmp_path: Path, postgresql) -> None:
    runs: list[int] = []
    completion = threading.Event()

    def executor(payload: Mapping[str, Any]) -> None:
        runs.append(payload["attempt"])
        if len(runs) >= 2:
            completion.set()

    service = _build_service(postgresql.dsn(), tmp_path, executor)
    try:
        result = service.schedule_cron(
            name="cron",
            cron_fields={"second": "*/1"},
            payload={},
            idempotency_key="cron-task",
        )

        assert result["state"] == "scheduled"
        assert completion.wait(timeout=10), "cron did not trigger twice"

        status = service.get_status(result["job_id"])
        assert status["state"] == "scheduled"
        assert status["last_run_status"] == "succeeded"
    finally:
        service.shutdown(wait=True)


def test_cancel_updates_status(tmp_path: Path, postgresql) -> None:
    blocked = threading.Event()

    def executor(_: Mapping[str, Any]) -> None:  # pragma: no cover - this task never executes
        blocked.set()

    service = _build_service(postgresql.dsn(), tmp_path, executor)
    try:
        result = service.enqueue_task(
            name="cancel-me",
            payload={},
            delay_seconds=10,
            idempotency_key="cancel-me",
        )

        cancelled = service.cancel(result["job_id"])
        assert cancelled["state"] == "cancelled"
        assert not blocked.wait(timeout=1)
    finally:
        service.shutdown(wait=True)
