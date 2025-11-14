"""Unit tests for the retention worker scheduler."""

from __future__ import annotations

from datetime import datetime

import pytest

from modules.background_tasks.retention import RetentionWorker


class DummyRepository:
    def __init__(self, result: dict[str, object] | None = None) -> None:
        self.result = result or {"messages": {"deleted": 0}}
        self.invocations: list[datetime] = []

    def run_retention(self, *, now: datetime) -> dict[str, object]:
        self.invocations.append(now)
        return self.result


def test_retention_worker_adjusts_interval_for_backlog() -> None:
    repo = DummyRepository(result={"messages": {"hard_deleted": 12}})
    worker = RetentionWorker(
        repo,
        interval_seconds=60,
        min_interval_seconds=10,
        max_interval_seconds=600,
        backlog_low_water=0,
        backlog_high_water=5,
        catchup_multiplier=0.5,
        recovery_growth=2.0,
        slow_run_threshold=0.9,
        fast_run_threshold=0.5,
        jitter_seconds=0.0,
    )

    first = worker._on_success(elapsed=45.0, result=repo.result)
    assert pytest.approx(first, rel=1e-6) == 30.0

    repo.result = {"messages": {"hard_deleted": 0}}
    second = worker._on_success(elapsed=5.0, result=repo.result)
    assert pytest.approx(second, rel=1e-6) == 60.0


def test_retention_worker_backs_off_after_error() -> None:
    repo = DummyRepository()
    worker = RetentionWorker(
        repo,
        interval_seconds=60,
        min_interval_seconds=10,
        max_interval_seconds=900,
        error_backoff_factor=3.0,
        error_backoff_max_seconds=400.0,
        jitter_seconds=0.0,
    )

    first = worker._on_error()
    assert pytest.approx(first, rel=1e-6) == 180.0

    second = worker._on_error()
    assert pytest.approx(second, rel=1e-6) == 400.0

    third = worker._on_error()
    assert pytest.approx(third, rel=1e-6) == 400.0
