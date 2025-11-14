"""Background worker for running conversation retention jobs."""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from modules.conversation_store import ConversationStoreRepository


class RetentionWorker:
    """Periodically execute retention cleanup for the conversation store."""

    def __init__(
        self,
        repository: ConversationStoreRepository,
        *,
        interval_seconds: float = 3600.0,
        min_interval_seconds: Optional[float] = None,
        max_interval_seconds: Optional[float] = None,
        backlog_low_water: int = 0,
        backlog_high_water: int = 50,
        catchup_multiplier: float = 0.5,
        recovery_growth: float = 1.5,
        slow_run_threshold: float = 0.8,
        fast_run_threshold: float = 0.3,
        jitter_seconds: float = 30.0,
        error_backoff_factor: float = 2.0,
        error_backoff_max_seconds: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        base_interval = float(interval_seconds)
        computed_min = min_interval_seconds
        if computed_min is None:
            computed_min = min(base_interval, 60.0)
        if computed_min <= 0:
            raise ValueError("min_interval_seconds must be positive")
        min_interval = float(computed_min)

        computed_max = max_interval_seconds
        if computed_max is None:
            computed_max = max(base_interval, 3600.0)
        if computed_max <= 0:
            raise ValueError("max_interval_seconds must be positive")
        max_interval = float(computed_max)

        if min_interval > max_interval:
            raise ValueError("min_interval_seconds cannot exceed max_interval_seconds")

        self._repository = repository
        self._min_interval = max(min_interval, 0.01)
        self._max_interval = max(self._min_interval, max_interval)
        self._catchup_multiplier = max(0.01, min(float(catchup_multiplier), 1.0))
        self._recovery_growth = max(1.0, float(recovery_growth))
        self._slow_run_threshold = max(0.0, float(slow_run_threshold))
        self._fast_run_threshold = max(0.0, float(fast_run_threshold))
        self._backlog_low_water = max(0, int(backlog_low_water))
        self._backlog_high_water = max(self._backlog_low_water, int(backlog_high_water))
        self._jitter_seconds = max(0.0, float(jitter_seconds))
        self._error_backoff_factor = max(1.0, float(error_backoff_factor))

        computed_error_max = error_backoff_max_seconds
        if computed_error_max is None:
            computed_error_max = self._max_interval
        elif computed_error_max <= 0:
            raise ValueError("error_backoff_max_seconds must be positive")
        self._error_backoff_max = max(self._min_interval, float(computed_error_max))

        initial_interval = max(self._min_interval, min(base_interval, self._max_interval))
        self._current_interval = initial_interval

        self._logger = logger or logging.getLogger(__name__)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the background retention thread if it is not running."""

        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RetentionWorker", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        """Signal the background worker to stop."""

        self._stop_event.set()
        thread = self._thread
        if wait and thread is not None:
            thread.join()

    def run_once(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Execute a single retention pass synchronously."""

        return self._execute(now=now)

    @property
    def current_interval(self) -> float:
        """Return the interval that will be used for the next wait cycle."""

        return self._current_interval

    def _run_loop(self) -> None:
        next_interval = self._current_interval
        while not self._stop_event.is_set():
            wait_duration = self._apply_jitter(next_interval)
            if self._stop_event.wait(wait_duration):
                break

            started = time.monotonic()
            try:
                result = self._execute()
            except Exception as exc:  # noqa: BLE001 - background error reporting
                self._logger.exception("Conversation retention failed: %s", exc)
                next_interval = self._on_error()
                continue

            elapsed = time.monotonic() - started
            if result:
                self._logger.info("Retention pass complete: %s", result)
            next_interval = self._on_success(elapsed=elapsed, result=result)

    def _execute(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        moment = now or datetime.now(timezone.utc)
        result = self._repository.run_retention(now=moment)
        return result

    def _apply_jitter(self, interval: float) -> float:
        if interval <= 0:
            return self._min_interval
        if self._jitter_seconds <= 0:
            return max(self._min_interval, interval)
        offset = random.uniform(-self._jitter_seconds, self._jitter_seconds)
        return max(self._min_interval, interval + offset)

    def _on_success(self, *, elapsed: float, result: Mapping[str, Any] | None) -> float:
        processed = self._count_processed(result)
        interval = self._current_interval

        next_interval = interval
        if processed >= self._backlog_high_water or (
            interval > 0 and elapsed >= interval * self._slow_run_threshold
        ):
            next_interval = max(self._min_interval, interval * self._catchup_multiplier)
        elif (
            processed <= self._backlog_low_water
            and interval > 0
            and elapsed <= interval * self._fast_run_threshold
        ):
            growth_target = interval * self._recovery_growth
            next_interval = min(self._max_interval, growth_target)

        next_interval = max(self._min_interval, min(next_interval, self._max_interval))
        self._current_interval = next_interval
        return next_interval

    def _on_error(self) -> float:
        interval = self._current_interval
        next_interval = interval * self._error_backoff_factor
        capped = min(self._max_interval, self._error_backoff_max)
        next_interval = min(next_interval, capped)
        next_interval = max(self._min_interval, next_interval)
        self._current_interval = next_interval
        return next_interval

    def _count_processed(self, result: Mapping[str, Any] | None) -> int:
        if not result:
            return 0

        total = 0

        def _accumulate(payload: Any) -> None:
            nonlocal total
            if isinstance(payload, Mapping):
                for value in payload.values():
                    _accumulate(value)
            elif isinstance(payload, (int, float)):
                if payload > 0:
                    total += int(payload)

        _accumulate(result)
        return total


__all__ = ["RetentionWorker"]
