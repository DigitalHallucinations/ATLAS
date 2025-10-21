"""Background worker for running conversation retention jobs."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from modules.conversation_store import ConversationStoreRepository


class RetentionWorker:
    """Periodically execute retention cleanup for the conversation store."""

    def __init__(
        self,
        repository: ConversationStoreRepository,
        *,
        interval_seconds: float = 3600.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._repository = repository
        self._interval = float(interval_seconds)
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

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            try:
                result = self._execute()
                if result:
                    self._logger.info("Retention pass complete: %s", result)
            except Exception as exc:  # noqa: BLE001 - background error reporting
                self._logger.exception("Conversation retention failed: %s", exc)

    def _execute(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        moment = now or datetime.now(timezone.utc)
        result = self._repository.run_retention(now=moment)
        return result


__all__ = ["RetentionWorker"]
