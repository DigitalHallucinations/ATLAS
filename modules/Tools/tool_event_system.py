# modules/event_system.py

import logging
from typing import Callable, Dict, List


_FAILURE_THRESHOLD = 3


class EventSystem:
    def __init__(self):
        self._events: Dict[str, List[Callable]] = {}
        self._failure_counts: Dict[str, Dict[Callable, int]] = {}
        self._logger = logging.getLogger(__name__)

    def subscribe(self, event_name: str, callback: Callable) -> None:
        if event_name not in self._events:
            self._events[event_name] = []
        self._events[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        callbacks = self._events.get(event_name)
        if not callbacks:
            return
        try:
            callbacks.remove(callback)
        except ValueError:
            return
        if not callbacks:
            self._events.pop(event_name, None)
        failure_counts = self._failure_counts.get(event_name)
        if failure_counts:
            failure_counts.pop(callback, None)
            if not failure_counts:
                self._failure_counts.pop(event_name, None)

    def publish(self, event_name: str, *args, **kwargs) -> None:
        callbacks = list(self._events.get(event_name, ()))
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception:  # pylint: disable=broad-except
                self._logger.exception(
                    "Error while notifying subscriber %r for event '%s'", callback, event_name
                )
                failure_counts = self._failure_counts.setdefault(event_name, {})
                failure_counts[callback] = failure_counts.get(callback, 0) + 1
                if failure_counts[callback] >= _FAILURE_THRESHOLD:
                    self._logger.warning(
                        "Removing subscriber %r from event '%s' after %d failures",
                        callback,
                        event_name,
                        failure_counts[callback],
                    )
                    self.unsubscribe(event_name, callback)
            else:
                failure_counts = self._failure_counts.get(event_name)
                if failure_counts and callback in failure_counts:
                    failure_counts.pop(callback, None)
                    if not failure_counts:
                        self._failure_counts.pop(event_name, None)


event_system = EventSystem()
