# modules/event_system.py

from typing import Callable, Dict, List


class EventSystem:
    def __init__(self):
        self._events: Dict[str, List[Callable]] = {}

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

    def publish(self, event_name: str, *args, **kwargs) -> None:
        callbacks = list(self._events.get(event_name, ()))
        for callback in callbacks:
            callback(*args, **kwargs)


event_system = EventSystem()
