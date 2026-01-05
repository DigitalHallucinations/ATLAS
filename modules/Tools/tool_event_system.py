# modules/event_system.py

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from core.messaging import (
    AgentBus,
    AgentMessage,
    MessagePriority,
    Subscription,
    get_agent_bus,
)


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


def publish_bus_event(
    event_name: str,
    payload: Dict[str, Any],
    *,
    priority: int = MessagePriority.NORMAL,
    correlation_id: Optional[str] = None,
    tracing: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    emit_legacy: bool = True,
) -> str:
    """Publish *payload* to the AgentBus and legacy subscribers.

    Args:
        event_name: Channel/topic identifier for the message.
        payload: Event payload forwarded to subscribers.
        priority: Relative priority used by the backend.
        correlation_id: Optional correlation identifier for tracing.
        tracing: Structured tracing metadata appended to the message.
        metadata: Free form metadata forwarded to the backend only.
        emit_legacy: When ``True`` (default) also emit via :func:`event_system.publish`.

    Returns:
        The correlation identifier assigned to the message.
    """
    bus_payload = {"event": event_name, "data": payload}
    headers = dict(metadata or {})
    if tracing:
        headers["tracing"] = str(tracing)
    
    message = AgentMessage(
        channel=event_name,
        payload=bus_payload,
        priority=priority,
        trace_id=correlation_id,
        headers=headers,
    )
    bus = get_agent_bus()
    correlation = bus.publish_from_sync(message)
    
    if emit_legacy:
        event_system.publish(event_name, payload)
    return correlation


def subscribe_bus_event(
    event_name: str,
    callback: Callable[..., Any],
    *,
    include_message: bool = False,
    retry_attempts: int = 3,
    retry_delay: float = 0.1,
    concurrency: int = 1,
) -> "DualSubscription":
    """Subscribe *callback* to both the legacy event system and the AgentBus."""

    callback_is_coro = asyncio.iscoroutinefunction(callback)
    try:
        signature = inspect.signature(callback)
        parameter_count = len(signature.parameters)
    except (TypeError, ValueError):
        parameter_count = 0
    accepts_message = include_message or parameter_count > 1

    async def bus_handler(message: AgentMessage) -> None:
        data = message.payload.get("data") if isinstance(message.payload, dict) else message.payload
        if callback_is_coro:
            if accepts_message:
                await callback(data, message)
            else:
                await callback(data)
        else:
            if accepts_message:
                await asyncio.to_thread(callback, data, message)
            else:
                await asyncio.to_thread(callback, data)

    # Subscribe asynchronously - need to schedule on event loop
    bus = get_agent_bus()
    
    async def _do_subscribe() -> Subscription:
        if not bus._running:
            await bus.start()
        return await bus.subscribe(
            event_name,
            bus_handler,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            concurrency=concurrency,
        )
    
    # Get or create event loop and schedule subscription
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(_do_subscribe(), loop)
        bus_subscription = future.result(timeout=5.0)
    except RuntimeError:
        # No running loop, create new one
        bus_subscription = asyncio.run(_do_subscribe())

    def legacy_handler(*args, **kwargs):
        data: Any
        if args:
            data = args[0]
        elif kwargs:
            data = kwargs
        else:
            data = None
        if callback_is_coro:
            if accepts_message:
                _schedule_coroutine(callback(data, None))
            else:
                _schedule_coroutine(callback(data))
        else:
            if accepts_message:
                callback(data, None)
            else:
                callback(data)

    event_system.subscribe(event_name, legacy_handler)
    return DualSubscription(bus_subscription, event_name, legacy_handler)


def _schedule_coroutine(coro: Any) -> None:
    """Schedule *coro* on the running loop, falling back to ``asyncio.run``."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
    else:
        loop.create_task(coro)


@dataclass
class DualSubscription:
    """Handle representing a combined bus and legacy subscription."""

    bus_subscription: Subscription
    event_name: str
    legacy_handler: Callable[..., Any]

    def cancel(self) -> None:
        event_system.unsubscribe(self.event_name, self.legacy_handler)
        # Schedule async cancel on event loop
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(self.bus_subscription.cancel(), loop)
        except RuntimeError:
            asyncio.run(self.bus_subscription.cancel())
