"""Background worker that records episodic conversation summaries."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping, MutableMapping

from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.memory_episodic import EpisodicMemoryTool
from modules.conversation_store import ConversationStoreRepository
from modules.logging.logger import setup_logger
from modules.orchestration.message_bus import MessageBus, Subscription


def _coerce_datetime(value: Any) -> datetime | None:
    """Convert serialized timestamps into timezone-aware ``datetime`` objects."""

    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class _BatchState:
    """Conversation specific aggregation context."""

    tenant_id: str
    conversation_id: str
    first_event: datetime
    last_event: datetime
    messages: list[MutableMapping[str, Any]] = field(default_factory=list)
    index: Dict[str, int] = field(default_factory=dict)

    def add_message(self, payload: Mapping[str, Any]) -> None:
        message_id = str(payload.get("id") or payload.get("message_id") or "")
        record = dict(payload)
        if message_id and message_id in self.index:
            position = self.index[message_id]
            self.messages[position] = record
        else:
            if message_id:
                self.index[message_id] = len(self.messages)
            self.messages.append(record)


class ConversationSummaryWorker:
    """Subscribe to conversation events and persist episodic summaries."""

    def __init__(
        self,
        repository: ConversationStoreRepository,
        *,
        config_getter: Callable[[], Mapping[str, Any]] | None = None,
        message_bus: MessageBus | None = None,
        poll_interval: float = 5.0,
        subscription_refresh: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._config_getter = config_getter or (lambda: {})
        self._bus = message_bus
        self._poll_interval = max(float(poll_interval), 1.0)
        self._subscription_refresh = max(float(subscription_refresh), 5.0)
        self._logger = logger or setup_logger(__name__)
        self._event_queue: asyncio.Queue[Mapping[str, Any]] = asyncio.Queue()
        self._batches: dict[str, _BatchState] = {}
        self._last_summary: dict[str, datetime] = {}
        self._poll_offsets: dict[str, datetime] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._summary_tool = EpisodicMemoryTool(repository=repository)
        self._thread: threading.Thread | None = None
        self._stop_signal = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_subscription_sync: datetime | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_signal.clear()
        self._thread = threading.Thread(target=self._run_thread, name="ConversationSummaryWorker", daemon=True)
        self._thread.start()

    def stop(self, *, wait: bool = True) -> None:
        self._stop_signal.set()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(lambda: None)
        thread = self._thread
        if wait and thread is not None:
            thread.join()

    def run_once(self) -> None:
        settings = self._load_settings()
        if not settings.get("enabled"):
            self._logger.debug("Conversation summary worker is disabled; run_once skipped")
            return
        asyncio.run(self._process_once(settings))

    # ------------------------------------------------------------------
    # Internal execution

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_loop())
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.exception("Conversation summary worker crashed")
        finally:
            self._cancel_subscriptions()
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    async def _run_loop(self) -> None:
        poll_task: asyncio.Task[None] | None = None
        try:
            if self._bus is None:
                poll_task = asyncio.create_task(self._polling_loop())
            while not self._stop_signal.is_set():
                settings = self._load_settings()
                if not settings.get("enabled"):
                    await asyncio.sleep(self._poll_interval)
                    continue
                if self._bus is not None:
                    await self._ensure_subscriptions(settings)
                await self._drain_events()
                await self._flush_due_batches(settings)
        finally:
            if poll_task is not None:
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poll_task

    async def _process_once(self, settings: Mapping[str, Any]) -> None:
        if self._bus is None:
            await asyncio.to_thread(self._poll_repository_once, settings)
        await self._flush_due_batches(settings, force=True)

    # ------------------------------------------------------------------
    # Event ingestion

    async def _drain_events(self) -> None:
        try:
            payload = await asyncio.wait_for(self._event_queue.get(), timeout=self._poll_interval)
        except asyncio.TimeoutError:
            return
        self._record_event(payload)
        while not self._event_queue.empty():
            payload = self._event_queue.get_nowait()
            self._record_event(payload)

    def _record_event(self, payload: Mapping[str, Any]) -> None:
        message = payload.get("message")
        if not isinstance(message, Mapping):
            self._logger.debug("Skipping event without message payload: %s", payload)
            return
        tenant_id = str(payload.get("tenant_id") or message.get("tenant_id") or "")
        conversation_id = str(payload.get("conversation_id") or message.get("conversation_id") or "")
        if not tenant_id or not conversation_id:
            self._logger.debug("Skipping event with incomplete identifiers: %s", payload)
            return
        created_at = _coerce_datetime(payload.get("created_at") or message.get("created_at"))
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        key = conversation_id
        state = self._batches.get(key)
        if state is None:
            state = _BatchState(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                first_event=created_at,
                last_event=created_at,
            )
            self._batches[key] = state
        state.last_event = created_at
        if created_at < state.first_event:
            state.first_event = created_at
        state.add_message(message)
        self._poll_offsets[key] = max(created_at, self._poll_offsets.get(key, created_at))

    # ------------------------------------------------------------------
    # Polling fall back

    async def _polling_loop(self) -> None:
        while not self._stop_signal.is_set():
            settings = self._load_settings()
            if settings.get("enabled"):
                await asyncio.to_thread(self._poll_repository_once, settings)
            await asyncio.sleep(self._poll_interval)

    def _poll_repository_once(self, settings: Mapping[str, Any]) -> None:
        try:
            tenant_ids = self._repository.list_known_tenants()
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.exception("Failed to enumerate tenants during polling")
            return
        for tenant_id in tenant_ids:
            try:
                conversations = self._repository.list_conversations_for_tenant(
                    tenant_id=tenant_id, include_archived=False
                )
            except Exception:  # pragma: no cover - defensive logging only
                self._logger.exception("Failed to enumerate conversations for tenant %s", tenant_id)
                continue
            for record in conversations:
                conversation_id = str(record.get("id"))
                if not conversation_id:
                    continue
                last_seen = self._poll_offsets.get(conversation_id)
                try:
                    messages = self._repository.fetch_messages(
                        conversation_id=conversation_id,
                        tenant_id=tenant_id,
                        limit=50,
                        order="desc",
                    )
                except Exception:  # pragma: no cover - defensive logging only
                    self._logger.exception(
                        "Failed to fetch messages for %s (tenant=%s)", conversation_id, tenant_id
                    )
                    continue
                if not messages:
                    continue
                recent = list(reversed(messages))
                newest_timestamp = last_seen
                for message in recent:
                    created_at = _coerce_datetime(message.get("created_at"))
                    if created_at is None:
                        continue
                    if last_seen is not None and created_at <= last_seen:
                        continue
                    event_payload = {
                        "tenant_id": tenant_id,
                        "conversation_id": conversation_id,
                        "created_at": created_at.isoformat(),
                        "message": message,
                    }
                    self._event_queue.put_nowait(event_payload)
                    if newest_timestamp is None or created_at > newest_timestamp:
                        newest_timestamp = created_at
                if newest_timestamp is not None:
                    self._poll_offsets[conversation_id] = newest_timestamp

    # ------------------------------------------------------------------
    # Message bus integration

    async def _ensure_subscriptions(self, settings: Mapping[str, Any]) -> None:
        if self._bus is None:
            return
        now = datetime.now(timezone.utc)
        if (
            self._last_subscription_sync is not None
            and (now - self._last_subscription_sync).total_seconds() < self._subscription_refresh
        ):
            return
        self._last_subscription_sync = now
        try:
            tenant_ids = self._repository.list_known_tenants()
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.exception("Failed to enumerate tenants for subscription sync")
            return
        for tenant_id in tenant_ids:
            try:
                conversations = self._repository.list_conversations_for_tenant(
                    tenant_id=tenant_id, include_archived=False
                )
            except Exception:  # pragma: no cover - defensive logging only
                self._logger.exception("Failed to enumerate conversations for tenant %s", tenant_id)
                continue
            for record in conversations:
                conversation_id = str(record.get("id"))
                if not conversation_id or conversation_id in self._subscriptions:
                    continue
                topic = f"conversation.events.{conversation_id}"

                async def _handler(message, _topic=topic) -> None:
                    payload = getattr(message, "payload", None)
                    if not isinstance(payload, Mapping):
                        return
                    payload = dict(payload)
                    payload.setdefault("conversation_id", conversation_id)
                    await self._event_queue.put(payload)

                subscription = self._bus.subscribe(topic, _handler)
                self._subscriptions[conversation_id] = subscription

    def _cancel_subscriptions(self) -> None:
        for subscription in list(self._subscriptions.values()):
            try:
                subscription.cancel()
            except Exception:  # pragma: no cover - defensive logging only
                self._logger.debug("Failed to cancel subscription cleanly", exc_info=True)
        self._subscriptions.clear()

    # ------------------------------------------------------------------
    # Batch evaluation and persistence

    async def _flush_due_batches(
        self,
        settings: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc)
        cadence = self._as_float(settings.get("cadence_seconds"))
        window = self._as_float(settings.get("window_seconds"))
        batch_size = self._as_int(settings.get("batch_size"), default=10)
        retention = settings.get("retention")
        tenant_overrides = settings.get("tenants") if isinstance(settings.get("tenants"), Mapping) else {}

        pending: list[_BatchState] = []
        for key, state in list(self._batches.items()):
            overrides = tenant_overrides.get(state.tenant_id) if isinstance(tenant_overrides, Mapping) else None
            tenant_window = self._as_float((overrides or {}).get("window_seconds"), fallback=window)
            tenant_batch = self._as_int((overrides or {}).get("batch_size"), default=batch_size)
            tenant_cadence = self._as_float((overrides or {}).get("cadence_seconds"), fallback=cadence)
            messages_ready = len(state.messages) >= tenant_batch if tenant_batch else False
            window_expired = False
            if tenant_window:
                window_expired = (now - state.first_event).total_seconds() >= tenant_window
            if force or messages_ready or window_expired:
                last_run = self._last_summary.get(key)
                if not force and tenant_cadence and last_run is not None:
                    if (now - last_run).total_seconds() < tenant_cadence:
                        continue
                pending.append(state)

        for state in pending:
            try:
                await self._persist_summary(state, retention, tenant_overrides)
            except Exception:  # pragma: no cover - defensive logging only
                self._logger.exception(
                    "Failed to persist episodic summary for conversation %s", state.conversation_id
                )
                state.first_event = now
                continue
            self._last_summary[state.conversation_id] = now
            self._batches.pop(state.conversation_id, None)

    async def _persist_summary(
        self,
        state: _BatchState,
        retention: Mapping[str, Any] | None,
        tenant_overrides: Mapping[str, Any] | None,
    ) -> None:
        ordered = sorted(
            state.messages,
            key=lambda item: _coerce_datetime(item.get("created_at")) or state.first_event,
        )
        snapshot = await self._generate_snapshot(state.conversation_id, ordered)
        if not snapshot:
            return
        occurred_at = state.last_event
        expires_at = self._resolve_expiration(state.tenant_id, occurred_at, retention, tenant_overrides)
        metadata = {
            "source": "conversation_summary_worker",
            "message_count": len(ordered),
            "first_event": state.first_event.isoformat(),
            "last_event": state.last_event.isoformat(),
            "persona": snapshot.get("persona"),
        }
        tags = ["conversation-summary", "auto"]
        if snapshot.get("participants"):
            metadata["participants"] = list(snapshot["participants"])
        await self._summary_tool.store(
            tenant_id=state.tenant_id,
            conversation_id=state.conversation_id,
            occurred_at=occurred_at,
            expires_at=expires_at,
            content={
                "summary": snapshot.get("summary"),
                "highlights": snapshot.get("highlights"),
                "snapshot": snapshot,
            },
            tags=tags,
            metadata=metadata,
            title=f"Conversation summary – {occurred_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        )

    async def _generate_snapshot(
        self,
        conversation_id: str,
        messages: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        payload = await context_tracker(
            conversation_id=conversation_id,
            conversation_history=messages,
        )
        payload = dict(payload)
        payload.setdefault("persona", "context_tracker")
        return payload

    # ------------------------------------------------------------------
    # Settings helpers

    def _load_settings(self) -> Mapping[str, Any]:
        try:
            settings = self._config_getter() or {}
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.exception("Failed to load conversation summary settings")
            return {}
        if not isinstance(settings, Mapping):
            return {}
        return settings

    @staticmethod
    def _as_float(value: Any, fallback: float | None = None) -> float | None:
        if value is None:
            return fallback
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if numeric > 0 else fallback

    @staticmethod
    def _as_int(value: Any, *, default: int | None = None) -> int | None:
        if value is None:
            return default
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        if numeric < 1:
            return default
        return numeric

    def _resolve_expiration(
        self,
        tenant_id: str,
        occurred_at: datetime,
        retention: Mapping[str, Any] | None,
        tenant_overrides: Mapping[str, Any] | None,
    ) -> datetime | None:
        ttl_days: int | None = None
        if isinstance(tenant_overrides, Mapping):
            override = tenant_overrides.get(tenant_id)
            if isinstance(override, Mapping):
                override_ttl = override.get("retention_days")
                ttl_days = self._as_int(override_ttl)
        if ttl_days is None and isinstance(retention, Mapping):
            tenants = retention.get("tenants")
            if isinstance(tenants, Mapping):
                tenant_block = tenants.get(tenant_id)
                if isinstance(tenant_block, Mapping):
                    ttl_days = self._as_int(tenant_block.get("days"))
            if ttl_days is None:
                ttl_days = self._as_int(retention.get("default_days"))
        if ttl_days is None:
            return None
        return occurred_at + timedelta(days=ttl_days)


__all__ = ["ConversationSummaryWorker"]
