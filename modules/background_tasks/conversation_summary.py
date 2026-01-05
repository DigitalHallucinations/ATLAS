"""Background worker that records episodic conversation summaries."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.memory_episodic import EpisodicMemoryTool
from modules.conversation_store import ConversationStoreRepository
from modules.logging.logger import setup_logger
from core.messaging import AgentBus, MessagePriority, Subscription


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


_QUESTION_PREFIXES = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "will",
    "may",
    "have",
    "has",
    "had",
)


def extract_followups(
    snapshot: Mapping[str, Any],
    templates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate *snapshot* against configured follow-up *templates*."""

    history = _normalize_history_messages(snapshot.get("history"))
    summary_text = _coerce_text(snapshot.get("summary"))
    highlights = _normalize_highlights(snapshot.get("highlights"))

    followups: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for template in templates:
        if not isinstance(template, Mapping):
            continue
        template_id = str(template.get("id") or "").strip()
        if not template_id:
            continue

        matching = template.get("matching")
        if not isinstance(matching, Mapping):
            matching = {}

        scope = matching.get("scope")
        if isinstance(scope, (list, tuple, set)):
            scoped = [str(item).strip().lower() for item in scope]
            scopes = [item for item in scoped if item in {"history", "summary", "highlights"}]
            if not scopes:
                scopes = ["history"]
        else:
            scopes = ["history"]

        pattern_text = matching.get("pattern")
        pattern = None
        if isinstance(pattern_text, str) and pattern_text.strip():
            try:
                pattern = re.compile(pattern_text.strip(), re.IGNORECASE)
            except re.error:
                pattern = None

        contexts: list[dict[str, Any]] = []
        if "history" in scopes:
            contexts.extend(_match_history(history, matching, pattern))
        if "summary" in scopes:
            contexts.extend(_match_summary(summary_text, matching, pattern))
        if "highlights" in scopes:
            contexts.extend(_match_highlights(highlights, matching, pattern))

        if not contexts:
            continue

        for counter, context in enumerate(contexts, start=1):
            identifier = str(context.get("identifier") or counter)
            followup_id = f"{template_id}::{identifier}"
            if followup_id in seen_ids:
                suffix = 1
                while f"{followup_id}-{suffix}" in seen_ids:
                    suffix += 1
                followup_id = f"{followup_id}-{suffix}"
            seen_ids.add(followup_id)

            entry: dict[str, Any] = {
                "id": followup_id,
                "template_id": template_id,
                "kind": str(template.get("kind") or "action_item"),
                "title": str(template.get("title") or template_id),
                "description": str(template.get("description") or ""),
                "source": _build_source_payload(context),
                "reasons": list(context.get("reasons", [])),
            }

            evidence = context.get("evidence")
            if isinstance(evidence, str) and evidence:
                entry["evidence"] = evidence

            task_spec = template.get("task")
            if isinstance(task_spec, Mapping) and task_spec:
                entry["task"] = dict(task_spec)

            escalation_spec = template.get("escalation")
            if isinstance(escalation_spec, Mapping) and escalation_spec:
                entry["escalation"] = dict(escalation_spec)

            followups.append(entry)

    return followups


def _build_source_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": context.get("type")}
    if "role" in context and context.get("role") is not None:
        payload["role"] = context.get("role")
    if "index" in context and context.get("index") is not None:
        payload["index"] = context.get("index")
    if "position" in context and context.get("position") is not None:
        payload["position"] = context.get("position")
    if "timestamp" in context and context.get("timestamp") is not None:
        payload["timestamp"] = context.get("timestamp")
    if "metadata" in context and isinstance(context.get("metadata"), Mapping):
        payload["metadata"] = dict(context.get("metadata"))
    content = context.get("content")
    if isinstance(content, str) and content:
        payload["content"] = content
    return payload


def _normalize_history_messages(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, Iterable):
        return []

    normalized: list[dict[str, Any]] = []
    for position, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            continue
        raw_role = entry.get("role")
        role = str(raw_role or "").strip().lower() or "unknown"
        content_value = entry.get("content")
        content = str(content_value).strip() if isinstance(content_value, str) else ""
        try:
            index_value = int(entry.get("index"))
        except (TypeError, ValueError):
            index_value = position

        record: dict[str, Any] = {
            "role": role,
            "content": content,
            "position": position,
            "index": index_value,
        }

        timestamp = entry.get("timestamp")
        if timestamp is not None:
            record["timestamp"] = timestamp

        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping) and metadata:
            record["metadata"] = dict(metadata)

        normalized.append(record)

    return normalized


def _normalize_highlights(highlights: Any) -> list[str]:
    if not isinstance(highlights, Iterable):
        return []
    result: list[str] = []
    for item in highlights:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
    return result


def _coerce_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _match_history(
    history: Sequence[Mapping[str, Any]],
    matching: Mapping[str, Any],
    pattern: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    if not history:
        return []

    window_value = matching.get("history_window")
    if isinstance(window_value, int) and window_value > 0:
        candidates = list(history[-window_value:])
    else:
        candidates = list(history)

    include_roles = {
        str(role).strip().lower()
        for role in matching.get("include_roles", [])
        if str(role).strip()
    }
    exclude_roles = {
        str(role).strip().lower()
        for role in matching.get("exclude_roles", [])
        if str(role).strip()
    }
    response_roles = {
        str(role).strip().lower()
        for role in matching.get("response_roles", [])
        if str(role).strip()
    }

    keywords = tuple(
        str(keyword).strip().lower()
        for keyword in matching.get("keywords", [])
        if str(keyword).strip()
    )
    require_question = bool(matching.get("unanswered_question"))

    contexts: list[dict[str, Any]] = []
    for message in candidates:
        role = message.get("role")
        if include_roles and role not in include_roles:
            continue
        if role in exclude_roles:
            continue

        text = message.get("content") or ""
        if not text:
            continue

        reasons: list[str] = []
        lowered = text.lower()

        if keywords:
            if not any(keyword in lowered for keyword in keywords):
                continue
            reasons.append("keyword")

        if pattern is not None:
            if not pattern.search(text):
                continue
            reasons.append("pattern")

        if require_question:
            if not _looks_like_question(text):
                continue
            responders = _resolve_responder_roles(role, response_roles)
            if _has_responder_message_after(history, message.get("position", 0), responders):
                continue
            reasons.append("unanswered_question")

        if not reasons and not keywords and pattern is None and not require_question:
            continue

        context: dict[str, Any] = {
            "type": "message",
            "role": role,
            "index": message.get("index"),
            "position": message.get("position"),
            "content": text,
            "reasons": reasons,
            "identifier": f"message-{message.get('index')}",
            "evidence": text,
        }
        if "timestamp" in message:
            context["timestamp"] = message.get("timestamp")
        if "metadata" in message:
            context["metadata"] = message.get("metadata")
        contexts.append(context)

    return contexts


def _match_summary(
    summary: str,
    matching: Mapping[str, Any],
    pattern: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    if not summary:
        return []

    keywords = tuple(
        str(keyword).strip().lower()
        for keyword in matching.get("keywords", [])
        if str(keyword).strip()
    )
    require_question = bool(matching.get("unanswered_question"))

    reasons: list[str] = []
    lowered = summary.lower()

    if keywords and any(keyword in lowered for keyword in keywords):
        reasons.append("keyword")

    if pattern is not None and pattern.search(summary):
        reasons.append("pattern")

    if require_question and _looks_like_question(summary):
        reasons.append("unanswered_question")

    if not reasons:
        return []

    return [
        {
            "type": "summary",
            "content": summary,
            "reasons": reasons,
            "identifier": "summary",
            "evidence": summary,
        }
    ]


def _match_highlights(
    highlights: Sequence[str],
    matching: Mapping[str, Any],
    pattern: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    if not highlights:
        return []

    keywords = tuple(
        str(keyword).strip().lower()
        for keyword in matching.get("keywords", [])
        if str(keyword).strip()
    )
    require_question = bool(matching.get("unanswered_question"))

    contexts: list[dict[str, Any]] = []
    for index, value in enumerate(highlights):
        reasons: list[str] = []
        lowered = value.lower()

        if keywords and any(keyword in lowered for keyword in keywords):
            reasons.append("keyword")

        if pattern is not None and pattern.search(value):
            reasons.append("pattern")

        if require_question and _looks_like_question(value):
            reasons.append("unanswered_question")

        if not reasons:
            continue

        contexts.append(
            {
                "type": "highlight",
                "index": index,
                "content": value,
                "reasons": reasons,
                "identifier": f"highlight-{index}",
                "evidence": value,
            }
        )

    return contexts


def _resolve_responder_roles(role: Any, configured: Iterable[str]) -> set[str]:
    responders = {candidate for candidate in configured if candidate}
    if responders:
        return responders
    role_key = str(role or "").strip().lower()
    if role_key == "assistant":
        return {"user"}
    return {"assistant"}


def _has_responder_message_after(
    history: Sequence[Mapping[str, Any]],
    position: int,
    responders: Iterable[str],
) -> bool:
    responder_set = {str(item).strip().lower() for item in responders if str(item).strip()}
    for message in history:
        message_position = int(message.get("position", 0))
        if message_position <= position:
            continue
        role = str(message.get("role") or "").strip().lower()
        if role in responder_set and str(message.get("content") or "").strip():
            return True
    return False


def _looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "?" in stripped:
        return True
    lowered = stripped.lower()
    for prefix in _QUESTION_PREFIXES:
        if lowered.startswith(prefix + " ") or lowered.startswith(prefix + "?"):
            return True
    return False


class ConversationSummaryWorker:
    """Subscribe to conversation events and persist episodic summaries."""

    FOLLOWUP_TOPIC = "conversation.followups"

    def __init__(
        self,
        repository: ConversationStoreRepository,
        *,
        config_getter: Callable[[], Mapping[str, Any]] | None = None,
        agent_bus: AgentBus | None = None,
        poll_interval: float = 5.0,
        subscription_refresh: float = 60.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._config_getter = config_getter or (lambda: {})
        self._bus = agent_bus
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
            loop.run_until_complete(self._cancel_subscriptions())
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

                subscription = await self._bus.subscribe(topic, _handler)
                self._subscriptions[conversation_id] = subscription

    async def _cancel_subscriptions(self) -> None:
        for subscription in list(self._subscriptions.values()):
            try:
                await subscription.cancel()
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
                await self._persist_summary(state, settings, retention, tenant_overrides)
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
        settings: Mapping[str, Any],
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
        persona = self._resolve_persona(settings, tenant_overrides, state.tenant_id, snapshot)
        metadata = {
            "source": "conversation_summary_worker",
            "message_count": len(ordered),
            "first_event": state.first_event.isoformat(),
            "last_event": state.last_event.isoformat(),
            "persona": persona,
        }
        tags = ["conversation-summary", "auto"]
        if snapshot.get("participants"):
            metadata["participants"] = list(snapshot["participants"])

        followup_templates = self._resolve_followup_templates(settings, tenant_overrides, state.tenant_id, persona)
        followups = extract_followups(snapshot, followup_templates)
        if followups:
            metadata["followup_count"] = len(followups)

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

        if followups:
            self._emit_followup_event(state, persona, occurred_at, snapshot, followups)

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

    def _resolve_persona(
        self,
        settings: Mapping[str, Any],
        tenant_overrides: Mapping[str, Any] | None,
        tenant_id: str,
        snapshot: Mapping[str, Any],
    ) -> str | None:
        candidate: Any = None
        if isinstance(tenant_overrides, Mapping):
            tenant_block = tenant_overrides.get(tenant_id)
            if isinstance(tenant_block, Mapping):
                candidate = tenant_block.get("persona")
        if not candidate and isinstance(settings.get("persona"), str):
            candidate = settings.get("persona")
        if not candidate:
            candidate = snapshot.get("persona")
        if isinstance(candidate, str):
            candidate = candidate.strip()
        return candidate or None

    def _resolve_followup_templates(
        self,
        settings: Mapping[str, Any],
        tenant_overrides: Mapping[str, Any] | None,
        tenant_id: str,
        persona: str | None,
    ) -> list[Mapping[str, Any]]:
        persona_key = persona.strip().lower() if isinstance(persona, str) and persona.strip() else None
        ordered_ids: list[str] = []
        templates: dict[str, Mapping[str, Any]] = {}

        def _append(entries: Iterable[Mapping[str, Any]], *, replace: bool = False) -> None:
            for template in entries:
                if not isinstance(template, Mapping):
                    continue
                template_id = str(template.get("id") or "").strip()
                if not template_id:
                    continue
                if template_id in templates:
                    if not replace:
                        continue
                    ordered_ids[:] = [item for item in ordered_ids if item != template_id]
                templates[template_id] = dict(template)
                ordered_ids.append(template_id)

        followup_block = settings.get("followups")
        if isinstance(followup_block, Mapping):
            defaults = followup_block.get("defaults")
            if isinstance(defaults, list):
                _append(defaults)
            persona_map = followup_block.get("personas")
            if persona_key and isinstance(persona_map, Mapping):
                persona_entries = persona_map.get(persona_key)
                if isinstance(persona_entries, list):
                    _append(persona_entries)

        if isinstance(tenant_overrides, Mapping):
            tenant_block = tenant_overrides.get(tenant_id)
            if isinstance(tenant_block, Mapping):
                tenant_followups = tenant_block.get("followups")
                if isinstance(tenant_followups, Mapping):
                    defaults = tenant_followups.get("defaults")
                    if isinstance(defaults, list):
                        _append(defaults, replace=True)
                    persona_map = tenant_followups.get("personas")
                    if persona_key and isinstance(persona_map, Mapping):
                        persona_entries = persona_map.get(persona_key)
                        if isinstance(persona_entries, list):
                            _append(persona_entries, replace=True)

        return [templates[identifier] for identifier in ordered_ids]

    def _emit_followup_event(
        self,
        state: _BatchState,
        persona: str | None,
        occurred_at: datetime,
        snapshot: Mapping[str, Any],
        followups: Sequence[Mapping[str, Any]],
    ) -> None:
        if self._bus is None:
            self._logger.debug(
                "Detected %d follow-up items for %s but no message bus is configured",
                len(followups),
                state.conversation_id,
            )
            return

        payload = {
            "tenant_id": state.tenant_id,
            "conversation_id": state.conversation_id,
            "persona": persona,
            "summary": {
                "occurred_at": occurred_at.isoformat(),
                "window_start": state.first_event.isoformat(),
                "window_end": state.last_event.isoformat(),
                "message_count": len(snapshot.get("history") or []),
                "summary": snapshot.get("summary"),
                "highlights": list(snapshot.get("highlights") or []),
                "participants": list(snapshot.get("participants") or []),
            },
            "followup_count": len(followups),
            "followups": [dict(item) for item in followups],
        }

        try:
            self._bus.publish_from_sync(
                self.FOLLOWUP_TOPIC,
                payload,
                priority=MessagePriority.HIGH,
                metadata={"component": "conversation_summary"},
            )
        except Exception:  # pragma: no cover - defensive logging only
            self._logger.exception(
                "Failed to publish follow-up event for conversation %s", state.conversation_id
            )

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


__all__ = ["ConversationSummaryWorker", "extract_followups"]
