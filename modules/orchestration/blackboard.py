"""Shared blackboard facilities for coordinating skill output.

This module implements a lightweight shared store that allows skills to post
hypotheses, claims and artifacts keyed by conversation or project.  Entries are
stored in-process and broadcast over the message bus so that other
subsystems—such as the GTK UI or external agents—can react to updates.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional

from ATLAS.messaging import (
    AgentBus,
    AgentMessage,
    MessagePriority,
    Subscription,
    get_agent_bus,
    BLACKBOARD_EVENT,
)

from ATLAS.utils.collections import dedupe_strings

_BLACKBOARD_EVENT_TOPIC = BLACKBOARD_EVENT.name
_SUPPORTED_CATEGORIES = {"hypothesis", "claim", "artifact"}
_SUPPORTED_SCOPE_TYPES = {"conversation", "project"}


@dataclass(slots=True)
class BlackboardEntry:
    """Structured record stored on the shared blackboard."""

    entry_id: str
    scope_id: str
    scope_type: str
    category: str
    title: str
    content: str
    author: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the entry."""

        return {
            "id": self.entry_id,
            "scope_id": self.scope_id,
            "scope_type": self.scope_type,
            "category": self.category,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "tags": list(self.tags),
            "metadata": dict(self.metadata or {}),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class BlackboardStore:
    """Thread-safe in-memory store for :class:`BlackboardEntry` objects."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, BlackboardEntry]] = defaultdict(dict)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------
    def create_entry(
        self,
        scope_id: str,
        *,
        category: str,
        title: str,
        content: str,
        scope_type: str = "conversation",
        author: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> BlackboardEntry:
        scope_key = _normalize_scope(scope_id, scope_type)
        category = _normalize_category(category)
        entry_id = uuid.uuid4().hex
        tag_tuple = _normalize_tags(tags)
        metadata = dict(metadata or {})
        tenant_value = metadata.get("tenant_id")
        if isinstance(tenant_value, str):
            tenant_token = tenant_value.strip()
            if tenant_token:
                metadata["tenant_id"] = tenant_token
            else:
                metadata.pop("tenant_id", None)
        elif tenant_value is not None:
            metadata.pop("tenant_id", None)
        if "tenant_id" not in metadata:
            metadata["tenant_id"] = "default"

        entry = BlackboardEntry(
            entry_id=entry_id,
            scope_id=scope_key.scope_id,
            scope_type=scope_key.scope_type,
            category=category,
            title=str(title).strip(),
            content=str(content).strip(),
            author=str(author).strip() if author else None,
            tags=tag_tuple,
            metadata=metadata,
        )

        with self._lock:
            self._entries[scope_key.composite][entry_id] = entry

        self._publish_event("created", entry)
        return entry

    def get_entry(
        self,
        entry_id: str,
        *,
        scope_id: str,
        scope_type: str = "conversation",
    ) -> Optional[BlackboardEntry]:
        scope_key = _normalize_scope(scope_id, scope_type)
        with self._lock:
            return self._entries.get(scope_key.composite, {}).get(entry_id)

    def update_entry(
        self,
        entry_id: str,
        *,
        scope_id: str,
        scope_type: str = "conversation",
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[BlackboardEntry]:
        scope_key = _normalize_scope(scope_id, scope_type)
        with self._lock:
            entry = self._entries.get(scope_key.composite, {}).get(entry_id)
            if entry is None:
                return None

            if title is not None:
                entry.title = str(title).strip()
            if content is not None:
                entry.content = str(content).strip()
            if tags is not None:
                entry.tags = _normalize_tags(tags)
            if metadata is not None:
                metadata_map = dict(metadata)
                existing_tenant = None
                if isinstance(entry.metadata, Mapping):
                    tenant_candidate = entry.metadata.get("tenant_id")
                    if isinstance(tenant_candidate, str) and tenant_candidate.strip():
                        existing_tenant = tenant_candidate.strip()

                tenant_value = metadata_map.get("tenant_id")
                if isinstance(tenant_value, str):
                    tenant_token = tenant_value.strip()
                    if tenant_token:
                        metadata_map["tenant_id"] = tenant_token
                    elif existing_tenant:
                        metadata_map["tenant_id"] = existing_tenant
                    else:
                        metadata_map.pop("tenant_id", None)
                elif existing_tenant:
                    metadata_map["tenant_id"] = existing_tenant
                else:
                    metadata_map.pop("tenant_id", None)

                if "tenant_id" not in metadata_map:
                    metadata_map["tenant_id"] = "default"

                entry.metadata = metadata_map
            entry.updated_at = time.time()

        self._publish_event("updated", entry)
        return entry

    def delete_entry(
        self,
        entry_id: str,
        *,
        scope_id: str,
        scope_type: str = "conversation",
    ) -> bool:
        scope_key = _normalize_scope(scope_id, scope_type)
        with self._lock:
            scoped_entries = self._entries.get(scope_key.composite)
            if not scoped_entries or entry_id not in scoped_entries:
                return False
            entry = scoped_entries.pop(entry_id)
            if not scoped_entries:
                self._entries.pop(scope_key.composite, None)

        self._publish_event("deleted", entry)
        return True

    def list_entries(
        self,
        scope_id: str,
        *,
        scope_type: str = "conversation",
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        scope_key = _normalize_scope(scope_id, scope_type)
        normalized_category = _normalize_category(category) if category else None
        with self._lock:
            scoped_entries = list(self._entries.get(scope_key.composite, {}).values())

        if normalized_category:
            scoped_entries = [
                entry for entry in scoped_entries if entry.category == normalized_category
            ]

        scoped_entries.sort(key=lambda entry: (entry.created_at, entry.entry_id))
        return [entry.as_dict() for entry in scoped_entries]

    def clear_scope(self, scope_id: str, *, scope_type: str = "conversation") -> None:
        scope_key = _normalize_scope(scope_id, scope_type)
        with self._lock:
            removed = list(self._entries.pop(scope_key.composite, {}).values())
        for entry in removed:
            self._publish_event("deleted", entry)

    def reset(self) -> None:
        """Remove all entries from the store."""

        with self._lock:
            removed = [entry for entries in self._entries.values() for entry in entries.values()]
            self._entries.clear()
        for entry in removed:
            self._publish_event("deleted", entry)

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------
    def get_summary(
        self,
        scope_id: str,
        *,
        scope_type: str = "conversation",
    ) -> Dict[str, Any]:
        entries = self.list_entries(scope_id, scope_type=scope_type)
        counts: Dict[str, int] = {category: 0 for category in _SUPPORTED_CATEGORIES}
        for entry in entries:
            counts[entry["category"]] = counts.get(entry["category"], 0) + 1

        return {
            "scope_id": scope_id,
            "scope_type": _normalize_scope(scope_id, scope_type).scope_type,
            "counts": counts,
            "entries": entries,
        }

    def client_for(self, scope_id: str, *, scope_type: str = "conversation") -> "BlackboardClient":
        scope_key = _normalize_scope(scope_id, scope_type)
        return BlackboardClient(self, scope_key.scope_id, scope_key.scope_type)

    # ------------------------------------------------------------------
    # Event propagation
    # ------------------------------------------------------------------
    def _publish_event(self, action: str, entry: BlackboardEntry) -> None:
        payload = {
            "action": action,
            "entry": entry.as_dict(),
        }
        topic = _topic_for(entry.scope_type, entry.scope_id)
        bus = get_agent_bus()
        
        # Create and publish message for main event topic
        main_msg = AgentMessage(
            channel=_BLACKBOARD_EVENT_TOPIC,
            payload=payload,
            priority=MessagePriority.NORMAL,
            headers={"scope": entry.scope_id, "scope_type": entry.scope_type},
        )
        bus.publish_from_sync(main_msg)
        
        # Create and publish message for scope-specific topic
        scope_msg = AgentMessage(
            channel=topic,
            payload=payload,
            priority=MessagePriority.NORMAL,
            headers={"scope": entry.scope_id, "scope_type": entry.scope_type},
        )
        bus.publish_from_sync(scope_msg)


class BlackboardClient:
    """Scope aware helper that wraps :class:`BlackboardStore`."""

    def __init__(self, store: BlackboardStore, scope_id: str, scope_type: str) -> None:
        self._store = store
        self.scope_id = scope_id
        self.scope_type = scope_type

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------
    def publish(
        self,
        category: str,
        title: str,
        content: str,
        *,
        author: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = self._store.create_entry(
            self.scope_id,
            scope_type=self.scope_type,
            category=category,
            title=title,
            content=content,
            author=author,
            tags=tags,
            metadata=metadata,
        )
        return entry.as_dict()

    def publish_hypothesis(self, title: str, content: str, **kwargs: Any) -> Dict[str, Any]:
        return self.publish("hypothesis", title, content, **kwargs)

    def publish_claim(self, title: str, content: str, **kwargs: Any) -> Dict[str, Any]:
        return self.publish("claim", title, content, **kwargs)

    def publish_artifact(self, title: str, content: str, **kwargs: Any) -> Dict[str, Any]:
        return self.publish("artifact", title, content, **kwargs)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def list_entries(self, *, category: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._store.list_entries(
            self.scope_id,
            scope_type=self.scope_type,
            category=category,
        )

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        entry = self._store.get_entry(entry_id, scope_id=self.scope_id, scope_type=self.scope_type)
        return entry.as_dict() if entry else None

    def update_entry(
        self,
        entry_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        entry = self._store.update_entry(
            entry_id,
            scope_id=self.scope_id,
            scope_type=self.scope_type,
            title=title,
            content=content,
            tags=tags,
            metadata=metadata,
        )
        return entry.as_dict() if entry else None

    def delete_entry(self, entry_id: str) -> bool:
        return self._store.delete_entry(
            entry_id,
            scope_id=self.scope_id,
            scope_type=self.scope_type,
        )

    def summary(self) -> Dict[str, Any]:
        return self._store.get_summary(self.scope_id, scope_type=self.scope_type)


@dataclass(slots=True)
class _ScopeKey:
    scope_id: str
    scope_type: str

    @property
    def composite(self) -> str:
        return f"{self.scope_type}:{self.scope_id}"


def _normalize_scope(scope_id: str, scope_type: str) -> _ScopeKey:
    scope_type = (scope_type or "conversation").strip().lower()
    if scope_type not in _SUPPORTED_SCOPE_TYPES:
        raise ValueError(f"Unsupported scope type: {scope_type}")
    scope_id = str(scope_id or "").strip()
    if not scope_id:
        raise ValueError("scope_id is required for blackboard operations")
    return _ScopeKey(scope_id=scope_id, scope_type=scope_type)


def _normalize_category(category: Optional[str]) -> str:
    if category is None:
        raise ValueError("category is required")
    normalized = str(category).strip().lower()
    if normalized not in _SUPPORTED_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")
    return normalized


def _normalize_tags(tags: Optional[Iterable[str]]) -> tuple[str, ...]:
    if not tags:
        return tuple()
    coerced = [str(tag) for tag in tags]
    return dedupe_strings(coerced)


def _topic_for(scope_type: str, scope_id: str) -> str:
    return f"blackboard.{scope_type}.{scope_id}"


_global_blackboard: Optional[BlackboardStore] = None


def configure_blackboard(store: Optional[BlackboardStore] = None) -> BlackboardStore:
    """Configure the global blackboard store instance."""

    global _global_blackboard
    _global_blackboard = store or BlackboardStore()
    return _global_blackboard


def get_blackboard() -> BlackboardStore:
    """Return the global blackboard instance, creating it if necessary."""

    global _global_blackboard
    if _global_blackboard is None:
        _global_blackboard = BlackboardStore()
    return _global_blackboard


async def stream_blackboard(scope_id: str, *, scope_type: str = "conversation") -> AsyncIterator[Dict[str, Any]]:
    """Asynchronous generator yielding blackboard events for the requested scope."""

    scope_key = _normalize_scope(scope_id, scope_type)
    queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
    topic = _topic_for(scope_key.scope_type, scope_key.scope_id)

    async def _handler(message: AgentMessage) -> None:
        payload = message.payload
        if isinstance(payload, Mapping):
            await queue.put(dict(payload))

    bus = get_agent_bus()
    subscription = await bus.subscribe(topic, _handler, concurrency=1)
    try:
        while True:
            payload = await queue.get()
            yield payload
    finally:
        await subscription.cancel()
