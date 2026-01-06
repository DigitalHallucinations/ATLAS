"""Shared helpers for service modules in data stores."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple, Type, TypeVar

from modules.Tools.tool_event_system import publish_bus_event

StatusT = TypeVar("StatusT", bound=Enum)


def coerce_enum_value(value: Any, enum_type: Type[StatusT]) -> StatusT:
    """Normalize arbitrary enum representations into a store enum instance."""

    if isinstance(value, enum_type):
        return value
    text = str(value).strip().lower()
    return enum_type(text)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse timestamp inputs into timezone-aware UTC datetimes."""

    if isinstance(value, datetime):
        timestamp = value
    elif value is None:
        return None
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            timestamp = datetime.fromisoformat(text)
        except ValueError:
            return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def normalize_owner_identifier(value: Any) -> Optional[str]:
    """Normalize owner identifiers into canonical UUID strings."""

    if value is None or value == "":
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return str(uuid.UUID(bytes=value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return str(uuid.UUID(hex=text.replace("-", "")))


class LifecycleServiceBase:
    """Mixin providing common lifecycle service utilities."""

    def __init__(
        self,
        *,
        event_emitter: Optional[Callable[[str, Mapping[str, Any]], Any]] = None,
    ) -> None:
        if event_emitter is None:
            self._emit: Callable[[str, Mapping[str, Any]], Any] = self._default_emit
        else:
            self._emit = event_emitter

    @staticmethod
    def _default_emit(event_name: str, payload: Mapping[str, Any]) -> None:
        publish_bus_event(event_name, dict(payload))

    def apply_owner_change(
        self,
        *,
        snapshot: Mapping[str, Any],
        changes: Mapping[str, Any],
        owner_field: str = "owner_id",
    ) -> Tuple[dict[str, Any], bool]:
        """Normalize owner updates and flag whether the owner changed."""

        change_payload = dict(changes)

        owner_changed = False
        if owner_field in change_payload:
            existing_owner = normalize_owner_identifier(snapshot.get(owner_field))
            requested_owner = normalize_owner_identifier(change_payload[owner_field])
            if existing_owner == requested_owner:
                change_payload.pop(owner_field)
            else:
                owner_changed = True
        return change_payload, owner_changed

    def prepare_noop_update(
        self,
        *,
        snapshot: Mapping[str, Any],
        expected_updated_at: Any | None,
        concurrency_error: Type[Exception],
        error_message: str,
        events_field: Optional[str] = "events",
    ) -> dict[str, Any]:
        """Validate optimistic concurrency expectations when no changes apply."""

        if expected_updated_at is not None:
            expected_timestamp = parse_timestamp(expected_updated_at)
            current_timestamp = parse_timestamp(snapshot.get("updated_at"))
            if (
                expected_timestamp is None
                or current_timestamp is None
                or current_timestamp != expected_timestamp
            ):
                raise concurrency_error(error_message)
        snapshot_payload = dict(snapshot)
        if events_field is not None:
            snapshot_payload.setdefault(events_field, [])
        return snapshot_payload


__all__ = [
    "LifecycleServiceBase",
    "coerce_enum_value",
    "normalize_owner_identifier",
    "parse_timestamp",
]
