"""Shared helpers for service modules in data stores."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Type, TypeVar

StatusT = TypeVar("StatusT")


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
