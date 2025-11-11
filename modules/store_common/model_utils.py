"""Shared utility helpers for SQLAlchemy store models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

__all__ = ["generate_uuid", "utcnow"]


def generate_uuid() -> uuid.UUID:
    """Return a new random UUID."""

    return uuid.uuid4()


def utcnow() -> datetime:
    """Return the current time with UTC timezone information."""

    return datetime.now(timezone.utc)
