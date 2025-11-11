"""Utilities shared by store repositories."""

from __future__ import annotations

import contextlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, Mapping, Optional, Type, TypeVar

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy import and_, or_, select
    from sqlalchemy.exc import IntegrityError
except Exception:  # pragma: no cover - fallback when SQLAlchemy is absent
    def select(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy select is unavailable in this environment")

    def and_(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy and_ is unavailable in this environment")

    def or_(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy or_ is unavailable in this environment")

    class IntegrityError(Exception):
        """Fallback IntegrityError when SQLAlchemy is not installed."""

        pass

try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.orm import Session, joinedload, sessionmaker
except Exception:  # pragma: no cover - fallback when SQLAlchemy is absent
    class _Session:  # pragma: no cover - lightweight placeholder type
        pass

    class _Sessionmaker:  # pragma: no cover - lightweight placeholder type
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("SQLAlchemy sessionmaker is unavailable in this environment")

    def joinedload(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("SQLAlchemy joinedload is unavailable in this environment")

    Session = _Session  # type: ignore[assignment]
    sessionmaker = _Sessionmaker  # type: ignore[assignment]
else:  # pragma: no cover - sanitize stubbed implementations
    if not isinstance(sessionmaker, type):  # pragma: no cover - test stub compatibility
        class _Sessionmaker:  # lightweight placeholder mirroring SQLAlchemy API
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("SQLAlchemy sessionmaker is unavailable in this environment")

        sessionmaker = _Sessionmaker  # type: ignore[assignment]


@contextlib.contextmanager
def _session_scope(session_factory: sessionmaker) -> Iterator[Session]:
    """Provide a transactional scope around a series of operations."""

    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_uuid(value: Any | None) -> Optional[uuid.UUID]:
    if value is None or value == "":
        return None
    if isinstance(value, uuid.UUID):
        return value
    if isinstance(value, bytes):
        return uuid.UUID(bytes=value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return uuid.UUID(text)
    except ValueError:
        return uuid.UUID(hex=text.replace("-", ""))


def _normalize_tenant_id(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError("Tenant identifier must be provided")
    return text


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        candidate = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Datetime value cannot be empty")
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            candidate = datetime.fromisoformat(normalized)
        except ValueError:
            candidate = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    if candidate.tzinfo is None:
        candidate = candidate.replace(tzinfo=timezone.utc)
    return candidate.astimezone(timezone.utc)


def _coerce_optional_dt(value: Any | None) -> Optional[datetime]:
    if value in (None, ""):
        return None
    return _coerce_dt(value)


def _dt_to_iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    normalized = moment.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _normalize_meta(
    metadata: Mapping[str, Any] | None,
    *,
    error_message: str = "Metadata must be a mapping",
) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(error_message)
    return dict(metadata)


def _normalize_enum_value(
    value: Any | None,
    enum_cls: Type[EnumT],
    default_member: EnumT,
) -> EnumT:
    """Normalize arbitrary input into an enum member.

    Parameters
    ----------
    value:
        Raw value that may be ``None``, an enum member, a string, or other primitive.
    enum_cls:
        Enumeration class to coerce into.
    default_member:
        Fallback enum member used when ``value`` is ``None`` or empty.

    Returns
    -------
    EnumT
        A member of ``enum_cls`` corresponding to the provided ``value``.
    """

    if value is None:
        return default_member
    if isinstance(value, enum_cls):
        return value

    text = str(value).strip()
    if not text:
        return default_member

    candidates: list[str] = []
    # For string-based enums we prefer a case-insensitive lookup.
    if issubclass(enum_cls, str):
        lowered = text.lower()
        if lowered not in candidates:
            candidates.append(lowered)
    if text not in candidates:
        candidates.append(text)
    uppered = text.upper()
    if uppered not in candidates:
        candidates.append(uppered)

    for candidate in candidates:
        try:
            return enum_cls(candidate)
        except ValueError:
            continue

    normalized_name = text.lower()
    for member in enum_cls:
        if member.name.lower() == normalized_name:
            return member

    # Allow the enum class to raise the canonical ValueError for unknown values.
    return enum_cls(text)


__all__ = [
    "IntegrityError",
    "Session",
    "and_",
    "joinedload",
    "or_",
    "select",
    "sessionmaker",
    "_coerce_dt",
    "_coerce_optional_dt",
    "_coerce_uuid",
    "_dt_to_iso",
    "_normalize_enum_value",
    "_normalize_meta",
    "_normalize_tenant_id",
    "_session_scope",
]
