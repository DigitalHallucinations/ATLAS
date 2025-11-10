"""Compatibility shims for optional SQLAlchemy dependencies.

This module mirrors the lightweight fallbacks that previously lived in
``repository.py`` so that the split modules can share a single source of
truth for SQLAlchemy primitives during import time.
"""

from __future__ import annotations

from typing import Any


def _raise_sqlalchemy(name: str) -> RuntimeError:
    return RuntimeError(
        f"SQLAlchemy function '{name}' is unavailable in this environment"
    )


try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy import (
        and_,
        create_engine,
        delete,
        func,
        inspect,
        or_,
        select,
        text,
    )
except Exception:  # pragma: no cover - lightweight fallbacks when SQLAlchemy is absent
    def create_engine(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("create_engine")

    def delete(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("delete")

    def func(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("func")

    def inspect(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("inspect")

    def select(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("select")

    def text(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("text")

    def and_(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("and_")

    def or_(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("or_")


try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.engine import Engine
except Exception:  # pragma: no cover - lightweight fallback when SQLAlchemy is absent
    class Engine:  # type: ignore[assignment]
        pass


try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.exc import IntegrityError
except Exception:  # pragma: no cover - lightweight fallback when SQLAlchemy is absent
    class IntegrityError(Exception):  # type: ignore[assignment]
        pass


try:  # pragma: no cover - optional SQLAlchemy dependency in test environments
    from sqlalchemy.orm import Session, joinedload, sessionmaker
except Exception:  # pragma: no cover - lightweight fallbacks when SQLAlchemy is absent
    class Session:  # type: ignore[assignment]
        pass

    class _Sessionmaker:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _raise_sqlalchemy("sessionmaker")

    def joinedload(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        raise _raise_sqlalchemy("joinedload")

    sessionmaker = _Sessionmaker  # type: ignore[assignment]
else:  # pragma: no cover - sanitize stubbed implementations
    if not isinstance(sessionmaker, type):

        class _Sessionmaker:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise _raise_sqlalchemy("sessionmaker")

        sessionmaker = _Sessionmaker  # type: ignore[assignment]


__all__ = [
    "Engine",
    "IntegrityError",
    "Session",
    "and_",
    "create_engine",
    "delete",
    "func",
    "inspect",
    "joinedload",
    "or_",
    "select",
    "sessionmaker",
    "text",
]
