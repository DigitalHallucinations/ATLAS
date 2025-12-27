"""Compatibility layer that imports required SQLAlchemy primitives."""

from __future__ import annotations

import importlib.util


_SQLALCHEMY_MISSING = (
    "SQLAlchemy is required for the conversation store. Install it alongside pgvector "
    "and the PostgreSQL dialect extras (e.g. `pip install SQLAlchemy pgvector "
    "psycopg[binary]`)."
)


if importlib.util.find_spec("sqlalchemy") is None:  # pragma: no cover - import guard
    raise ImportError(_SQLALCHEMY_MISSING)


from sqlalchemy import and_, create_engine, delete, func, inspect, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload, sessionmaker


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
