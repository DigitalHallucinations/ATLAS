"""Schema creation utilities for the ATLAS Master Calendar store.

This module provides functions to create and initialize the calendar
database schema, including seeding built-in categories.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    CalendarCategoryModel,
    ensure_calendar_schema,
)
from .dataclasses import BUILTIN_CATEGORIES, SyncDirection


def create_calendar_engine(
    connection_string: str,
    *,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_pre_ping: bool = True,
    **kwargs: Any,
) -> Engine:
    """Create a SQLAlchemy engine configured for the calendar store.

    Parameters
    ----------
    connection_string
        Database connection URL (e.g., postgresql://user:pass@host/db)
    echo
        If True, log all SQL statements
    pool_size
        Number of connections to keep in the pool
    max_overflow
        Maximum overflow connections beyond pool_size
    pool_pre_ping
        Test connections before use to handle stale connections
    **kwargs
        Additional arguments passed to create_engine

    Returns
    -------
    Engine
        Configured SQLAlchemy engine
    """
    # Adjust pool settings for SQLite which doesn't support pooling
    if connection_string.startswith("sqlite"):
        return create_engine(
            connection_string,
            echo=echo,
            **kwargs,
        )

    return create_engine(
        connection_string,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
        **kwargs,
    )


def create_schema(engine: Engine, *, seed_categories: bool = True) -> None:
    """Create all calendar tables and optionally seed built-in categories.

    Parameters
    ----------
    engine
        SQLAlchemy engine connected to the database
    seed_categories
        If True, insert built-in categories (Work, Personal, etc.)
    """
    # Create all tables defined in the models
    Base.metadata.create_all(engine)

    # Set up PostgreSQL-specific features (triggers, GIN indexes)
    ensure_calendar_schema(engine)

    # Seed built-in categories if requested
    if seed_categories:
        seed_builtin_categories(engine)


def seed_builtin_categories(engine: Engine) -> None:
    """Insert built-in categories if they don't already exist.

    This is idempotent - existing categories are not modified.

    Parameters
    ----------
    engine
        SQLAlchemy engine connected to the database
    """
    Session = sessionmaker(bind=engine)

    with Session() as session:
        for category_data in BUILTIN_CATEGORIES:
            slug = category_data["slug"]

            # Check if category already exists
            existing = (
                session.query(CalendarCategoryModel)
                .filter(CalendarCategoryModel.slug == slug)
                .first()
            )

            if existing is None:
                category = CalendarCategoryModel(
                    name=category_data["name"],
                    slug=category_data["slug"],
                    color=category_data["color"],
                    icon=category_data.get("icon"),
                    is_builtin=category_data["is_builtin"],
                    is_default=category_data["is_default"],
                    is_readonly=category_data.get("is_readonly", False),
                    sort_order=category_data["sort_order"],
                    sync_direction=SyncDirection.BIDIRECTIONAL,
                )
                session.add(category)

        session.commit()


def reset_schema(engine: Engine) -> None:
    """Drop and recreate all calendar tables.

    WARNING: This will delete all calendar data!

    Parameters
    ----------
    engine
        SQLAlchemy engine connected to the database
    """
    # Drop PostgreSQL-specific objects first
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            conn.execute(text("DROP TRIGGER IF EXISTS calendar_events_search_update ON calendar_events"))
            conn.execute(text("DROP FUNCTION IF EXISTS calendar_events_search_trigger()"))
            conn.commit()

    # Drop tables in reverse dependency order
    tables_to_drop = [
        "calendar_sync_state",
        "calendar_import_mappings",
        "calendar_events",
        "calendar_categories",
    ]

    with engine.connect() as conn:
        for table_name in tables_to_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        conn.commit()

    # Recreate schema
    create_schema(engine, seed_categories=True)


__all__ = [
    "create_calendar_engine",
    "create_schema",
    "seed_builtin_categories",
    "reset_schema",
]
