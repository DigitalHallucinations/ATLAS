"""Repository layer for the ATLAS Master Calendar store.

This module provides the CalendarStoreRepository class which handles
all CRUD operations for calendar categories, events, import mappings,
and sync state.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy import and_, or_, func, text
from sqlalchemy.orm import Session, joinedload, sessionmaker

from modules.store_common.repository_utils import (
    BaseStoreRepository,
    _coerce_dt,
    _coerce_optional_dt,
    _coerce_uuid,
    _dt_to_iso,
    _normalize_enum_value,
    _normalize_meta,
)

from .models import (
    Base,
    CalendarCategoryModel,
    CalendarEventModel,
    CalendarImportMappingModel,
    CalendarSyncStateModel,
    ensure_calendar_schema,
)
from .link_models import (
    JobEventLink,
    TaskEventLink,
    LinkType,
    SyncBehavior,
    ensure_link_schema,
)
from .dataclasses import (
    Attendee,
    CalendarReminder,
    EventStatus,
    EventVisibility,
    BusyStatus,
    SyncDirection,
    SyncStatus,
    ReminderMethod,
)


# ============================================================================
# Exceptions
# ============================================================================


class CalendarStoreError(RuntimeError):
    """Base class for calendar store errors."""


class CategoryNotFoundError(CalendarStoreError):
    """Raised when a category cannot be found."""


class EventNotFoundError(CalendarStoreError):
    """Raised when an event cannot be found."""


class CategorySlugExistsError(CalendarStoreError):
    """Raised when attempting to create a category with a duplicate slug."""


class ReadOnlyCategoryError(CalendarStoreError):
    """Raised when attempting to modify a read-only category."""


# ============================================================================
# Normalization helpers
# ============================================================================


def _normalize_slug(name: str, existing_slug: Optional[str] = None) -> str:
    """Generate a URL-safe slug from a name."""
    if existing_slug:
        return existing_slug
    # Convert to lowercase, replace spaces with hyphens, remove special chars
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _normalize_color(value: Any) -> str:
    """Normalize a color value to hex format."""
    if value is None:
        return "#4285F4"
    text = str(value).strip()
    if not text:
        return "#4285F4"
    # Validate hex color format
    if re.match(r"^#[0-9A-Fa-f]{6}$", text):
        return text.upper()
    if re.match(r"^[0-9A-Fa-f]{6}$", text):
        return f"#{text.upper()}"
    return "#4285F4"


def _normalize_status(value: Any) -> EventStatus:
    return _normalize_enum_value(value, EventStatus, EventStatus.CONFIRMED)


def _normalize_visibility(value: Any) -> EventVisibility:
    return _normalize_enum_value(value, EventVisibility, EventVisibility.PUBLIC)


def _normalize_busy_status(value: Any) -> BusyStatus:
    return _normalize_enum_value(value, BusyStatus, BusyStatus.BUSY)


def _normalize_sync_direction(value: Any) -> SyncDirection:
    return _normalize_enum_value(value, SyncDirection, SyncDirection.BIDIRECTIONAL)


def _normalize_sync_status(value: Any) -> SyncStatus:
    return _normalize_enum_value(value, SyncStatus, SyncStatus.SYNCED)


def _normalize_tags(value: Any) -> List[str]:
    """Normalize tags to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t).strip() for t in value if str(t).strip()]
    return []


def _normalize_attendees(value: Any) -> List[Dict[str, Any]]:
    """Normalize attendees to a list of dictionaries."""
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, Attendee):
                result.append(item.to_dict())
            elif isinstance(item, dict):
                result.append(item)
        return result
    return []


def _normalize_reminders(value: Any) -> List[Dict[str, Any]]:
    """Normalize reminders to a list of dictionaries."""
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, CalendarReminder):
                result.append(item.to_dict())
            elif isinstance(item, dict):
                result.append(item)
        return result
    return []


# ============================================================================
# Serialization helpers
# ============================================================================


def _serialize_category(category: CalendarCategoryModel) -> Dict[str, Any]:
    """Serialize a category model to a dictionary."""
    return {
        "id": str(category.id),
        "name": category.name,
        "slug": category.slug,
        "color": category.color,
        "icon": category.icon,
        "description": category.description,
        "is_builtin": category.is_builtin,
        "is_visible": category.is_visible,
        "is_default": category.is_default,
        "is_readonly": category.is_readonly,
        "sort_order": category.sort_order,
        "sync_direction": category.sync_direction.value if category.sync_direction else "bidirectional",
        "created_at": _dt_to_iso(category.created_at),
        "updated_at": _dt_to_iso(category.updated_at),
    }


def _serialize_event(event: CalendarEventModel) -> Dict[str, Any]:
    """Serialize an event model to a dictionary."""
    return {
        "id": str(event.id),
        "external_id": event.external_id,
        "external_source": event.external_source,
        "title": event.title,
        "description": event.description,
        "location": event.location,
        "start_time": _dt_to_iso(event.start_time),
        "end_time": _dt_to_iso(event.end_time),
        "timezone": event.timezone,
        "is_all_day": event.is_all_day,
        "recurrence_rule": event.recurrence_rule,
        "recurrence_id": _dt_to_iso(event.recurrence_id),
        "original_start": _dt_to_iso(event.original_start),
        "category_id": str(event.category_id) if event.category_id else None,
        "tags": list(event.tags) if event.tags else [],
        "color_override": event.color_override,
        "status": event.status.value if event.status else "confirmed",
        "visibility": event.visibility.value if event.visibility else "public",
        "busy_status": event.busy_status.value if event.busy_status else "busy",
        "organizer": event.organizer,
        "attendees": list(event.attendees) if event.attendees else [],
        "reminders": list(event.reminders) if event.reminders else [],
        "url": event.url,
        "attachments": list(event.attachments) if event.attachments else [],
        "custom_properties": dict(event.custom_properties) if event.custom_properties else {},
        "etag": event.etag,
        "sync_status": event.sync_status.value if event.sync_status else "synced",
        "last_synced_at": _dt_to_iso(event.last_synced_at),
        "created_at": _dt_to_iso(event.created_at),
        "updated_at": _dt_to_iso(event.updated_at),
    }


def _serialize_import_mapping(mapping: CalendarImportMappingModel) -> Dict[str, Any]:
    """Serialize an import mapping model to a dictionary."""
    return {
        "id": str(mapping.id),
        "source_type": mapping.source_type,
        "source_account": mapping.source_account,
        "source_calendar": mapping.source_calendar,
        "target_category_id": str(mapping.target_category_id),
        "created_at": _dt_to_iso(mapping.created_at),
    }


def _serialize_sync_state(state: CalendarSyncStateModel) -> Dict[str, Any]:
    """Serialize a sync state model to a dictionary."""
    return {
        "id": str(state.id),
        "source_type": state.source_type,
        "source_account": state.source_account,
        "source_calendar": state.source_calendar,
        "sync_token": state.sync_token,
        "last_sync_at": _dt_to_iso(state.last_sync_at),
        "last_sync_status": state.last_sync_status,
        "error_message": state.error_message,
        "created_at": _dt_to_iso(state.created_at),
        "updated_at": _dt_to_iso(state.updated_at),
    }


# ============================================================================
# Repository
# ============================================================================


class CalendarStoreRepository(BaseStoreRepository):
    """Repository for calendar store CRUD operations."""

    def __init__(self, session_factory: sessionmaker) -> None:
        super().__init__(session_factory)

    # -- Schema helpers -------------------------------------------------

    def _get_schema_metadata(self) -> Any:
        return Base.metadata

    def _ensure_additional_schema(self, engine: Any) -> None:
        ensure_calendar_schema(engine)
        ensure_link_schema(engine)

    # ========================================================================
    # Category operations
    # ========================================================================

    def list_categories(
        self,
        *,
        include_hidden: bool = False,
        include_builtin: bool = True,
    ) -> List[Dict[str, Any]]:
        """List all calendar categories.

        Parameters
        ----------
        include_hidden
            If False, exclude categories where is_visible=False
        include_builtin
            If False, exclude built-in system categories

        Returns
        -------
        List of serialized category dictionaries
        """
        with self._session_scope() as session:
            query = session.query(CalendarCategoryModel).order_by(
                CalendarCategoryModel.sort_order,
                CalendarCategoryModel.name,
            )

            if not include_hidden:
                query = query.filter(CalendarCategoryModel.is_visible == True)
            if not include_builtin:
                query = query.filter(CalendarCategoryModel.is_builtin == False)

            categories = query.all()
            return [_serialize_category(cat) for cat in categories]

    def get_category(self, category_id: Any) -> Dict[str, Any]:
        """Get a category by ID.

        Raises
        ------
        CategoryNotFoundError
            If the category does not exist
        """
        cat_uuid = _coerce_uuid(category_id)
        if cat_uuid is None:
            raise CategoryNotFoundError("Invalid category identifier")

        with self._session_scope() as session:
            category = session.get(CalendarCategoryModel, cat_uuid)
            if category is None:
                raise CategoryNotFoundError(f"Category {category_id} not found")
            return _serialize_category(category)

    def get_category_by_slug(self, slug: str) -> Dict[str, Any]:
        """Get a category by its slug.

        Raises
        ------
        CategoryNotFoundError
            If the category does not exist
        """
        with self._session_scope() as session:
            category = (
                session.query(CalendarCategoryModel)
                .filter(CalendarCategoryModel.slug == slug)
                .first()
            )
            if category is None:
                raise CategoryNotFoundError(f"Category with slug '{slug}' not found")
            return _serialize_category(category)

    def get_default_category(self) -> Optional[Dict[str, Any]]:
        """Get the default category for new events."""
        with self._session_scope() as session:
            category = (
                session.query(CalendarCategoryModel)
                .filter(CalendarCategoryModel.is_default == True)
                .first()
            )
            if category is None:
                # Fall back to Personal if no default is set
                category = (
                    session.query(CalendarCategoryModel)
                    .filter(CalendarCategoryModel.slug == "personal")
                    .first()
                )
            if category is None:
                return None
            return _serialize_category(category)

    def create_category(
        self,
        *,
        name: str,
        slug: Optional[str] = None,
        color: str = "#4285F4",
        icon: Optional[str] = None,
        description: Optional[str] = None,
        is_visible: bool = True,
        is_default: bool = False,
        sort_order: int = 0,
        sync_direction: Any = SyncDirection.BIDIRECTIONAL,
    ) -> Dict[str, Any]:
        """Create a new custom category.

        Raises
        ------
        CategorySlugExistsError
            If a category with the same slug already exists
        """
        normalized_slug = _normalize_slug(name, slug)
        normalized_color = _normalize_color(color)
        normalized_sync_dir = _normalize_sync_direction(sync_direction)

        with self._session_scope() as session:
            # Check for existing slug
            existing = (
                session.query(CalendarCategoryModel)
                .filter(CalendarCategoryModel.slug == normalized_slug)
                .first()
            )
            if existing is not None:
                raise CategorySlugExistsError(
                    f"Category with slug '{normalized_slug}' already exists"
                )

            # If setting as default, clear other defaults
            if is_default:
                session.query(CalendarCategoryModel).update({"is_default": False})

            category = CalendarCategoryModel(
                name=name.strip(),
                slug=normalized_slug,
                color=normalized_color,
                icon=icon,
                description=description,
                is_builtin=False,
                is_visible=is_visible,
                is_default=is_default,
                is_readonly=False,
                sort_order=sort_order,
                sync_direction=normalized_sync_dir,
            )
            session.add(category)
            session.flush()
            session.refresh(category)
            return _serialize_category(category)

    def update_category(
        self,
        category_id: Any,
        *,
        changes: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Update a category's properties.

        Parameters
        ----------
        category_id
            The category ID to update
        changes
            Dictionary of fields to update

        Raises
        ------
        CategoryNotFoundError
            If the category does not exist
        ReadOnlyCategoryError
            If attempting to modify a read-only category
        """
        cat_uuid = _coerce_uuid(category_id)
        if cat_uuid is None:
            raise CategoryNotFoundError("Invalid category identifier")

        with self._session_scope() as session:
            category = session.get(CalendarCategoryModel, cat_uuid)
            if category is None:
                raise CategoryNotFoundError(f"Category {category_id} not found")

            # Check if read-only (but allow visibility changes)
            if category.is_readonly and not set(changes.keys()).issubset(
                {"is_visible"}
            ):
                raise ReadOnlyCategoryError(
                    "Cannot modify read-only category (except visibility)"
                )

            # Apply changes
            for field, value in changes.items():
                if field == "name":
                    category.name = str(value).strip()
                elif field == "slug":
                    new_slug = _normalize_slug(str(value))
                    # Check for conflicts
                    existing = (
                        session.query(CalendarCategoryModel)
                        .filter(
                            CalendarCategoryModel.slug == new_slug,
                            CalendarCategoryModel.id != cat_uuid,
                        )
                        .first()
                    )
                    if existing:
                        raise CategorySlugExistsError(
                            f"Category with slug '{new_slug}' already exists"
                        )
                    category.slug = new_slug
                elif field == "color":
                    category.color = _normalize_color(value)
                elif field == "icon":
                    category.icon = value
                elif field == "description":
                    category.description = value
                elif field == "is_visible":
                    category.is_visible = bool(value)
                elif field == "is_default":
                    if value:
                        # Clear other defaults first
                        session.query(CalendarCategoryModel).filter(
                            CalendarCategoryModel.id != cat_uuid
                        ).update({"is_default": False})
                    category.is_default = bool(value)
                elif field == "sort_order":
                    category.sort_order = int(value)
                elif field == "sync_direction":
                    category.sync_direction = _normalize_sync_direction(value)

            session.flush()
            session.refresh(category)
            return _serialize_category(category)

    def delete_category(self, category_id: Any) -> None:
        """Delete a category.

        Events in this category will have their category_id set to NULL.

        Raises
        ------
        CategoryNotFoundError
            If the category does not exist
        ReadOnlyCategoryError
            If attempting to delete a built-in category
        """
        cat_uuid = _coerce_uuid(category_id)
        if cat_uuid is None:
            raise CategoryNotFoundError("Invalid category identifier")

        with self._session_scope() as session:
            category = session.get(CalendarCategoryModel, cat_uuid)
            if category is None:
                raise CategoryNotFoundError(f"Category {category_id} not found")

            if category.is_builtin:
                raise ReadOnlyCategoryError("Cannot delete built-in categories")

            session.delete(category)

    def set_default_category(self, category_id: Any) -> Dict[str, Any]:
        """Set a category as the default for new events."""
        cat_uuid = _coerce_uuid(category_id)
        if cat_uuid is None:
            raise CategoryNotFoundError("Invalid category identifier")

        with self._session_scope() as session:
            category = session.get(CalendarCategoryModel, cat_uuid)
            if category is None:
                raise CategoryNotFoundError(f"Category {category_id} not found")

            # Clear all defaults
            session.query(CalendarCategoryModel).update({"is_default": False})

            # Set new default
            category.is_default = True
            session.flush()
            session.refresh(category)
            return _serialize_category(category)

    def toggle_category_visibility(
        self, category_id: Any, *, visible: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Toggle or set category visibility.

        Parameters
        ----------
        category_id
            The category to modify
        visible
            If provided, set to this value; otherwise toggle current state
        """
        cat_uuid = _coerce_uuid(category_id)
        if cat_uuid is None:
            raise CategoryNotFoundError("Invalid category identifier")

        with self._session_scope() as session:
            category = session.get(CalendarCategoryModel, cat_uuid)
            if category is None:
                raise CategoryNotFoundError(f"Category {category_id} not found")

            if visible is None:
                category.is_visible = not category.is_visible
            else:
                category.is_visible = bool(visible)

            session.flush()
            session.refresh(category)
            return _serialize_category(category)

    # ========================================================================
    # Event operations
    # ========================================================================

    def list_events(
        self,
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        category_id: Optional[Any] = None,
        category_ids: Optional[Sequence[Any]] = None,
        include_hidden_categories: bool = False,
        status: Optional[Any] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List events with optional filtering.

        Parameters
        ----------
        start
            Filter events starting at or after this time
        end
            Filter events ending at or before this time
        category_id
            Filter by single category
        category_ids
            Filter by multiple categories
        include_hidden_categories
            If False, exclude events in hidden categories
        status
            Filter by event status
        limit
            Maximum number of events to return
        offset
            Number of events to skip (for pagination)

        Returns
        -------
        List of serialized event dictionaries
        """
        start_dt = _coerce_optional_dt(start)
        end_dt = _coerce_optional_dt(end)

        with self._session_scope() as session:
            query = (
                session.query(CalendarEventModel)
                .options(joinedload(CalendarEventModel.category))
                .order_by(CalendarEventModel.start_time)
            )

            # Date range filter
            if start_dt is not None:
                query = query.filter(CalendarEventModel.end_time >= start_dt)
            if end_dt is not None:
                query = query.filter(CalendarEventModel.start_time <= end_dt)

            # Category filter
            if category_id is not None:
                cat_uuid = _coerce_uuid(category_id)
                query = query.filter(CalendarEventModel.category_id == cat_uuid)
            elif category_ids is not None:
                cat_uuids = [_coerce_uuid(cid) for cid in category_ids if _coerce_uuid(cid)]
                if cat_uuids:
                    query = query.filter(CalendarEventModel.category_id.in_(cat_uuids))

            # Hidden category filter
            if not include_hidden_categories:
                query = query.outerjoin(CalendarCategoryModel).filter(
                    or_(
                        CalendarCategoryModel.is_visible == True,
                        CalendarEventModel.category_id == None,
                    )
                )

            # Status filter
            if status is not None:
                normalized_status = _normalize_status(status)
                query = query.filter(CalendarEventModel.status == normalized_status)

            # Pagination
            query = query.offset(offset).limit(limit)

            events = query.all()
            return [_serialize_event(event) for event in events]

    def get_event(self, event_id: Any) -> Dict[str, Any]:
        """Get an event by ID.

        Raises
        ------
        EventNotFoundError
            If the event does not exist
        """
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            raise EventNotFoundError("Invalid event identifier")

        with self._session_scope() as session:
            event = (
                session.query(CalendarEventModel)
                .options(joinedload(CalendarEventModel.category))
                .filter(CalendarEventModel.id == event_uuid)
                .first()
            )
            if event is None:
                raise EventNotFoundError(f"Event {event_id} not found")
            return _serialize_event(event)

    def get_event_by_external_id(
        self, external_source: str, external_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get an event by its external source and ID.

        Returns None if not found (does not raise).
        """
        with self._session_scope() as session:
            event = (
                session.query(CalendarEventModel)
                .options(joinedload(CalendarEventModel.category))
                .filter(
                    CalendarEventModel.external_source == external_source,
                    CalendarEventModel.external_id == external_id,
                )
                .first()
            )
            if event is None:
                return None
            return _serialize_event(event)

    def create_event(
        self,
        *,
        title: str,
        start_time: Any,
        end_time: Any,
        external_id: Optional[str] = None,
        external_source: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        timezone: str = "UTC",
        is_all_day: bool = False,
        recurrence_rule: Optional[str] = None,
        category_id: Optional[Any] = None,
        tags: Optional[Any] = None,
        color_override: Optional[str] = None,
        status: Any = EventStatus.CONFIRMED,
        visibility: Any = EventVisibility.PUBLIC,
        busy_status: Any = BusyStatus.BUSY,
        organizer: Optional[Dict[str, Any]] = None,
        attendees: Optional[Any] = None,
        reminders: Optional[Any] = None,
        url: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
        etag: Optional[str] = None,
        sync_status: Any = SyncStatus.SYNCED,
    ) -> Dict[str, Any]:
        """Create a new calendar event.

        Parameters
        ----------
        title
            Event title (required)
        start_time
            Event start time (required)
        end_time
            Event end time (required)
        category_id
            Category to assign (uses default if not provided)
        ...other parameters match the event model

        Returns
        -------
        Serialized event dictionary
        """
        start_dt = _coerce_dt(start_time)
        end_dt = _coerce_dt(end_time)
        cat_uuid = _coerce_uuid(category_id)

        with self._session_scope() as session:
            # Use default category if none specified
            if cat_uuid is None:
                default_cat = (
                    session.query(CalendarCategoryModel)
                    .filter(CalendarCategoryModel.is_default == True)
                    .first()
                )
                if default_cat:
                    cat_uuid = default_cat.id

            event = CalendarEventModel(
                title=title.strip(),
                start_time=start_dt,
                end_time=end_dt,
                external_id=external_id,
                external_source=external_source,
                description=description,
                location=location,
                timezone=timezone,
                is_all_day=is_all_day,
                recurrence_rule=recurrence_rule,
                category_id=cat_uuid,
                tags=_normalize_tags(tags),
                color_override=_normalize_color(color_override) if color_override else None,
                status=_normalize_status(status),
                visibility=_normalize_visibility(visibility),
                busy_status=_normalize_busy_status(busy_status),
                organizer=organizer,
                attendees=_normalize_attendees(attendees),
                reminders=_normalize_reminders(reminders),
                url=url,
                attachments=attachments or [],
                custom_properties=custom_properties or {},
                etag=etag,
                sync_status=_normalize_sync_status(sync_status),
            )
            session.add(event)
            session.flush()
            session.refresh(event)
            return _serialize_event(event)

    def update_event(
        self,
        event_id: Any,
        *,
        changes: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Update an event's properties.

        Parameters
        ----------
        event_id
            The event ID to update
        changes
            Dictionary of fields to update

        Raises
        ------
        EventNotFoundError
            If the event does not exist
        """
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            raise EventNotFoundError("Invalid event identifier")

        with self._session_scope() as session:
            event = session.get(CalendarEventModel, event_uuid)
            if event is None:
                raise EventNotFoundError(f"Event {event_id} not found")

            # Apply changes
            for field, value in changes.items():
                if field == "title":
                    event.title = str(value).strip()
                elif field == "description":
                    event.description = value
                elif field == "location":
                    event.location = value
                elif field == "start_time":
                    event.start_time = _coerce_dt(value)
                elif field == "end_time":
                    event.end_time = _coerce_dt(value)
                elif field == "timezone":
                    event.timezone = str(value)
                elif field == "is_all_day":
                    event.is_all_day = bool(value)
                elif field == "recurrence_rule":
                    event.recurrence_rule = value
                elif field == "category_id":
                    event.category_id = _coerce_uuid(value)
                elif field == "tags":
                    event.tags = _normalize_tags(value)
                elif field == "color_override":
                    event.color_override = _normalize_color(value) if value else None
                elif field == "status":
                    event.status = _normalize_status(value)
                elif field == "visibility":
                    event.visibility = _normalize_visibility(value)
                elif field == "busy_status":
                    event.busy_status = _normalize_busy_status(value)
                elif field == "organizer":
                    event.organizer = value
                elif field == "attendees":
                    event.attendees = _normalize_attendees(value)
                elif field == "reminders":
                    event.reminders = _normalize_reminders(value)
                elif field == "url":
                    event.url = value
                elif field == "attachments":
                    event.attachments = value or []
                elif field == "custom_properties":
                    event.custom_properties = value or {}
                elif field == "etag":
                    event.etag = value
                elif field == "sync_status":
                    event.sync_status = _normalize_sync_status(value)
                elif field == "last_synced_at":
                    event.last_synced_at = _coerce_optional_dt(value)

            session.flush()
            session.refresh(event)
            return _serialize_event(event)

    def delete_event(self, event_id: Any) -> None:
        """Delete an event.

        Raises
        ------
        EventNotFoundError
            If the event does not exist
        """
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            raise EventNotFoundError("Invalid event identifier")

        with self._session_scope() as session:
            event = session.get(CalendarEventModel, event_uuid)
            if event is None:
                raise EventNotFoundError(f"Event {event_id} not found")
            session.delete(event)

    def search_events(
        self,
        query: str,
        *,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        category_id: Optional[Any] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Full-text search for events.

        Uses PostgreSQL's full-text search when available,
        falls back to LIKE queries for other databases.

        Parameters
        ----------
        query
            Search query string
        start
            Filter events after this time
        end
            Filter events before this time
        category_id
            Filter by category
        limit
            Maximum results to return

        Returns
        -------
        List of matching events sorted by relevance
        """
        if not query or not query.strip():
            return []

        search_term = query.strip()
        start_dt = _coerce_optional_dt(start)
        end_dt = _coerce_optional_dt(end)
        cat_uuid = _coerce_uuid(category_id)

        with self._session_scope() as session:
            engine = session.get_bind()

            if engine.dialect.name == "postgresql":
                # Use full-text search with ranking
                ts_query = func.plainto_tsquery("english", search_term)
                stmt = (
                    session.query(CalendarEventModel)
                    .options(joinedload(CalendarEventModel.category))
                    .filter(CalendarEventModel.search_vector.op("@@")(ts_query))
                    .order_by(
                        func.ts_rank(CalendarEventModel.search_vector, ts_query).desc()
                    )
                )
            else:
                # Fallback to LIKE for non-PostgreSQL
                pattern = f"%{search_term}%"
                stmt = (
                    session.query(CalendarEventModel)
                    .options(joinedload(CalendarEventModel.category))
                    .filter(
                        or_(
                            CalendarEventModel.title.ilike(pattern),
                            CalendarEventModel.description.ilike(pattern),
                            CalendarEventModel.location.ilike(pattern),
                        )
                    )
                    .order_by(CalendarEventModel.start_time)
                )

            # Apply additional filters
            if start_dt is not None:
                stmt = stmt.filter(CalendarEventModel.end_time >= start_dt)
            if end_dt is not None:
                stmt = stmt.filter(CalendarEventModel.start_time <= end_dt)
            if cat_uuid is not None:
                stmt = stmt.filter(CalendarEventModel.category_id == cat_uuid)

            stmt = stmt.limit(limit)
            events = stmt.all()
            return [_serialize_event(event) for event in events]

    # ========================================================================
    # Import mapping operations
    # ========================================================================

    def list_import_mappings(
        self, *, source_type: Optional[str] = None, source_account: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List import mappings with optional filtering."""
        with self._session_scope() as session:
            query = session.query(CalendarImportMappingModel).options(
                joinedload(CalendarImportMappingModel.target_category)
            )

            if source_type:
                query = query.filter(CalendarImportMappingModel.source_type == source_type)
            if source_account:
                query = query.filter(
                    CalendarImportMappingModel.source_account == source_account
                )

            mappings = query.all()
            return [_serialize_import_mapping(m) for m in mappings]

    def get_import_mapping(
        self, source_type: str, source_account: str, source_calendar: str
    ) -> Optional[Dict[str, Any]]:
        """Get the target category for an external calendar."""
        with self._session_scope() as session:
            mapping = (
                session.query(CalendarImportMappingModel)
                .filter(
                    CalendarImportMappingModel.source_type == source_type,
                    CalendarImportMappingModel.source_account == source_account,
                    CalendarImportMappingModel.source_calendar == source_calendar,
                )
                .first()
            )
            if mapping is None:
                return None
            return _serialize_import_mapping(mapping)

    def create_import_mapping(
        self,
        *,
        source_type: str,
        source_account: str,
        source_calendar: str,
        target_category_id: Any,
    ) -> Dict[str, Any]:
        """Create or update an import mapping."""
        cat_uuid = _coerce_uuid(target_category_id)
        if cat_uuid is None:
            raise ValueError("Invalid target category identifier")

        with self._session_scope() as session:
            # Check if mapping exists
            existing = (
                session.query(CalendarImportMappingModel)
                .filter(
                    CalendarImportMappingModel.source_type == source_type,
                    CalendarImportMappingModel.source_account == source_account,
                    CalendarImportMappingModel.source_calendar == source_calendar,
                )
                .first()
            )

            if existing:
                existing.target_category_id = cat_uuid
                session.flush()
                session.refresh(existing)
                return _serialize_import_mapping(existing)

            mapping = CalendarImportMappingModel(
                source_type=source_type,
                source_account=source_account,
                source_calendar=source_calendar,
                target_category_id=cat_uuid,
            )
            session.add(mapping)
            session.flush()
            session.refresh(mapping)
            return _serialize_import_mapping(mapping)

    def delete_import_mapping(
        self, source_type: str, source_account: str, source_calendar: str
    ) -> None:
        """Delete an import mapping."""
        with self._session_scope() as session:
            mapping = (
                session.query(CalendarImportMappingModel)
                .filter(
                    CalendarImportMappingModel.source_type == source_type,
                    CalendarImportMappingModel.source_account == source_account,
                    CalendarImportMappingModel.source_calendar == source_calendar,
                )
                .first()
            )
            if mapping:
                session.delete(mapping)

    # ========================================================================
    # Sync state operations
    # ========================================================================

    def get_sync_state(
        self,
        source_type: str,
        source_account: str,
        source_calendar: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get sync state for an external calendar source."""
        with self._session_scope() as session:
            query = session.query(CalendarSyncStateModel).filter(
                CalendarSyncStateModel.source_type == source_type,
                CalendarSyncStateModel.source_account == source_account,
            )
            if source_calendar is not None:
                query = query.filter(
                    CalendarSyncStateModel.source_calendar == source_calendar
                )
            else:
                query = query.filter(CalendarSyncStateModel.source_calendar == None)

            state = query.first()
            if state is None:
                return None
            return _serialize_sync_state(state)

    def update_sync_state(
        self,
        *,
        source_type: str,
        source_account: str,
        source_calendar: Optional[str] = None,
        sync_token: Optional[str] = None,
        last_sync_status: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update or create sync state for an external calendar source."""
        with self._session_scope() as session:
            query = session.query(CalendarSyncStateModel).filter(
                CalendarSyncStateModel.source_type == source_type,
                CalendarSyncStateModel.source_account == source_account,
            )
            if source_calendar is not None:
                query = query.filter(
                    CalendarSyncStateModel.source_calendar == source_calendar
                )
            else:
                query = query.filter(CalendarSyncStateModel.source_calendar == None)

            state = query.first()

            if state is None:
                state = CalendarSyncStateModel(
                    source_type=source_type,
                    source_account=source_account,
                    source_calendar=source_calendar,
                )
                session.add(state)

            if sync_token is not None:
                state.sync_token = sync_token
            state.last_sync_at = datetime.now(timezone.utc)
            if last_sync_status is not None:
                state.last_sync_status = last_sync_status
            if error_message is not None:
                state.error_message = error_message
            elif last_sync_status == "success":
                state.error_message = None

            session.flush()
            session.refresh(state)
            return _serialize_sync_state(state)

    # ========================================================================
    # Job/Task Link operations
    # ========================================================================

    def create_job_link(
        self,
        *,
        event_id: Any,
        job_id: Any,
        link_type: LinkType = LinkType.AUTO_CREATED,
        sync_behavior: SyncBehavior = SyncBehavior.FROM_SOURCE,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a link between a calendar event and a job.

        Parameters
        ----------
        event_id
            The calendar event ID
        job_id
            The job ID
        link_type
            Type of link relationship
        sync_behavior
            How changes should propagate
        notes
            Optional description of the link
        metadata
            Additional link metadata
        created_by
            User ID or 'system' who created the link

        Returns
        -------
        Serialized link dictionary
        """
        event_uuid = _coerce_uuid(event_id)
        job_uuid = _coerce_uuid(job_id)

        if event_uuid is None:
            raise EventNotFoundError("Invalid event identifier")

        with self._session_scope() as session:
            # Verify event exists
            event = session.get(CalendarEventModel, event_uuid)
            if event is None:
                raise EventNotFoundError(f"Event {event_id} not found")

            link = JobEventLink(
                event_id=event_uuid,
                job_id=job_uuid,
                link_type=link_type,
                sync_behavior=sync_behavior,
                notes=notes,
                meta=metadata or {},
                created_by=created_by,
            )
            session.add(link)
            session.flush()
            session.refresh(link)
            return _serialize_job_link(link)

    def create_task_link(
        self,
        *,
        event_id: Any,
        task_id: Any,
        link_type: LinkType = LinkType.AUTO_CREATED,
        sync_behavior: SyncBehavior = SyncBehavior.FROM_SOURCE,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a link between a calendar event and a task.

        Parameters
        ----------
        event_id
            The calendar event ID
        task_id
            The task ID
        link_type
            Type of link relationship
        sync_behavior
            How changes should propagate
        notes
            Optional description of the link
        metadata
            Additional link metadata
        created_by
            User ID or 'system' who created the link

        Returns
        -------
        Serialized link dictionary
        """
        event_uuid = _coerce_uuid(event_id)
        task_uuid = _coerce_uuid(task_id)

        if event_uuid is None:
            raise EventNotFoundError("Invalid event identifier")

        with self._session_scope() as session:
            # Verify event exists
            event = session.get(CalendarEventModel, event_uuid)
            if event is None:
                raise EventNotFoundError(f"Event {event_id} not found")

            link = TaskEventLink(
                event_id=event_uuid,
                task_id=task_uuid,
                link_type=link_type,
                sync_behavior=sync_behavior,
                notes=notes,
                meta=metadata or {},
                created_by=created_by,
            )
            session.add(link)
            session.flush()
            session.refresh(link)
            return _serialize_task_link(link)

    def get_job_links_for_event(self, event_id: Any) -> List[Dict[str, Any]]:
        """Get all job links for a calendar event."""
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            return []

        with self._session_scope() as session:
            links = (
                session.query(JobEventLink)
                .filter(JobEventLink.event_id == event_uuid)
                .all()
            )
            return [_serialize_job_link(link) for link in links]

    def get_task_links_for_event(self, event_id: Any) -> List[Dict[str, Any]]:
        """Get all task links for a calendar event."""
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            return []

        with self._session_scope() as session:
            links = (
                session.query(TaskEventLink)
                .filter(TaskEventLink.event_id == event_uuid)
                .all()
            )
            return [_serialize_task_link(link) for link in links]

    def get_events_for_job(self, job_id: Any) -> List[Dict[str, Any]]:
        """Get all calendar events linked to a job."""
        job_uuid = _coerce_uuid(job_id)
        if job_uuid is None:
            return []

        with self._session_scope() as session:
            links = (
                session.query(JobEventLink)
                .options(joinedload(JobEventLink.event))
                .filter(JobEventLink.job_id == job_uuid)
                .all()
            )
            return [_serialize_event(link.event) for link in links if link.event]

    def get_events_for_task(self, task_id: Any) -> List[Dict[str, Any]]:
        """Get all calendar events linked to a task."""
        task_uuid = _coerce_uuid(task_id)
        if task_uuid is None:
            return []

        with self._session_scope() as session:
            links = (
                session.query(TaskEventLink)
                .options(joinedload(TaskEventLink.event))
                .filter(TaskEventLink.task_id == task_uuid)
                .all()
            )
            return [_serialize_event(link.event) for link in links if link.event]

    def delete_job_link(
        self,
        event_id: Any,
        job_id: Any,
    ) -> bool:
        """Delete a job link by event and job ID.

        Returns
        -------
        True if the link was deleted, False if it didn't exist
        """
        event_uuid = _coerce_uuid(event_id)
        job_uuid = _coerce_uuid(job_id)

        if event_uuid is None or job_uuid is None:
            return False

        with self._session_scope() as session:
            link = (
                session.query(JobEventLink)
                .filter(
                    JobEventLink.event_id == event_uuid,
                    JobEventLink.job_id == job_uuid,
                )
                .first()
            )
            if link is None:
                return False

            session.delete(link)
            return True

    def delete_task_link(
        self,
        event_id: Any,
        task_id: Any,
    ) -> bool:
        """Delete a task link by event and task ID.

        Returns
        -------
        True if the link was deleted, False if it didn't exist
        """
        event_uuid = _coerce_uuid(event_id)
        task_uuid = _coerce_uuid(task_id)

        if event_uuid is None or task_uuid is None:
            return False

        with self._session_scope() as session:
            link = (
                session.query(TaskEventLink)
                .filter(
                    TaskEventLink.event_id == event_uuid,
                    TaskEventLink.task_id == task_uuid,
                )
                .first()
            )
            if link is None:
                return False

            session.delete(link)
            return True

    def delete_all_job_links_for_event(self, event_id: Any) -> int:
        """Delete all job links for an event.

        Returns
        -------
        Number of links deleted
        """
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            return 0

        with self._session_scope() as session:
            count = (
                session.query(JobEventLink)
                .filter(JobEventLink.event_id == event_uuid)
                .delete()
            )
            return count

    def delete_all_task_links_for_event(self, event_id: Any) -> int:
        """Delete all task links for an event.

        Returns
        -------
        Number of links deleted
        """
        event_uuid = _coerce_uuid(event_id)
        if event_uuid is None:
            return 0

        with self._session_scope() as session:
            count = (
                session.query(TaskEventLink)
                .filter(TaskEventLink.event_id == event_uuid)
                .delete()
            )
            return count

    def get_links_with_sync_behavior(
        self,
        sync_behavior: SyncBehavior,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all links with a specific sync behavior.

        Returns
        -------
        Dictionary with 'job_links' and 'task_links' keys
        """
        with self._session_scope() as session:
            job_links = (
                session.query(JobEventLink)
                .filter(JobEventLink.sync_behavior == sync_behavior)
                .all()
            )
            task_links = (
                session.query(TaskEventLink)
                .filter(TaskEventLink.sync_behavior == sync_behavior)
                .all()
            )
            return {
                "job_links": [_serialize_job_link(link) for link in job_links],
                "task_links": [_serialize_task_link(link) for link in task_links],
            }


def _serialize_job_link(link: JobEventLink) -> Dict[str, Any]:
    """Serialize a JobEventLink to a dictionary."""
    return {
        "id": str(link.id),
        "event_id": str(link.event_id),
        "job_id": str(link.job_id),
        "link_type": link.link_type.value,
        "sync_behavior": link.sync_behavior.value,
        "notes": link.notes,
        "metadata": link.meta,
        "created_at": _dt_to_iso(link.created_at),
        "created_by": link.created_by,
    }


def _serialize_task_link(link: TaskEventLink) -> Dict[str, Any]:
    """Serialize a TaskEventLink to a dictionary."""
    return {
        "id": str(link.id),
        "event_id": str(link.event_id),
        "task_id": str(link.task_id),
        "link_type": link.link_type.value,
        "sync_behavior": link.sync_behavior.value,
        "notes": link.notes,
        "metadata": link.meta,
        "created_at": _dt_to_iso(link.created_at),
        "created_by": link.created_by,
    }


__all__ = [
    # Exceptions
    "CalendarStoreError",
    "CategoryNotFoundError",
    "EventNotFoundError",
    "CategorySlugExistsError",
    "ReadOnlyCategoryError",
    # Repository
    "CalendarStoreRepository",
    # Link types (re-exported for convenience)
    "LinkType",
    "SyncBehavior",
]
