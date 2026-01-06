"""Tests for the ATLAS Master Calendar store.

This module tests the CalendarStoreRepository CRUD operations
for categories, events, import mappings, and sync state.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.calendar_store import (
    CalendarStoreRepository,
    CalendarCategoryModel,
    CalendarEventModel,
    create_schema,
    EventStatus,
    BusyStatus,
    SyncDirection,
    SyncStatus,
    CategoryNotFoundError,
    EventNotFoundError,
    CategorySlugExistsError,
    ReadOnlyCategoryError,
    BUILTIN_CATEGORIES,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    create_schema(engine, seed_categories=True)
    return engine


@pytest.fixture
def session_factory(engine):
    """Create a session factory bound to the test engine."""
    return sessionmaker(bind=engine)


@pytest.fixture
def repo(session_factory):
    """Create a repository instance for testing."""
    return CalendarStoreRepository(session_factory)


class TestCategoryOperations:
    """Tests for category CRUD operations."""

    def test_list_categories_returns_builtin(self, repo):
        """Built-in categories should be seeded on schema creation."""
        categories = repo.list_categories()
        assert len(categories) >= len(BUILTIN_CATEGORIES)
        
        names = {c["name"] for c in categories}
        for builtin in BUILTIN_CATEGORIES:
            assert builtin["name"] in names

    def test_list_categories_filters_hidden(self, repo):
        """Hidden categories should be excluded when include_hidden=False."""
        # Hide a category
        categories = repo.list_categories()
        work_cat = next(c for c in categories if c["slug"] == "work")
        repo.toggle_category_visibility(work_cat["id"], visible=False)
        
        # Should not appear when filtering hidden
        visible_categories = repo.list_categories(include_hidden=False)
        visible_slugs = {c["slug"] for c in visible_categories}
        assert "work" not in visible_slugs
        
        # Should appear when including hidden
        all_categories = repo.list_categories(include_hidden=True)
        all_slugs = {c["slug"] for c in all_categories}
        assert "work" in all_slugs

    def test_get_category_by_id(self, repo):
        """Should retrieve a category by its UUID."""
        categories = repo.list_categories()
        first_cat = categories[0]
        
        retrieved = repo.get_category(first_cat["id"])
        assert retrieved["id"] == first_cat["id"]
        assert retrieved["name"] == first_cat["name"]

    def test_get_category_not_found(self, repo):
        """Should raise CategoryNotFoundError for invalid ID."""
        with pytest.raises(CategoryNotFoundError):
            repo.get_category(str(uuid.uuid4()))

    def test_get_category_by_slug(self, repo):
        """Should retrieve a category by its slug."""
        personal = repo.get_category_by_slug("personal")
        assert personal["name"] == "Personal"
        assert personal["is_default"] is True

    def test_get_default_category(self, repo):
        """Should return the default category (Personal)."""
        default = repo.get_default_category()
        assert default is not None
        assert default["slug"] == "personal"
        assert default["is_default"] is True

    def test_create_custom_category(self, repo):
        """Should create a new custom category."""
        category = repo.create_category(
            name="Gaming",
            color="#FF5722",
            icon="🎮",
            description="Gaming events and tournaments",
        )
        
        assert category["name"] == "Gaming"
        assert category["slug"] == "gaming"
        assert category["color"] == "#FF5722"
        assert category["icon"] == "🎮"
        assert category["is_builtin"] is False
        assert category["is_readonly"] is False

    def test_create_category_auto_generates_slug(self, repo):
        """Should generate a URL-safe slug from the name."""
        category = repo.create_category(name="My Custom Events!")
        assert category["slug"] == "my-custom-events"

    def test_create_category_duplicate_slug_fails(self, repo):
        """Should raise error when creating category with existing slug."""
        repo.create_category(name="Test Category")
        
        with pytest.raises(CategorySlugExistsError):
            repo.create_category(name="Test Category")

    def test_update_category(self, repo):
        """Should update category properties."""
        category = repo.create_category(name="Original Name", color="#000000")
        
        updated = repo.update_category(
            category["id"],
            changes={"name": "New Name", "color": "#FFFFFF"},
        )
        
        assert updated["name"] == "New Name"
        assert updated["color"] == "#FFFFFF"

    def test_update_readonly_category_fails(self, repo):
        """Should not allow modifying read-only categories (except visibility)."""
        holidays = repo.get_category_by_slug("holidays")
        
        with pytest.raises(ReadOnlyCategoryError):
            repo.update_category(holidays["id"], changes={"name": "My Holidays"})

    def test_update_readonly_category_visibility_allowed(self, repo):
        """Should allow changing visibility on read-only categories."""
        holidays = repo.get_category_by_slug("holidays")
        
        # This should succeed
        updated = repo.update_category(
            holidays["id"],
            changes={"is_visible": False},
        )
        assert updated["is_visible"] is False

    def test_delete_custom_category(self, repo):
        """Should delete custom categories."""
        category = repo.create_category(name="Temporary")
        repo.delete_category(category["id"])
        
        with pytest.raises(CategoryNotFoundError):
            repo.get_category(category["id"])

    def test_delete_builtin_category_fails(self, repo):
        """Should not allow deleting built-in categories."""
        work = repo.get_category_by_slug("work")
        
        with pytest.raises(ReadOnlyCategoryError):
            repo.delete_category(work["id"])

    def test_set_default_category(self, repo):
        """Should change the default category."""
        work = repo.get_category_by_slug("work")
        
        updated = repo.set_default_category(work["id"])
        assert updated["is_default"] is True
        
        # Previous default should no longer be default
        personal = repo.get_category_by_slug("personal")
        assert personal["is_default"] is False

    def test_toggle_category_visibility(self, repo):
        """Should toggle category visibility."""
        work = repo.get_category_by_slug("work")
        assert work["is_visible"] is True
        
        toggled = repo.toggle_category_visibility(work["id"])
        assert toggled["is_visible"] is False
        
        toggled_again = repo.toggle_category_visibility(work["id"])
        assert toggled_again["is_visible"] is True


class TestEventOperations:
    """Tests for event CRUD operations."""

    @pytest.fixture
    def sample_event_data(self):
        """Sample event data for testing."""
        now = datetime.now(timezone.utc)
        return {
            "title": "Team Meeting",
            "start_time": now,
            "end_time": now + timedelta(hours=1),
            "description": "Weekly team sync",
            "location": "Conference Room A",
        }

    def test_create_event(self, repo, sample_event_data):
        """Should create a new event."""
        event = repo.create_event(**sample_event_data)
        
        assert event["title"] == "Team Meeting"
        assert event["description"] == "Weekly team sync"
        assert event["location"] == "Conference Room A"
        assert event["status"] == "confirmed"

    def test_create_event_uses_default_category(self, repo, sample_event_data):
        """Events without category should use the default category."""
        event = repo.create_event(**sample_event_data)
        
        default_cat = repo.get_default_category()
        assert event["category_id"] == default_cat["id"]

    def test_create_event_with_category(self, repo, sample_event_data):
        """Should assign event to specified category."""
        work = repo.get_category_by_slug("work")
        
        event = repo.create_event(
            **sample_event_data,
            category_id=work["id"],
        )
        
        assert event["category_id"] == work["id"]

    def test_get_event(self, repo, sample_event_data):
        """Should retrieve an event by ID."""
        created = repo.create_event(**sample_event_data)
        
        retrieved = repo.get_event(created["id"])
        assert retrieved["id"] == created["id"]
        assert retrieved["title"] == created["title"]

    def test_get_event_not_found(self, repo):
        """Should raise EventNotFoundError for invalid ID."""
        with pytest.raises(EventNotFoundError):
            repo.get_event(str(uuid.uuid4()))

    def test_list_events_by_date_range(self, repo):
        """Should filter events by date range."""
        now = datetime.now(timezone.utc)
        
        # Create events at different times
        repo.create_event(
            title="Past Event",
            start_time=now - timedelta(days=7),
            end_time=now - timedelta(days=7) + timedelta(hours=1),
        )
        repo.create_event(
            title="Current Event",
            start_time=now,
            end_time=now + timedelta(hours=1),
        )
        repo.create_event(
            title="Future Event",
            start_time=now + timedelta(days=7),
            end_time=now + timedelta(days=7) + timedelta(hours=1),
        )
        
        # Query only current and future
        events = repo.list_events(
            start=now - timedelta(hours=1),
            end=now + timedelta(days=1),
        )
        titles = {e["title"] for e in events}
        assert "Current Event" in titles
        assert "Past Event" not in titles
        assert "Future Event" not in titles

    def test_list_events_by_category(self, repo, sample_event_data):
        """Should filter events by category."""
        work = repo.get_category_by_slug("work")
        personal = repo.get_category_by_slug("personal")
        
        repo.create_event(**sample_event_data, category_id=work["id"])
        repo.create_event(
            title="Personal Event",
            start_time=sample_event_data["start_time"],
            end_time=sample_event_data["end_time"],
            category_id=personal["id"],
        )
        
        work_events = repo.list_events(category_id=work["id"])
        assert len(work_events) == 1
        assert work_events[0]["title"] == "Team Meeting"

    def test_update_event(self, repo, sample_event_data):
        """Should update event properties."""
        event = repo.create_event(**sample_event_data)
        
        updated = repo.update_event(
            event["id"],
            changes={
                "title": "Updated Meeting",
                "status": "tentative",
            },
        )
        
        assert updated["title"] == "Updated Meeting"
        assert updated["status"] == "tentative"

    def test_delete_event(self, repo, sample_event_data):
        """Should delete an event."""
        event = repo.create_event(**sample_event_data)
        repo.delete_event(event["id"])
        
        with pytest.raises(EventNotFoundError):
            repo.get_event(event["id"])

    def test_create_all_day_event(self, repo):
        """Should support all-day events."""
        now = datetime.now(timezone.utc)
        
        event = repo.create_event(
            title="Conference",
            start_time=now.replace(hour=0, minute=0, second=0),
            end_time=now.replace(hour=23, minute=59, second=59),
            is_all_day=True,
        )
        
        assert event["is_all_day"] is True

    def test_event_with_attendees(self, repo, sample_event_data):
        """Should store attendee information."""
        event = repo.create_event(
            **sample_event_data,
            attendees=[
                {"email": "alice@example.com", "name": "Alice", "status": "accepted"},
                {"email": "bob@example.com", "name": "Bob", "status": "tentative"},
            ],
        )
        
        assert len(event["attendees"]) == 2
        emails = {a["email"] for a in event["attendees"]}
        assert "alice@example.com" in emails
        assert "bob@example.com" in emails

    def test_event_with_reminders(self, repo, sample_event_data):
        """Should store reminder configuration."""
        event = repo.create_event(
            **sample_event_data,
            reminders=[
                {"minutes_before": 15, "method": "notification"},
                {"minutes_before": 60, "method": "email"},
            ],
        )
        
        assert len(event["reminders"]) == 2

    def test_event_with_tags(self, repo, sample_event_data):
        """Should store event tags."""
        event = repo.create_event(
            **sample_event_data,
            tags=["important", "recurring", "team"],
        )
        
        assert "important" in event["tags"]
        assert "team" in event["tags"]


class TestImportMappingOperations:
    """Tests for import mapping CRUD operations."""

    def test_create_import_mapping(self, repo):
        """Should create an import mapping."""
        work = repo.get_category_by_slug("work")
        
        mapping = repo.create_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
            target_category_id=work["id"],
        )
        
        assert mapping["source_type"] == "google"
        assert mapping["source_account"] == "user@gmail.com"
        assert mapping["target_category_id"] == work["id"]

    def test_get_import_mapping(self, repo):
        """Should retrieve an import mapping."""
        work = repo.get_category_by_slug("work")
        
        repo.create_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
            target_category_id=work["id"],
        )
        
        mapping = repo.get_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
        )
        
        assert mapping is not None
        assert mapping["target_category_id"] == work["id"]

    def test_list_import_mappings(self, repo):
        """Should list all import mappings."""
        work = repo.get_category_by_slug("work")
        personal = repo.get_category_by_slug("personal")
        
        repo.create_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
            target_category_id=work["id"],
        )
        repo.create_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Personal",
            target_category_id=personal["id"],
        )
        
        mappings = repo.list_import_mappings()
        assert len(mappings) == 2

    def test_delete_import_mapping(self, repo):
        """Should delete an import mapping."""
        work = repo.get_category_by_slug("work")
        
        repo.create_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
            target_category_id=work["id"],
        )
        
        repo.delete_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
        )
        
        mapping = repo.get_import_mapping(
            source_type="google",
            source_account="user@gmail.com",
            source_calendar="Work",
        )
        assert mapping is None


class TestSyncStateOperations:
    """Tests for sync state tracking."""

    def test_update_sync_state_creates_new(self, repo):
        """Should create sync state if it doesn't exist."""
        state = repo.update_sync_state(
            source_type="google",
            source_account="user@gmail.com",
            sync_token="token123",
            last_sync_status="success",
        )
        
        assert state["source_type"] == "google"
        assert state["sync_token"] == "token123"
        assert state["last_sync_status"] == "success"
        assert state["last_sync_at"] is not None

    def test_update_sync_state_updates_existing(self, repo):
        """Should update existing sync state."""
        repo.update_sync_state(
            source_type="google",
            source_account="user@gmail.com",
            sync_token="token1",
        )
        
        updated = repo.update_sync_state(
            source_type="google",
            source_account="user@gmail.com",
            sync_token="token2",
            last_sync_status="success",
        )
        
        assert updated["sync_token"] == "token2"

    def test_get_sync_state(self, repo):
        """Should retrieve sync state."""
        repo.update_sync_state(
            source_type="google",
            source_account="user@gmail.com",
            sync_token="token123",
        )
        
        state = repo.get_sync_state(
            source_type="google",
            source_account="user@gmail.com",
        )
        
        assert state is not None
        assert state["sync_token"] == "token123"

    def test_get_sync_state_not_found(self, repo):
        """Should return None for non-existent sync state."""
        state = repo.get_sync_state(
            source_type="nonexistent",
            source_account="nobody@example.com",
        )
        assert state is None


class TestSearchOperations:
    """Tests for event search functionality."""

    def test_search_events_by_title(self, repo):
        """Should find events by title."""
        now = datetime.now(timezone.utc)
        
        repo.create_event(
            title="Python Conference",
            start_time=now,
            end_time=now + timedelta(hours=8),
        )
        repo.create_event(
            title="Team Meeting",
            start_time=now,
            end_time=now + timedelta(hours=1),
        )
        
        results = repo.search_events("conference")
        assert len(results) == 1
        assert results[0]["title"] == "Python Conference"

    def test_search_events_by_description(self, repo):
        """Should find events by description."""
        now = datetime.now(timezone.utc)
        
        repo.create_event(
            title="Event A",
            description="Discuss Python frameworks",
            start_time=now,
            end_time=now + timedelta(hours=1),
        )
        repo.create_event(
            title="Event B",
            description="Review Java code",
            start_time=now,
            end_time=now + timedelta(hours=1),
        )
        
        results = repo.search_events("python")
        assert len(results) == 1
        assert results[0]["title"] == "Event A"

    def test_search_events_by_location(self, repo):
        """Should find events by location."""
        now = datetime.now(timezone.utc)
        
        repo.create_event(
            title="Meeting",
            location="Conference Room Alpha",
            start_time=now,
            end_time=now + timedelta(hours=1),
        )
        
        results = repo.search_events("alpha")
        assert len(results) == 1

    def test_search_events_empty_query(self, repo):
        """Should return empty results for empty query."""
        results = repo.search_events("")
        assert results == []

        results = repo.search_events("   ")
        assert results == []
