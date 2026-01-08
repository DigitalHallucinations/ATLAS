"""GTK Calendar View Stack controller.

Provides view switching between month, week, day, and agenda views.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Set

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk # type: ignore[import-untyped]

from .calendar_month_view import CalendarMonthView
from .calendar_week_view import CalendarWeekView
from .calendar_day_view import CalendarDayView
from .calendar_agenda_view import CalendarAgendaView

logger = logging.getLogger(__name__)


class CalendarViewStack(Gtk.Box):
    """Calendar view stack with view switching.

    Provides a unified interface to switch between different calendar
    views (month, week, day, agenda) with consistent navigation.
    """

    def __init__(
        self,
        atlas: Any = None,
        on_event_clicked: Optional[Callable[[str], None]] = None,
        on_create_event: Optional[Callable[[date, Optional[int]], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._atlas = atlas
        self._on_event_clicked = on_event_clicked
        self._on_create_event = on_create_event

        self._current_view = "month"
        self._month_view: Optional[CalendarMonthView] = None
        self._week_view: Optional[CalendarWeekView] = None
        self._day_view: Optional[CalendarDayView] = None
        self._agenda_view: Optional[CalendarAgendaView] = None

        # Filter state
        self._filter_categories: Set[str] = set()  # Empty = all categories
        self._filter_linked_jobs: bool = False
        self._filter_linked_tasks: bool = False
        self._categories: List[Dict[str, Any]] = []
        self._filter_popover: Optional[Gtk.Popover] = None

        # MessageBus subscriptions
        self._bus_subscriptions: List[Any] = []

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-view-stack")

        self._build_ui()
        self._subscribe_to_bus()

    def _build_ui(self) -> None:
        """Build the view stack UI."""
        # View selector toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        toolbar.set_margin_start(8)
        toolbar.set_margin_end(8)
        toolbar.set_margin_top(8)
        toolbar.set_margin_bottom(4)

        # View type toggle buttons (segmented control style)
        view_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        view_box.add_css_class("linked")

        self._month_btn = Gtk.ToggleButton(label="Month")
        self._month_btn.set_active(True)
        self._month_btn.connect("toggled", self._on_view_toggled, "month")
        view_box.append(self._month_btn)

        self._week_btn = Gtk.ToggleButton(label="Week")
        self._week_btn.connect("toggled", self._on_view_toggled, "week")
        view_box.append(self._week_btn)

        self._day_btn = Gtk.ToggleButton(label="Day")
        self._day_btn.connect("toggled", self._on_view_toggled, "day")
        view_box.append(self._day_btn)

        self._agenda_btn = Gtk.ToggleButton(label="Agenda")
        self._agenda_btn.connect("toggled", self._on_view_toggled, "agenda")
        view_box.append(self._agenda_btn)

        toolbar.append(view_box)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        toolbar.append(spacer)

        # Filter button with popover
        filter_btn = Gtk.MenuButton()
        filter_btn.set_icon_name("view-filter-symbolic")
        filter_btn.set_tooltip_text("Filter events")
        filter_btn.add_css_class("flat")
        self._filter_btn = filter_btn
        
        # Build filter popover
        self._build_filter_popover()
        filter_btn.set_popover(self._filter_popover)
        toolbar.append(filter_btn)

        # Add event button
        add_btn = Gtk.Button()
        add_btn.set_icon_name("list-add-symbolic")
        add_btn.set_tooltip_text("Add Event")
        add_btn.add_css_class("suggested-action")
        add_btn.connect("clicked", self._on_add_event_clicked)
        toolbar.append(add_btn)

        self.append(toolbar)

        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self.append(sep)

        # View stack
        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(150)
        self._stack.set_hexpand(True)
        self._stack.set_vexpand(True)

        # Create views
        self._month_view = CalendarMonthView(
            atlas=self._atlas,
            on_date_selected=self._on_date_selected,
            on_date_activated=self._on_date_activated,
            on_event_clicked=self._on_event_clicked,
        )
        self._stack.add_named(self._month_view, "month")

        self._week_view = CalendarWeekView(
            atlas=self._atlas,
            on_hour_clicked=self._on_hour_clicked,
            on_event_clicked=self._on_event_clicked,
        )
        self._stack.add_named(self._week_view, "week")

        self._day_view = CalendarDayView(
            atlas=self._atlas,
            on_hour_clicked=self._on_hour_clicked,
            on_event_clicked=self._on_event_clicked,
        )
        self._stack.add_named(self._day_view, "day")

        self._agenda_view = CalendarAgendaView(
            atlas=self._atlas,
            on_event_clicked=self._on_event_clicked,
        )
        self._stack.add_named(self._agenda_view, "agenda")

        self.append(self._stack)

    def _on_view_toggled(self, button: Gtk.ToggleButton, view_name: str) -> None:
        """Handle view toggle button."""
        if not button.get_active():
            # Prevent untoggling - ensure one is always active
            if self._current_view == view_name:
                button.set_active(True)
            return

        # Untoggle other buttons
        buttons = {
            "month": self._month_btn,
            "week": self._week_btn,
            "day": self._day_btn,
            "agenda": self._agenda_btn,
        }

        for name, btn in buttons.items():
            if name != view_name:
                btn.set_active(False)

        self._current_view = view_name
        self._stack.set_visible_child_name(view_name)

    def _on_date_selected(self, selected_date: date) -> None:
        """Handle date selection in month view."""
        logger.debug("Date selected: %s", selected_date)

    def _on_date_activated(self, activated_date: date) -> None:
        """Handle date double-click (switch to day view)."""
        if self._day_view:
            self._day_view.go_to_date(activated_date)
        self.set_view("day")

    def _on_hour_clicked(self, click_date: date, hour: int) -> None:
        """Handle hour slot click (create event)."""
        if self._on_create_event:
            self._on_create_event(click_date, hour)

    def _on_add_event_clicked(self, button: Gtk.Button) -> None:
        """Handle add event button click."""
        if self._on_create_event:
            # Use currently selected/viewed date
            current_date = date.today()

            if self._current_view == "month" and self._month_view:
                current_date = self._month_view.selected_date or date.today()
            elif self._current_view == "week" and self._week_view:
                current_date = self._week_view.week_start
            elif self._current_view == "day" and self._day_view:
                current_date = self._day_view.view_date

            self._on_create_event(current_date, None)

    def set_view(self, view_name: str) -> None:
        """Set the active view programmatically."""
        buttons = {
            "month": self._month_btn,
            "week": self._week_btn,
            "day": self._day_btn,
            "agenda": self._agenda_btn,
        }

        if view_name in buttons:
            buttons[view_name].set_active(True)

    def go_to_date(self, target_date: date) -> None:
        """Navigate all views to target date."""
        if self._month_view:
            self._month_view.go_to_date(target_date)
        if self._week_view:
            self._week_view.go_to_date(target_date)
        if self._day_view:
            self._day_view.go_to_date(target_date)

    def go_to_today(self) -> None:
        """Navigate all views to today."""
        if self._month_view:
            self._month_view.go_to_today()
        if self._week_view:
            self._week_view.go_to_today()
        if self._day_view:
            self._day_view.go_to_today()
        if self._agenda_view:
            self._agenda_view.refresh()

    def refresh(self) -> None:
        """Refresh all views."""
        if self._month_view:
            self._month_view.refresh()
        if self._week_view:
            self._week_view.refresh()
        if self._day_view:
            self._day_view.refresh()
        if self._agenda_view:
            self._agenda_view.refresh()

    @property
    def current_view(self) -> str:
        """Get current view name."""
        return self._current_view

    # ========================================================================
    # Filter functionality
    # ========================================================================

    def _build_filter_popover(self) -> None:
        """Build the filter popover content."""
        popover = Gtk.Popover()
        popover.set_autohide(True)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_size_request(250, -1)

        # Header
        header = Gtk.Label(label="Filter Events")
        header.add_css_class("heading")
        header.set_xalign(0)
        content.append(header)

        # Linked entities section
        linked_header = Gtk.Label(label="Show only events linked to:")
        linked_header.add_css_class("dim-label")
        linked_header.set_xalign(0)
        content.append(linked_header)

        linked_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        linked_box.set_margin_start(8)

        # Jobs filter
        self._jobs_check = Gtk.CheckButton(label="Jobs")
        self._jobs_check.connect("toggled", self._on_linked_filter_toggled, "jobs")
        linked_box.append(self._jobs_check)

        # Tasks filter
        self._tasks_check = Gtk.CheckButton(label="Tasks")
        self._tasks_check.connect("toggled", self._on_linked_filter_toggled, "tasks")
        linked_box.append(self._tasks_check)

        content.append(linked_box)

        # Separator
        content.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Categories section
        categories_header = Gtk.Label(label="Categories:")
        categories_header.add_css_class("dim-label")
        categories_header.set_xalign(0)
        content.append(categories_header)

        self._categories_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._categories_box.set_margin_start(8)
        content.append(self._categories_box)

        # Will be populated dynamically
        loading_label = Gtk.Label(label="Loading categories...")
        loading_label.add_css_class("dim-label")
        self._categories_box.append(loading_label)

        # Clear filters button
        clear_btn = Gtk.Button(label="Clear All Filters")
        clear_btn.connect("clicked", self._on_clear_filters_clicked)
        clear_btn.set_margin_top(8)
        content.append(clear_btn)

        popover.set_child(content)

        # Load categories when popover is shown
        popover.connect("show", self._on_filter_popover_show)
        
        self._filter_popover = popover

    def _on_filter_popover_show(self, popover: Gtk.Popover) -> None:
        """Load categories when filter popover is shown."""
        self._load_categories_for_filter()

    def _load_categories_for_filter(self) -> None:
        """Load categories from the calendar service."""
        async def fetch_categories():
            try:
                if not self._atlas:
                    return []

                service = self._atlas.calendar_service
                actor = self._atlas.get_current_actor()
                result = await service.list_categories(actor)

                if result.is_success:
                    return [
                        {"id": str(c.id), "name": c.name, "color": c.color, "icon": c.icon}
                        for c in result.data
                    ]
            except Exception as exc:
                logger.warning(f"Failed to load categories for filter: {exc}")
            return []

        async def update_ui():
            self._categories = await fetch_categories()
            GLib.idle_add(self._populate_category_checkboxes)

        asyncio.create_task(update_ui())

    def _populate_category_checkboxes(self) -> bool:
        """Populate category checkboxes in filter popover."""
        # Clear existing children
        while True:
            child = self._categories_box.get_first_child()
            if child is None:
                break
            self._categories_box.remove(child)

        if not self._categories:
            no_cats_label = Gtk.Label(label="No categories available")
            no_cats_label.add_css_class("dim-label")
            self._categories_box.append(no_cats_label)
            return True

        # All categories checkbox
        all_check = Gtk.CheckButton(label="All categories")
        all_check.set_active(len(self._filter_categories) == 0)
        all_check.connect("toggled", self._on_all_categories_toggled)
        self._all_categories_check = all_check
        self._categories_box.append(all_check)

        # Individual category checkboxes
        self._category_checks: Dict[str, Gtk.CheckButton] = {}
        for cat in self._categories:
            icon = cat.get("icon", "")
            name = cat.get("name", "Unknown")
            cat_id = cat.get("id", "")
            label = f"{icon} {name}" if icon else name

            check = Gtk.CheckButton(label=label)
            check.set_active(
                len(self._filter_categories) == 0 or 
                cat_id in self._filter_categories
            )
            check.set_margin_start(16)
            check.connect("toggled", self._on_category_toggled, cat_id)
            self._category_checks[cat_id] = check
            self._categories_box.append(check)

        return True

    def _on_all_categories_toggled(self, check: Gtk.CheckButton) -> None:
        """Handle 'All categories' checkbox toggle."""
        if check.get_active():
            self._filter_categories.clear()
            # Check all individual boxes
            for cat_check in self._category_checks.values():
                cat_check.set_active(True)
            self._apply_filters()

    def _on_category_toggled(self, check: Gtk.CheckButton, cat_id: str) -> None:
        """Handle individual category checkbox toggle."""
        if check.get_active():
            if cat_id in self._filter_categories:
                self._filter_categories.remove(cat_id)
        else:
            self._filter_categories.add(cat_id)
            # Uncheck "All categories" when filtering
            if hasattr(self, "_all_categories_check"):
                self._all_categories_check.set_active(False)
        
        self._apply_filters()

    def _on_linked_filter_toggled(self, check: Gtk.CheckButton, entity_type: str) -> None:
        """Handle linked entity filter toggle."""
        is_active = check.get_active()
        if entity_type == "jobs":
            self._filter_linked_jobs = is_active
        elif entity_type == "tasks":
            self._filter_linked_tasks = is_active
        
        self._apply_filters()

    def _on_clear_filters_clicked(self, button: Gtk.Button) -> None:
        """Clear all filters."""
        self._filter_categories.clear()
        self._filter_linked_jobs = False
        self._filter_linked_tasks = False

        # Reset UI
        if hasattr(self, "_jobs_check"):
            self._jobs_check.set_active(False)
        if hasattr(self, "_tasks_check"):
            self._tasks_check.set_active(False)
        if hasattr(self, "_all_categories_check"):
            self._all_categories_check.set_active(True)
        for cat_check in getattr(self, "_category_checks", {}).values():
            cat_check.set_active(True)

        self._apply_filters()
        if self._filter_popover:
            self._filter_popover.popdown()

    def _apply_filters(self) -> None:
        """Apply current filters to all views."""
        filter_config = {
            "category_ids": list(self._filter_categories) if self._filter_categories else None,
            "linked_jobs_only": self._filter_linked_jobs,
            "linked_tasks_only": self._filter_linked_tasks,
        }

        # Update filter indicator on button
        has_filters = (
            bool(self._filter_categories) or 
            self._filter_linked_jobs or 
            self._filter_linked_tasks
        )
        if has_filters:
            self._filter_btn.add_css_class("accent")
        else:
            self._filter_btn.remove_css_class("accent")

        # Apply to views
        for view in [self._month_view, self._week_view, self._day_view, self._agenda_view]:
            if view and hasattr(view, "set_filter"):
                view.set_filter(filter_config)

        # Refresh views
        self.refresh()

    @property
    def active_filters(self) -> Dict[str, Any]:
        """Get currently active filter configuration."""
        return {
            "category_ids": list(self._filter_categories) if self._filter_categories else None,
            "linked_jobs_only": self._filter_linked_jobs,
            "linked_tasks_only": self._filter_linked_tasks,
        }

    # ------------------------------------------------------------------
    # MessageBus integration
    # ------------------------------------------------------------------
    def _subscribe_to_bus(self) -> None:
        """Subscribe to job/task/calendar events for reactive updates."""
        if not self._atlas:
            return
        if self._bus_subscriptions:
            return

        # Calendar events for immediate refresh
        calendar_events = (
            "calendar.event_created",
            "calendar.event_updated",
            "calendar.event_deleted",
            "calendar.event_linked",
            "calendar.event_unlinked",
        )
        for event_name in calendar_events:
            try:
                handle = self._atlas.subscribe_event(event_name, self._handle_calendar_event)
                self._bus_subscriptions.append(handle)
            except Exception as exc:
                logger.debug("Unable to subscribe to %s events: %s", event_name, exc)

        # Job/task events for linked entity badge updates
        entity_events = (
            "job.status_changed",
            "job.updated",
            "task.status_changed",
            "task.updated",
        )
        for event_name in entity_events:
            try:
                handle = self._atlas.subscribe_event(event_name, self._handle_entity_event)
                self._bus_subscriptions.append(handle)
            except Exception as exc:
                logger.debug("Unable to subscribe to %s events: %s", event_name, exc)

    async def _handle_calendar_event(self, payload: Any, *_args: Any) -> None:
        """Handle calendar events - refresh views."""
        logger.debug("Received calendar event; scheduling view refresh")
        GLib.idle_add(self.refresh)

    async def _handle_entity_event(self, payload: Any, *_args: Any) -> None:
        """Handle job/task events - refresh views for badge updates."""
        logger.debug("Received job/task event; scheduling view refresh")
        GLib.idle_add(self.refresh)

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all MessageBus events."""
        if not self._atlas:
            return
        for subscription in list(self._bus_subscriptions):
            try:
                self._atlas.unsubscribe_event(subscription)
            except Exception as exc:
                logger.debug("Failed to unsubscribe: %s", exc)
            self._bus_subscriptions.remove(subscription)
