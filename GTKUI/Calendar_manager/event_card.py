"""Event card widget for ATLAS Calendar.

Displays calendar events as compact cards for use in calendar views.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Any, Callable, Dict, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk, GObject

logger = logging.getLogger(__name__)


class EventCard(Gtk.Box):
    """A compact card displaying a calendar event.

    Attributes:
        event_id: Unique identifier of the event
        event_data: Full event data dictionary
    """

    __gtype_name__ = "ATLASEventCard"

    # Signals
    __gsignals__ = {
        "clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "edit-requested": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "delete-requested": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(
        self,
        event_data: Dict[str, Any],
        compact: bool = False,
        show_time: bool = True,
        show_category: bool = True,
    ) -> None:
        """Initialize the event card.

        Args:
            event_data: Event data dictionary
            compact: If True, show minimal info
            show_time: Whether to show event time
            show_category: Whether to show category indicator
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)

        self.event_id = event_data.get("id", "")
        self.event_data = event_data
        self._compact = compact
        self._show_time = show_time
        self._show_category = show_category

        self._setup_styles()
        self._build_card()
        self._setup_interactions()

    def _setup_styles(self) -> None:
        """Set up CSS styling for the card."""
        self.add_css_class("event-card")
        self.set_margin_top(2)
        self.set_margin_bottom(2)
        self.set_margin_start(4)
        self.set_margin_end(4)

        # Get category color for border/background
        color = self.event_data.get("category_color", "#4285F4")
        color_override = self.event_data.get("color_override")
        if color_override:
            color = color_override

        # Apply color styling via inline CSS
        css_provider = Gtk.CssProvider()
        css = f"""
            .event-card {{
                border-left: 3px solid {color};
                border-radius: 4px;
                padding: 4px 6px;
                background-color: alpha({color}, 0.1);
            }}
            .event-card:hover {{
                background-color: alpha({color}, 0.2);
            }}
            .event-card.all-day {{
                background-color: {color};
                color: white;
                border-left: none;
            }}
            .event-time {{
                font-size: 0.85em;
                opacity: 0.8;
            }}
            .event-title {{
                font-weight: 500;
            }}
            .event-location {{
                font-size: 0.85em;
                opacity: 0.7;
            }}
        """
        css_provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Mark all-day events
        if self.event_data.get("is_all_day"):
            self.add_css_class("all-day")

    def _build_card(self) -> None:
        """Build the card contents."""
        # Time row (if not all-day and show_time is True)
        if self._show_time and not self.event_data.get("is_all_day"):
            time_label = Gtk.Label()
            time_label.set_xalign(0)
            time_label.add_css_class("event-time")

            start = self.event_data.get("start_time")
            end = self.event_data.get("end_time")
            time_text = self._format_time_range(start, end)
            time_label.set_text(time_text)
            self.append(time_label)

        # Title row
        title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Category icon (if showing)
        if self._show_category:
            icon = self.event_data.get("category_icon", "")
            if icon:
                icon_label = Gtk.Label(label=icon)
                icon_label.set_tooltip_text(
                    self.event_data.get("category_name", "")
                )
                title_box.append(icon_label)

        # Title
        title_label = Gtk.Label()
        title_label.set_xalign(0)
        title_label.set_hexpand(True)
        title_label.add_css_class("event-title")
        title_label.set_ellipsize(True)  # Truncate if too long

        title = self.event_data.get("title", "Untitled")
        title_label.set_text(title)
        title_label.set_tooltip_text(title)
        title_box.append(title_label)

        # Status indicator
        status = self.event_data.get("status", "confirmed")
        if status == "tentative":
            status_icon = Gtk.Image.new_from_icon_name("dialog-question-symbolic")
            status_icon.set_tooltip_text("Tentative")
            status_icon.set_opacity(0.6)
            title_box.append(status_icon)
        elif status == "cancelled":
            title_label.add_css_class("dim-label")
            self.add_css_class("event-cancelled")

        self.append(title_box)

        # Location (if not compact and has location)
        if not self._compact:
            location = self.event_data.get("location", "")
            if location:
                loc_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

                loc_icon = Gtk.Image.new_from_icon_name("mark-location-symbolic")
                loc_icon.set_pixel_size(12)
                loc_icon.set_opacity(0.6)
                loc_box.append(loc_icon)

                loc_label = Gtk.Label()
                loc_label.set_xalign(0)
                loc_label.add_css_class("event-location")
                loc_label.set_ellipsize(True)
                loc_label.set_text(location)
                loc_label.set_tooltip_text(location)
                loc_box.append(loc_label)

                self.append(loc_box)

            # Video call indicator
            url = self.event_data.get("url", "")
            if url and self._is_video_call_url(url):
                video_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

                video_icon = Gtk.Image.new_from_icon_name("camera-video-symbolic")
                video_icon.set_pixel_size(12)
                video_icon.set_opacity(0.6)
                video_box.append(video_icon)

                video_label = Gtk.Label(label="Video call")
                video_label.add_css_class("event-location")
                video_box.append(video_label)

                self.append(video_box)

            # Linked job/task indicators
            self._build_linked_entity_badges()

    def _build_linked_entity_badges(self) -> None:
        """Build badges for linked jobs and tasks."""
        linked_jobs = self.event_data.get("linked_jobs", [])
        linked_tasks = self.event_data.get("linked_tasks", [])

        if not linked_jobs and not linked_tasks:
            return

        links_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        links_box.set_margin_top(2)

        # Job badge(s)
        for job in linked_jobs[:2]:  # Max 2 badges
            job_name = job.get("job_name") or job.get("name") or "Job"
            job_status = job.get("status") or job.get("job_status")

            badge = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
            badge.add_css_class("linked-entity-badge")
            badge.add_css_class("job-badge")

            icon = Gtk.Image.new_from_icon_name("emblem-system-symbolic")
            icon.set_pixel_size(10)
            badge.append(icon)

            name_label = Gtk.Label(label=self._truncate(str(job_name), 12))
            name_label.add_css_class("caption")
            badge.append(name_label)

            if job_status:
                self._apply_status_class_to_badge(badge, str(job_status))

            badge.set_tooltip_text(f"Job: {job_name}")
            links_box.append(badge)

        # Task badge(s)
        for task in linked_tasks[:2]:  # Max 2 badges
            task_title = task.get("task_title") or task.get("title") or "Task"
            task_status = task.get("status") or task.get("task_status")

            badge = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
            badge.add_css_class("linked-entity-badge")
            badge.add_css_class("task-badge")

            icon = Gtk.Image.new_from_icon_name("view-list-symbolic")
            icon.set_pixel_size(10)
            badge.append(icon)

            name_label = Gtk.Label(label=self._truncate(str(task_title), 12))
            name_label.add_css_class("caption")
            badge.append(name_label)

            if task_status:
                self._apply_status_class_to_badge(badge, str(task_status))

            badge.set_tooltip_text(f"Task: {task_title}")
            links_box.append(badge)

        # Overflow indicator
        total = len(linked_jobs) + len(linked_tasks)
        shown = min(len(linked_jobs), 2) + min(len(linked_tasks), 2)
        if total > shown:
            more_label = Gtk.Label(label=f"+{total - shown}")
            more_label.add_css_class("dim-label")
            more_label.add_css_class("caption")
            links_box.append(more_label)

        self.append(links_box)

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 1] + "…"

    def _apply_status_class_to_badge(self, badge: Gtk.Box, status: str) -> None:
        """Apply CSS class to badge based on status."""
        status_lower = status.lower()
        if status_lower in ("done", "succeeded", "completed"):
            badge.add_css_class("status-success")
        elif status_lower in ("running", "in_progress"):
            badge.add_css_class("status-active")
        elif status_lower in ("failed", "cancelled"):
            badge.add_css_class("status-error")
        elif status_lower in ("scheduled", "ready", "pending"):
            badge.add_css_class("status-pending")

    def _setup_interactions(self) -> None:
        """Set up click and gesture handlers."""
        # Click gesture
        click = Gtk.GestureClick.new()
        click.set_button(1)  # Left click
        click.connect("released", self._on_click)
        self.add_controller(click)

        # Right-click for context menu
        right_click = Gtk.GestureClick.new()
        right_click.set_button(3)  # Right click
        right_click.connect("released", self._on_right_click)
        self.add_controller(right_click)

        # Make focusable
        self.set_focusable(True)

    def _on_click(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ) -> None:
        """Handle click event."""
        if n_press == 1:
            self.emit("clicked")
        elif n_press == 2:
            self.emit("edit-requested")

    def _on_right_click(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ) -> None:
        """Handle right-click for context menu."""
        menu = Gtk.PopoverMenu()
        menu.set_parent(self)

        menu_model = Gio_Menu_new()
        menu_model.append("Edit", "event.edit")
        menu_model.append("Delete", "event.delete")

        menu.set_menu_model(menu_model)
        menu.set_pointing_to(Gdk.Rectangle(int(x), int(y), 1, 1))
        menu.popup()

    def _format_time_range(
        self,
        start: Any,
        end: Any,
    ) -> str:
        """Format a time range for display."""
        start_dt = self._ensure_datetime(start)
        end_dt = self._ensure_datetime(end)

        if not start_dt:
            return ""

        start_str = start_dt.strftime("%H:%M")

        if end_dt and end_dt.date() == start_dt.date():
            end_str = end_dt.strftime("%H:%M")
            return f"{start_str} - {end_str}"

        return start_str

    def _ensure_datetime(self, value: Any) -> Optional[datetime]:
        """Convert value to datetime if needed."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _is_video_call_url(self, url: str) -> bool:
        """Check if URL looks like a video call link."""
        url_lower = url.lower()
        video_patterns = [
            "zoom.us",
            "meet.google.com",
            "teams.microsoft.com",
            "whereby.com",
            "webex.com",
            "gotomeeting.com",
            "bluejeans.com",
        ]
        return any(pattern in url_lower for pattern in video_patterns)


# Helper to avoid import issues
def Gio_Menu_new():
    """Create a new Gio.Menu."""
    from gi.repository import Gio
    return Gio.Menu.new()


class EventCardList(Gtk.Box):
    """A vertical list of event cards for a specific day."""

    def __init__(self, show_date_header: bool = True) -> None:
        """Initialize the event list.

        Args:
            show_date_header: Whether to show the date header
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._show_date_header = show_date_header
        self._cards: Dict[str, EventCard] = {}

    def set_events(
        self,
        events: list,
        date_obj: Optional[datetime] = None,
        on_event_clicked: Optional[Callable[[Dict], None]] = None,
        on_event_edit: Optional[Callable[[Dict], None]] = None,
        on_event_delete: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """Set the events to display.

        Args:
            events: List of event data dictionaries
            date_obj: Date for the header
            on_event_clicked: Callback when event is clicked
            on_event_edit: Callback when edit is requested
            on_event_delete: Callback when delete is requested
        """
        # Clear existing
        self._clear()

        # Add date header if configured
        if self._show_date_header and date_obj:
            header = Gtk.Label()
            header.set_xalign(0)
            header.add_css_class("heading")
            header.set_text(date_obj.strftime("%A, %B %d"))
            self.append(header)

        # Sort events by start time
        sorted_events = sorted(
            events,
            key=lambda e: e.get("start_time") or "",
        )

        # All-day events first
        all_day = [e for e in sorted_events if e.get("is_all_day")]
        timed = [e for e in sorted_events if not e.get("is_all_day")]

        for event in all_day + timed:
            card = EventCard(event)
            event_id = event.get("id", "")

            if on_event_clicked:
                card.connect("clicked", lambda c, e=event: on_event_clicked(e))
            if on_event_edit:
                card.connect("edit-requested", lambda c, e=event: on_event_edit(e))
            if on_event_delete:
                card.connect("delete-requested", lambda c, e=event: on_event_delete(e))

            self._cards[event_id] = card
            self.append(card)

        # Empty state
        if not events:
            empty = Gtk.Label(label="No events")
            empty.add_css_class("dim-label")
            empty.set_margin_top(20)
            empty.set_margin_bottom(20)
            self.append(empty)

    def _clear(self) -> None:
        """Remove all children."""
        child = self.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self.remove(child)
            child = next_child
        self._cards.clear()

    def get_card(self, event_id: str) -> Optional[EventCard]:
        """Get an event card by ID."""
        return self._cards.get(event_id)
