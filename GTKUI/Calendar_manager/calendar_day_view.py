"""GTK Calendar Day View widget.

Displays a single day view with hourly timeline and event positioning.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk, Pango # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Hour range and display settings
START_HOUR = 0
END_HOUR = 24
HOUR_HEIGHT = 80  # pixels per hour (larger for day view)


class DayEventBlock(Gtk.Box):
    """Event block for day view with more detail than week view."""

    def __init__(
        self,
        event: Dict[str, Any],
        on_click: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._event = event
        self._on_click = on_click

        self.add_css_class("day-event-block")
        self.set_margin_start(4)
        self.set_margin_end(4)

        color = event.get("color", "#4285F4")
        self._apply_color(color)
        self._build_ui()

    def _apply_color(self, color: str) -> None:
        """Apply background color."""
        try:
            css_provider = Gtk.CssProvider()
            css = f"""
                .day-event-block {{
                    background-color: {color};
                    border-radius: 6px;
                    padding: 8px;
                }}
                .day-event-block label {{
                    color: white;
                }}
            """
            css_provider.load_from_data(css.encode())
            self.get_style_context().add_provider(
                css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        except Exception as e:
            logger.debug("Failed to apply color: %s", e)

    def _build_ui(self) -> None:
        """Build event block UI."""
        # Title
        title = self._event.get("title", "Untitled")
        title_label = Gtk.Label(label=title)
        title_label.set_xalign(0.0)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("heading")
        self.append(title_label)

        # Time range
        start_time = self._event.get("start_time")
        end_time = self._event.get("end_time")
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            time_str = f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
            time_label = Gtk.Label(label=time_str)
            time_label.set_xalign(0.0)
            time_label.add_css_class("caption")
            self.append(time_label)

        # Location
        location = self._event.get("location")
        if location:
            loc_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            loc_icon = Gtk.Image.new_from_icon_name("mark-location-symbolic")
            loc_icon.set_pixel_size(12)
            loc_box.append(loc_icon)

            loc_label = Gtk.Label(label=location)
            loc_label.set_xalign(0.0)
            loc_label.set_ellipsize(Pango.EllipsizeMode.END)
            loc_label.add_css_class("caption")
            loc_box.append(loc_label)
            self.append(loc_box)

        # Click handler
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_clicked)
        self.add_controller(click)

    def _on_clicked(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float,
    ) -> None:
        """Handle click."""
        if self._on_click:
            event_id = self._event.get("id")
            if event_id:
                self._on_click(event_id)


class AllDayEventBar(Gtk.Box):
    """Bar for all-day events at top of day view."""

    def __init__(
        self,
        event: Dict[str, Any],
        on_click: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._event = event
        self._on_click = on_click

        self.add_css_class("all-day-event-bar")
        self.set_margin_start(4)
        self.set_margin_end(4)
        self.set_margin_top(2)
        self.set_margin_bottom(2)

        color = event.get("color", "#4285F4")
        self._apply_color(color)
        self._build_ui()

    def _apply_color(self, color: str) -> None:
        """Apply background color."""
        try:
            css_provider = Gtk.CssProvider()
            css = f"""
                .all-day-event-bar {{
                    background-color: {color};
                    border-radius: 4px;
                    padding: 4px 8px;
                }}
                .all-day-event-bar label {{
                    color: white;
                }}
            """
            css_provider.load_from_data(css.encode())
            self.get_style_context().add_provider(
                css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        except Exception:
            pass

    def _build_ui(self) -> None:
        """Build UI."""
        title = self._event.get("title", "Untitled")
        label = Gtk.Label(label=title)
        label.set_hexpand(True)
        label.set_xalign(0.0)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        self.append(label)

        click = Gtk.GestureClick()
        click.connect("pressed", self._on_clicked)
        self.add_controller(click)

    def _on_clicked(self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float) -> None:
        if self._on_click:
            event_id = self._event.get("id")
            if event_id:
                self._on_click(event_id)


class CalendarDayView(Gtk.Box):
    """Day view calendar widget showing a single day with hourly timeline.

    Displays a detailed day view with positioned event blocks
    and hour grid markers.
    """

    def __init__(
        self,
        atlas: Any = None,
        on_hour_clicked: Optional[Callable[[date, int], None]] = None,
        on_event_clicked: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._atlas = atlas
        self._on_hour_clicked = on_hour_clicked
        self._on_event_clicked = on_event_clicked

        self._current_date = date.today()
        self._view_date = self._current_date
        self._events: List[Dict[str, Any]] = []
        self._repo = None

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-day-view")

        self._build_ui()
        self._init_repository()
        self._load_events()

    def _build_ui(self) -> None:
        """Build the day view UI."""
        # Navigation header
        nav_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        nav_box.set_margin_top(8)
        nav_box.set_margin_bottom(8)
        nav_box.set_margin_start(8)
        nav_box.set_margin_end(8)

        # Previous day
        prev_btn = Gtk.Button()
        prev_btn.set_icon_name("go-previous-symbolic")
        prev_btn.add_css_class("flat")
        prev_btn.connect("clicked", lambda _: self._navigate(-1))
        nav_box.append(prev_btn)

        # Date label
        self._header_label = Gtk.Label()
        self._header_label.set_hexpand(True)
        self._header_label.add_css_class("title-2")
        nav_box.append(self._header_label)

        # Today button
        today_btn = Gtk.Button(label="Today")
        today_btn.add_css_class("flat")
        today_btn.connect("clicked", lambda _: self.go_to_today())
        nav_box.append(today_btn)

        # Next day
        next_btn = Gtk.Button()
        next_btn.set_icon_name("go-next-symbolic")
        next_btn.add_css_class("flat")
        next_btn.connect("clicked", lambda _: self._navigate(1))
        nav_box.append(next_btn)

        self.append(nav_box)

        # All-day events section
        self._all_day_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._all_day_box.add_css_class("all-day-section")
        self.append(self._all_day_box)

        # Main scrollable content
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        main_box.set_vexpand(True)

        # Time labels
        time_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        time_col.set_size_request(60, -1)
        time_col.add_css_class("time-labels")

        time_scroll = Gtk.ScrolledWindow()
        time_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        time_scroll.set_vexpand(True)

        time_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        for hour in range(START_HOUR, END_HOUR):
            hour_box = Gtk.Box()
            hour_box.set_size_request(-1, HOUR_HEIGHT)

            hour_label = Gtk.Label(label=f"{hour:02d}:00")
            hour_label.set_valign(Gtk.Align.START)
            hour_label.set_margin_top(4)
            hour_label.set_margin_end(8)
            hour_label.add_css_class("dim-label")
            hour_box.append(hour_label)

            time_grid.append(hour_box)

        time_scroll.set_child(time_grid)
        time_col.append(time_scroll)
        main_box.append(time_col)

        # Events area with overlay
        self._events_scroll = Gtk.ScrolledWindow()
        self._events_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._events_scroll.set_hexpand(True)
        self._events_scroll.set_vexpand(True)

        # Sync scrolling
        adj = self._events_scroll.get_vadjustment()
        time_scroll.set_vadjustment(adj)

        self._overlay = Gtk.Overlay()

        # Hour grid background
        self._hour_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._hour_grid.set_hexpand(True)

        for hour in range(START_HOUR, END_HOUR):
            hour_slot = Gtk.Box()
            hour_slot.set_size_request(-1, HOUR_HEIGHT)
            hour_slot.add_css_class("hour-slot")
            hour_slot.set_hexpand(True)

            # Double-click to create event
            click = Gtk.GestureClick()
            click.connect("pressed", self._on_hour_slot_clicked, hour)
            hour_slot.add_controller(click)

            self._hour_grid.append(hour_slot)

        self._overlay.set_child(self._hour_grid)

        # Events container
        self._events_container = Gtk.Fixed()
        self._events_container.set_hexpand(True)
        self._overlay.add_overlay(self._events_container)

        self._events_scroll.set_child(self._overlay)
        main_box.append(self._events_scroll)

        self.append(main_box)

        # Update header
        self._update_header()

    def _update_header(self) -> None:
        """Update the header label."""
        weekday = self._view_date.strftime("%A")
        date_str = self._view_date.strftime("%B %d, %Y")

        if self._view_date == self._current_date:
            self._header_label.set_label(f"Today - {weekday}, {date_str}")
        else:
            self._header_label.set_label(f"{weekday}, {date_str}")

    def _init_repository(self) -> None:
        """Initialize calendar repository."""
        try:
            from modules.calendar_store import CalendarStoreRepository

            session_factory = self._get_session_factory()
            if session_factory:
                self._repo = CalendarStoreRepository(session_factory)
        except Exception as e:
            logger.debug("Could not initialize repository: %s", e)

    def _get_session_factory(self) -> Any:
        """Get session factory."""
        if self._atlas and hasattr(self._atlas, "services"):
            if "db_session" in self._atlas.services:
                return self._atlas.services["db_session"]
        return None

    def _load_events(self) -> None:
        """Load events for current day."""
        self._update_header()

        # Clear events
        while child := self._all_day_box.get_first_child():
            self._all_day_box.remove(child)
        while child := self._events_container.get_first_child():
            self._events_container.remove(child)

        if not self._repo:
            return

        try:
            start_dt = datetime.combine(self._view_date, time.min)
            end_dt = datetime.combine(self._view_date + timedelta(days=1), time.min)

            events = self._repo.list_events(start=start_dt, end=end_dt)

            # Add category colors
            for event in events:
                cat_id = event.get("category_id")
                if cat_id:
                    cat = self._repo.get_category(cat_id)
                    if cat:
                        event["color"] = cat.get("color", "#4285F4")

            # Separate all-day and timed events
            all_day_events = [e for e in events if e.get("is_all_day")]
            timed_events = [e for e in events if not e.get("is_all_day")]

            # Render all-day events
            for event in all_day_events:
                bar = AllDayEventBar(event, self._on_event_clicked)
                self._all_day_box.append(bar)

            # Render timed events
            # Calculate column width for overlapping events
            self._position_timed_events(timed_events)

        except Exception as e:
            logger.warning("Failed to load day events: %s", e)

    def _position_timed_events(self, events: List[Dict[str, Any]]) -> None:
        """Position timed events, handling overlaps."""
        if not events:
            return

        # Sort by start time
        sorted_events = sorted(
            events,
            key=lambda e: e.get("start_time", datetime.min),
        )

        # Simple positioning (no overlap handling for now)
        for event in sorted_events:
            start_time = event.get("start_time")
            end_time = event.get("end_time")

            if not isinstance(start_time, datetime) or not isinstance(end_time, datetime):
                continue

            # Calculate position
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            duration_minutes = max(end_minutes - start_minutes, 30)

            top = (start_minutes / 60) * HOUR_HEIGHT
            height = (duration_minutes / 60) * HOUR_HEIGHT

            # Create block
            block = DayEventBlock(event, self._on_event_clicked)
            block.set_size_request(300, int(height))

            self._events_container.put(block, 0, top)

    def _on_hour_slot_clicked(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float,
        hour: int,
    ) -> None:
        """Handle hour slot double-click."""
        if self._on_hour_clicked and n_press == 2:
            self._on_hour_clicked(self._view_date, hour)

    def _navigate(self, delta_days: int) -> None:
        """Navigate by days."""
        self._view_date += timedelta(days=delta_days)
        self._load_events()

    def go_to_today(self) -> None:
        """Go to today."""
        self._current_date = date.today()
        self._view_date = self._current_date
        self._load_events()

    def go_to_date(self, target_date: date) -> None:
        """Go to specific date."""
        self._view_date = target_date
        self._load_events()

    def refresh(self) -> None:
        """Refresh events."""
        self._load_events()

    @property
    def view_date(self) -> date:
        """Get current view date."""
        return self._view_date
