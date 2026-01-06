"""Mini calendar widget for sidebar integration.

Provides a compact month view with date selection and today's events summary.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Gdk, Pango # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class MiniCalendar(Gtk.Box):
    """Compact calendar widget showing a month grid with event indicators."""

    def __init__(
        self,
        atlas: Any = None,
        on_date_selected: Optional[Callable[[date], None]] = None,
        on_open_calendar: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.ATLAS = atlas
        self._on_date_selected = on_date_selected
        self._on_open_calendar = on_open_calendar

        self._current_month = date.today().replace(day=1)
        self._selected_date: date = date.today()
        self._event_dates: set[date] = set()
        self._day_buttons: dict[int, Gtk.Button] = {}

        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(4)
        self.set_margin_end(4)

        self._build_ui()
        self._load_events_for_month()

    def _build_ui(self) -> None:
        """Build the mini calendar UI."""
        # Header with month/year and navigation
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        header.set_halign(Gtk.Align.FILL)

        prev_btn = Gtk.Button()
        prev_btn.set_icon_name("go-previous-symbolic")
        prev_btn.add_css_class("flat")
        prev_btn.add_css_class("mini-calendar-nav")
        prev_btn.set_tooltip_text("Previous month")
        prev_btn.connect("clicked", self._on_prev_month)
        header.append(prev_btn)

        self._month_label = Gtk.Label()
        self._month_label.set_hexpand(True)
        self._month_label.add_css_class("mini-calendar-month")
        header.append(self._month_label)

        next_btn = Gtk.Button()
        next_btn.set_icon_name("go-next-symbolic")
        next_btn.add_css_class("flat")
        next_btn.add_css_class("mini-calendar-nav")
        next_btn.set_tooltip_text("Next month")
        next_btn.connect("clicked", self._on_next_month)
        header.append(next_btn)

        self.append(header)

        # Day-of-week headers
        dow_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        dow_box.set_halign(Gtk.Align.FILL)
        dow_box.set_homogeneous(True)
        for day_name in ["S", "M", "T", "W", "T", "F", "S"]:
            lbl = Gtk.Label(label=day_name)
            lbl.add_css_class("mini-calendar-dow")
            lbl.set_size_request(24, 20)
            dow_box.append(lbl)
        self.append(dow_box)

        # Day grid (6 weeks max)
        self._day_grid = Gtk.Grid()
        self._day_grid.set_row_homogeneous(True)
        self._day_grid.set_column_homogeneous(True)
        self._day_grid.set_halign(Gtk.Align.FILL)
        self._day_grid.set_hexpand(True)
        self.append(self._day_grid)

        # Today's events summary
        events_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        events_header.set_margin_top(8)

        events_label = Gtk.Label(label="Today")
        events_label.set_xalign(0.0)
        events_label.add_css_class("mini-calendar-events-header")
        events_label.set_hexpand(True)
        events_header.append(events_label)

        if self._on_open_calendar:
            callback = self._on_open_calendar
            open_btn = Gtk.Button()
            open_btn.set_icon_name("go-next-symbolic")
            open_btn.add_css_class("flat")
            open_btn.add_css_class("mini-calendar-open")
            open_btn.set_tooltip_text("Open Calendar")
            open_btn.connect("clicked", lambda _: callback())
            events_header.append(open_btn)

        self.append(events_header)

        # Events list (compact)
        self._events_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._events_box.add_css_class("mini-calendar-events")
        self.append(self._events_box)

        self._update_month_display()

    def _update_month_display(self) -> None:
        """Update the calendar grid for the current month."""
        # Update header
        month_name = self._current_month.strftime("%B %Y")
        self._month_label.set_label(month_name)

        # Clear existing day buttons
        while child := self._day_grid.get_first_child():
            self._day_grid.remove(child)
        self._day_buttons.clear()

        # Get calendar data
        cal = calendar.Calendar(firstweekday=6)  # Sunday first
        month_days = cal.monthdayscalendar(
            self._current_month.year, self._current_month.month
        )

        today = date.today()

        for week_idx, week in enumerate(month_days):
            for day_idx, day in enumerate(week):
                if day == 0:
                    # Empty cell
                    placeholder = Gtk.Label(label="")
                    placeholder.set_size_request(24, 24)
                    self._day_grid.attach(placeholder, day_idx, week_idx, 1, 1)
                else:
                    btn = Gtk.Button(label=str(day))
                    btn.add_css_class("flat")
                    btn.add_css_class("mini-calendar-day")
                    btn.set_size_request(24, 24)

                    current_date = self._current_month.replace(day=day)

                    # Highlight today
                    if current_date == today:
                        btn.add_css_class("mini-calendar-today")

                    # Highlight selected
                    if current_date == self._selected_date:
                        btn.add_css_class("mini-calendar-selected")

                    # Event indicator
                    if current_date in self._event_dates:
                        btn.add_css_class("mini-calendar-has-event")

                    btn.connect("clicked", self._on_day_clicked, day)
                    self._day_grid.attach(btn, day_idx, week_idx, 1, 1)
                    self._day_buttons[day] = btn

    def _on_prev_month(self, _button: Gtk.Button) -> None:
        """Navigate to previous month."""
        year = self._current_month.year
        month = self._current_month.month - 1
        if month < 1:
            month = 12
            year -= 1
        self._current_month = date(year, month, 1)
        self._load_events_for_month()
        self._update_month_display()

    def _on_next_month(self, _button: Gtk.Button) -> None:
        """Navigate to next month."""
        year = self._current_month.year
        month = self._current_month.month + 1
        if month > 12:
            month = 1
            year += 1
        self._current_month = date(year, month, 1)
        self._load_events_for_month()
        self._update_month_display()

    def _on_day_clicked(self, _button: Gtk.Button, day: int) -> None:
        """Handle day button click."""
        self._selected_date = self._current_month.replace(day=day)
        self._update_month_display()
        self._update_events_summary()

        if self._on_date_selected:
            self._on_date_selected(self._selected_date)

    def _load_events_for_month(self) -> None:
        """Load event dates for the current month from storage."""
        self._event_dates.clear()

        if self.ATLAS is None:
            return

        try:
            # Get storage manager
            storage = getattr(self.ATLAS, "storage", None)
            if storage is None:
                return

            calendars = getattr(storage, "calendars", None)
            if calendars is None:
                return

            # Calculate month range
            year = self._current_month.year
            month = self._current_month.month

            # First day of month
            start_date = datetime(year, month, 1)

            # Last day of month
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)

            # Query events
            events = calendars.list_events(start_date=start_date, end_date=end_date)

            for event in events:
                event_start = event.start_time
                if hasattr(event_start, "date"):
                    self._event_dates.add(event_start.date())
                else:
                    self._event_dates.add(event_start)

        except Exception as e:
            logger.warning(f"Failed to load events for mini calendar: {e}")

    def _update_events_summary(self) -> None:
        """Update the today's events summary section."""
        # Clear existing
        while child := self._events_box.get_first_child():
            self._events_box.remove(child)

        if self.ATLAS is None:
            self._show_no_events()
            return

        try:
            storage = getattr(self.ATLAS, "storage", None)
            if storage is None:
                self._show_no_events()
                return

            calendars = getattr(storage, "calendars", None)
            if calendars is None:
                self._show_no_events()
                return

            # Get events for selected date
            start_dt = datetime.combine(self._selected_date, datetime.min.time())
            end_dt = datetime.combine(self._selected_date, datetime.max.time())

            events = calendars.list_events(start_date=start_dt, end_date=end_dt)

            if not events:
                self._show_no_events()
                return

            # Show up to 3 events
            for event in events[:3]:
                row = self._create_event_row(event)
                self._events_box.append(row)

            if len(events) > 3:
                more_label = Gtk.Label(label=f"+{len(events) - 3} more")
                more_label.set_xalign(0.0)
                more_label.add_css_class("dim-label")
                more_label.add_css_class("mini-calendar-more")
                self._events_box.append(more_label)

        except Exception as e:
            logger.warning(f"Failed to load events summary: {e}")
            self._show_no_events()

    def _create_event_row(self, event: Any) -> Gtk.Box:
        """Create a compact event row."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        row.add_css_class("mini-calendar-event-row")

        # Color dot
        dot = Gtk.Label(label="●")
        color = getattr(event, "color", None) or "#4285F4"
        dot.set_markup(f'<span foreground="{color}">●</span>')
        row.append(dot)

        # Time
        start = event.start_time
        if event.all_day:
            time_str = "All day"
        elif hasattr(start, "strftime"):
            time_str = start.strftime("%H:%M")
        else:
            time_str = ""

        if time_str:
            time_label = Gtk.Label(label=time_str)
            time_label.add_css_class("mini-calendar-event-time")
            time_label.add_css_class("dim-label")
            row.append(time_label)

        # Title (truncated)
        title = event.title or "Untitled"
        if len(title) > 15:
            title = title[:14] + "…"
        title_label = Gtk.Label(label=title)
        title_label.set_xalign(0.0)
        title_label.set_hexpand(True)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("mini-calendar-event-title")
        row.append(title_label)

        return row

    def _show_no_events(self) -> None:
        """Show 'no events' placeholder."""
        label = Gtk.Label(label="No events")
        label.set_xalign(0.0)
        label.add_css_class("dim-label")
        label.add_css_class("mini-calendar-no-events")
        self._events_box.append(label)

    def select_date(self, new_date: date) -> None:
        """Programmatically select a date."""
        self._selected_date = new_date
        # Navigate to the month if different
        if (
            new_date.year != self._current_month.year
            or new_date.month != self._current_month.month
        ):
            self._current_month = new_date.replace(day=1)
            self._load_events_for_month()
        self._update_month_display()
        self._update_events_summary()

    def refresh(self) -> None:
        """Refresh the calendar display and events."""
        self._load_events_for_month()
        self._update_month_display()
        self._update_events_summary()

    def go_to_today(self) -> None:
        """Navigate to today's date."""
        self.select_date(date.today())


# CSS for mini calendar styling
MINI_CALENDAR_CSS = """
.mini-calendar-month {
    font-weight: bold;
    font-size: 0.9em;
}

.mini-calendar-dow {
    font-size: 0.75em;
    color: @theme_fg_color;
    opacity: 0.6;
}

.mini-calendar-day {
    font-size: 0.8em;
    min-width: 24px;
    min-height: 24px;
    padding: 2px;
    border-radius: 50%;
}

.mini-calendar-today {
    background-color: @accent_bg_color;
    color: @accent_fg_color;
}

.mini-calendar-selected {
    border: 2px solid @accent_color;
}

.mini-calendar-has-event {
    font-weight: bold;
}

.mini-calendar-has-event::after {
    content: "•";
    position: absolute;
    bottom: 2px;
    color: @accent_color;
}

.mini-calendar-nav {
    min-width: 24px;
    min-height: 24px;
    padding: 0;
}

.mini-calendar-events-header {
    font-weight: bold;
    font-size: 0.85em;
}

.mini-calendar-event-row {
    padding: 2px 0;
}

.mini-calendar-event-time {
    font-size: 0.75em;
}

.mini-calendar-event-title {
    font-size: 0.8em;
}

.mini-calendar-more {
    font-size: 0.75em;
    padding-left: 12px;
}

.mini-calendar-no-events {
    font-size: 0.8em;
    padding: 4px 0;
}
"""
