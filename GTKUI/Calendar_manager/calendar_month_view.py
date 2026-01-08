"""GTK Calendar Month View widget.

Displays a month grid with date cells, event indicators, and navigation.
"""

from __future__ import annotations

import calendar
import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk, Pango # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Weekday headers
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_NAMES_FULL = [
    "Monday", "Tuesday", "Wednesday", "Thursday", 
    "Friday", "Saturday", "Sunday"
]


class DayCell(Gtk.Box):
    """A single day cell in the month grid."""

    def __init__(
        self,
        cell_date: date,
        is_current_month: bool = True,
        is_today: bool = False,
        on_selected: Optional[Callable[["DayCell", date], None]] = None,
        on_activated: Optional[Callable[["DayCell", date], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.cell_date = cell_date
        self.is_current_month = is_current_month
        self.is_today = is_today
        self._on_selected = on_selected
        self._on_activated = on_activated
        self._events: List[Dict[str, Any]] = []
        self._max_visible_events = 3

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-day-cell")

        if not is_current_month:
            self.add_css_class("other-month")

        if is_today:
            self.add_css_class("today")

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the cell UI."""
        # Day number header
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header.set_margin_start(4)
        header.set_margin_end(4)
        header.set_margin_top(2)

        self._day_label = Gtk.Label(label=str(self.cell_date.day))
        self._day_label.set_xalign(0.0)
        if self.is_today:
            self._day_label.add_css_class("accent")
            self._day_label.add_css_class("heading")
        elif not self.is_current_month:
            self._day_label.add_css_class("dim-label")
        header.append(self._day_label)

        self.append(header)

        # Events container
        self._events_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        self._events_box.set_margin_start(2)
        self._events_box.set_margin_end(2)
        self.append(self._events_box)

        # "More" indicator
        self._more_label = Gtk.Label()
        self._more_label.set_xalign(0.0)
        self._more_label.add_css_class("dim-label")
        self._more_label.add_css_class("caption")
        self._more_label.set_margin_start(4)
        self._more_label.set_visible(False)
        self.append(self._more_label)

        # Click gesture
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_clicked)
        self.add_controller(click)

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        """Set events for this day cell."""
        self._events = events

        # Clear existing event widgets
        while child := self._events_box.get_first_child():
            self._events_box.remove(child)

        # Add event indicators
        visible_count = min(len(events), self._max_visible_events)
        for i in range(visible_count):
            event = events[i]
            indicator = self._create_event_indicator(event)
            self._events_box.append(indicator)

        # Show "more" indicator
        remaining = len(events) - visible_count
        if remaining > 0:
            self._more_label.set_label(f"+{remaining} more")
            self._more_label.set_visible(True)
        else:
            self._more_label.set_visible(False)

    def _create_event_indicator(self, event: Dict[str, Any]) -> Gtk.Box:
        """Create an event indicator widget."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        box.add_css_class("event-indicator")

        # Color dot
        color = event.get("color", "#4285F4")
        dot = Gtk.DrawingArea()
        dot.set_size_request(8, 8)
        dot.set_draw_func(self._draw_color_dot, color)
        box.append(dot)

        # Event title (truncated)
        title = event.get("title", "Untitled")
        label = Gtk.Label(label=title)
        label.set_xalign(0.0)
        label.set_ellipsize(Pango.EllipsizeMode.END)
        label.set_hexpand(True)
        label.add_css_class("caption")

        # All-day events show differently
        if event.get("is_all_day"):
            box.add_css_class("all-day-event")

        box.append(label)
        return box

    def _draw_color_dot(
        self,
        area: Gtk.DrawingArea,
        cr: Any,
        width: int,
        height: int,
        color: str,
    ) -> None:
        """Draw a colored dot."""
        try:
            rgba = Gdk.RGBA()
            rgba.parse(color)
            cr.set_source_rgba(rgba.red, rgba.green, rgba.blue, rgba.alpha)
            cr.arc(width / 2, height / 2, min(width, height) / 2 - 1, 0, 2 * 3.14159)
            cr.fill()
        except Exception:
            pass

    def _on_clicked(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float,
    ) -> None:
        """Handle cell click."""
        if n_press == 1 and self._on_selected:
            self._on_selected(self, self.cell_date)
        elif n_press == 2 and self._on_activated:
            self._on_activated(self, self.cell_date)

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Get events for this cell."""
        return self._events


class CalendarMonthView(Gtk.Box):
    """Month view calendar widget showing a grid of days.

    Displays a traditional month calendar with event indicators,
    navigation controls, and click handlers for day selection.
    """

    def __init__(
        self,
        atlas: Any = None,
        on_date_selected: Optional[Callable[[date], None]] = None,
        on_date_activated: Optional[Callable[[date], None]] = None,
        on_event_clicked: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._atlas = atlas
        self._on_date_selected = on_date_selected
        self._on_date_activated = on_date_activated
        self._on_event_clicked = on_event_clicked

        self._current_date = date.today()
        self._selected_date: Optional[date] = None
        self._view_year = self._current_date.year
        self._view_month = self._current_date.month
        self._day_cells: Dict[date, DayCell] = {}
        self._events_by_date: Dict[date, List[Dict[str, Any]]] = {}
        self._repo = None

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-month-view")

        self._build_ui()
        self._init_repository()
        self._populate_grid()

    def _build_ui(self) -> None:
        """Build the month view UI."""
        # Navigation header
        nav_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        nav_box.set_margin_top(8)
        nav_box.set_margin_bottom(8)
        nav_box.set_margin_start(8)
        nav_box.set_margin_end(8)

        # Previous month button
        prev_btn = Gtk.Button()
        prev_btn.set_icon_name("go-previous-symbolic")
        prev_btn.add_css_class("flat")
        prev_btn.connect("clicked", lambda _: self._navigate(-1))
        nav_box.append(prev_btn)

        # Month/Year label
        self._header_label = Gtk.Label()
        self._header_label.set_hexpand(True)
        self._header_label.add_css_class("title-2")
        nav_box.append(self._header_label)

        # Today button
        today_btn = Gtk.Button(label="Today")
        today_btn.add_css_class("flat")
        today_btn.connect("clicked", lambda _: self.go_to_today())
        nav_box.append(today_btn)

        # Next month button
        next_btn = Gtk.Button()
        next_btn.set_icon_name("go-next-symbolic")
        next_btn.add_css_class("flat")
        next_btn.connect("clicked", lambda _: self._navigate(1))
        nav_box.append(next_btn)

        self.append(nav_box)

        # Weekday headers
        weekday_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        weekday_box.set_homogeneous(True)
        weekday_box.add_css_class("weekday-header")

        for name in WEEKDAY_NAMES:
            label = Gtk.Label(label=name)
            label.add_css_class("dim-label")
            label.add_css_class("heading")
            label.set_hexpand(True)
            weekday_box.append(label)

        self.append(weekday_box)

        # Calendar grid
        self._grid = Gtk.Grid()
        self._grid.set_row_homogeneous(True)
        self._grid.set_column_homogeneous(True)
        self._grid.set_row_spacing(1)
        self._grid.set_column_spacing(1)
        self._grid.set_hexpand(True)
        self._grid.set_vexpand(True)
        self._grid.add_css_class("calendar-grid")
        self.append(self._grid)

    def _init_repository(self) -> None:
        """Initialize the calendar repository."""
        # Service is accessed directly from ATLAS when needed
        pass

    def _get_session_factory(self) -> Any:
        """Get session factory from ATLAS or config."""
        if self._atlas and hasattr(self._atlas, "services"):
            if "db_session" in self._atlas.services:
                return self._atlas.services["db_session"]
        return None

    def _populate_grid(self) -> None:
        """Populate the calendar grid for current month."""
        # Update header
        month_name = calendar.month_name[self._view_month]
        self._header_label.set_label(f"{month_name} {self._view_year}")

        # Clear existing cells
        while child := self._grid.get_first_child():
            self._grid.remove(child)
        self._day_cells.clear()

        # Get calendar info
        cal = calendar.Calendar(firstweekday=0)  # Monday start
        month_days = cal.monthdatescalendar(self._view_year, self._view_month)

        # Create day cells
        for row, week in enumerate(month_days):
            for col, day_date in enumerate(week):
                is_current_month = day_date.month == self._view_month
                is_today = day_date == self._current_date

                cell = DayCell(
                    day_date,
                    is_current_month,
                    is_today,
                    on_selected=self._on_day_selected,
                    on_activated=self._on_day_activated,
                )

                self._grid.attach(cell, col, row, 1, 1)
                self._day_cells[day_date] = cell

        # Load events for visible range
        self._load_events()

    def _load_events(self) -> None:
        """Load events for the visible date range."""
        if not self._repo:
            return

        try:
            # Get date range (include padding for other-month days)
            visible_dates = list(self._day_cells.keys())
            if not visible_dates:
                return

            start_date = min(visible_dates)
            end_date = max(visible_dates) + timedelta(days=1)

            # Query events
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())

            events = self._repo.list_events(start=start_dt, end=end_dt)

            # Group by date
            self._events_by_date.clear()
            for event in events:
                event_start = event.get("start_time")
                if isinstance(event_start, datetime):
                    event_date = event_start.date()
                elif isinstance(event_start, date):
                    event_date = event_start
                else:
                    continue

                if event_date not in self._events_by_date:
                    self._events_by_date[event_date] = []

                # Get category color
                cat_id = event.get("category_id")
                if cat_id:
                    cat = self._repo.get_category(cat_id)
                    if cat:
                        event["color"] = cat.get("color", "#4285F4")

                self._events_by_date[event_date].append(event)

            # Update cells
            for cell_date, cell in self._day_cells.items():
                events = self._events_by_date.get(cell_date, [])
                cell.set_events(events)

        except Exception as e:
            logger.warning("Failed to load events: %s", e)

    def _navigate(self, delta_months: int) -> None:
        """Navigate by delta months."""
        new_month = self._view_month + delta_months
        new_year = self._view_year

        while new_month < 1:
            new_month += 12
            new_year -= 1
        while new_month > 12:
            new_month -= 12
            new_year += 1

        self._view_month = new_month
        self._view_year = new_year
        self._populate_grid()

    def _on_day_selected(self, cell: DayCell, selected_date: date) -> None:
        """Handle day selection."""
        # Update visual selection
        if self._selected_date and self._selected_date in self._day_cells:
            self._day_cells[self._selected_date].remove_css_class("selected")

        self._selected_date = selected_date
        cell.add_css_class("selected")

        # Callback
        if self._on_date_selected:
            self._on_date_selected(selected_date)

    def _on_day_activated(self, cell: DayCell, activated_date: date) -> None:
        """Handle day double-click (activation)."""
        if self._on_date_activated:
            self._on_date_activated(activated_date)

    def go_to_today(self) -> None:
        """Navigate to today's date."""
        self._current_date = date.today()
        self._view_year = self._current_date.year
        self._view_month = self._current_date.month
        self._populate_grid()

    def go_to_date(self, target_date: date) -> None:
        """Navigate to a specific date."""
        self._view_year = target_date.year
        self._view_month = target_date.month
        self._populate_grid()

        # Select the date
        if target_date in self._day_cells:
            self._on_day_selected(self._day_cells[target_date], target_date)

    def refresh(self) -> None:
        """Refresh events from repository."""
        self._load_events()

    @property
    def selected_date(self) -> Optional[date]:
        """Get currently selected date."""
        return self._selected_date

    @property
    def view_month(self) -> Tuple[int, int]:
        """Get current view month as (year, month)."""
        return (self._view_year, self._view_month)
