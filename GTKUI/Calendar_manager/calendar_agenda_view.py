"""GTK Calendar Agenda View widget.

Displays events in a scrollable list grouped by date.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Pango  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class AgendaEventRow(Gtk.ListBoxRow):
    """A single event row in the agenda view."""

    def __init__(
        self,
        event: Dict[str, Any],
        on_click: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        self._event = event
        self._on_click = on_click

        self.add_css_class("agenda-event-row")
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the event row UI."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Color indicator
        color = self._event.get("color", "#4285F4")
        color_bar = Gtk.DrawingArea()
        color_bar.set_size_request(4, 40)
        color_bar.set_draw_func(self._draw_color_bar, color)
        box.append(color_bar)

        # Time column
        time_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        time_box.set_size_request(70, -1)
        time_box.set_valign(Gtk.Align.CENTER)

        if self._event.get("is_all_day"):
            time_label = Gtk.Label(label="All day")
            time_label.add_css_class("dim-label")
            time_box.append(time_label)
        else:
            start_time = self._event.get("start_time")
            end_time = self._event.get("end_time")

            if isinstance(start_time, datetime):
                start_label = Gtk.Label(label=start_time.strftime("%H:%M"))
                start_label.set_xalign(0.0)
                time_box.append(start_label)

            if isinstance(end_time, datetime):
                end_label = Gtk.Label(label=end_time.strftime("%H:%M"))
                end_label.set_xalign(0.0)
                end_label.add_css_class("dim-label")
                end_label.add_css_class("caption")
                time_box.append(end_label)

        box.append(time_box)

        # Event details
        details_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        details_box.set_hexpand(True)
        details_box.set_valign(Gtk.Align.CENTER)

        # Title
        title = self._event.get("title", "Untitled")
        title_label = Gtk.Label(label=title)
        title_label.set_xalign(0.0)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("heading")
        details_box.append(title_label)

        # Location
        location = self._event.get("location")
        if location:
            loc_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            loc_icon = Gtk.Image.new_from_icon_name("mark-location-symbolic")
            loc_icon.set_pixel_size(12)
            loc_icon.add_css_class("dim-label")
            loc_box.append(loc_icon)

            loc_label = Gtk.Label(label=location)
            loc_label.set_xalign(0.0)
            loc_label.set_ellipsize(Pango.EllipsizeMode.END)
            loc_label.add_css_class("caption")
            loc_label.add_css_class("dim-label")
            loc_box.append(loc_label)
            details_box.append(loc_box)

        # Category
        category_name = self._event.get("category_name")
        if category_name:
            cat_label = Gtk.Label(label=category_name)
            cat_label.set_xalign(0.0)
            cat_label.add_css_class("caption")
            cat_label.add_css_class("dim-label")
            details_box.append(cat_label)

        box.append(details_box)

        # Chevron
        chevron = Gtk.Image.new_from_icon_name("go-next-symbolic")
        chevron.add_css_class("dim-label")
        box.append(chevron)

        self.set_child(box)

    def _draw_color_bar(
        self,
        area: Gtk.DrawingArea,
        cr: Any,
        width: int,
        height: int,
        color: str,
    ) -> None:
        """Draw color indicator bar."""
        try:
            from gi.repository import Gdk  # type: ignore[import-untyped]
            rgba = Gdk.RGBA()
            rgba.parse(color)
            cr.set_source_rgba(rgba.red, rgba.green, rgba.blue, rgba.alpha)
            cr.rectangle(0, 0, width, height)
            cr.fill()
        except Exception:
            pass

    @property
    def event(self) -> Dict[str, Any]:
        """Get event data."""
        return self._event

    @property
    def event_id(self) -> Optional[str]:
        """Get event ID."""
        return self._event.get("id")


class DateHeaderRow(Gtk.ListBoxRow):
    """A date header row in the agenda."""

    def __init__(self, header_date: date, is_today: bool = False):
        super().__init__()
        self._date = header_date
        self._is_today = is_today

        self.set_selectable(False)
        self.set_activatable(False)
        self.add_css_class("date-header-row")

        self._build_ui()

    def _build_ui(self) -> None:
        """Build header UI."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_top(16)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Weekday
        weekday = self._date.strftime("%A")
        weekday_label = Gtk.Label(label=weekday)
        weekday_label.add_css_class("heading")
        if self._is_today:
            weekday_label.add_css_class("accent")
        box.append(weekday_label)

        # Date
        date_str = self._date.strftime("%B %d")
        date_label = Gtk.Label(label=date_str)
        date_label.add_css_class("dim-label")
        box.append(date_label)

        # Today badge
        if self._is_today:
            badge = Gtk.Label(label="Today")
            badge.add_css_class("accent")
            badge.add_css_class("caption")
            box.append(badge)

        self.set_child(box)


class CalendarAgendaView(Gtk.Box):
    """Agenda view showing events in a scrollable list grouped by date.

    Provides a linear view of upcoming events, grouped by date headers,
    ideal for quickly scanning what's coming up.
    """

    def __init__(
        self,
        atlas: Any = None,
        on_event_clicked: Optional[Callable[[str], None]] = None,
        days_ahead: int = 14,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._atlas = atlas
        self._on_event_clicked = on_event_clicked
        self._days_ahead = days_ahead

        self._current_date = date.today()
        self._start_date = self._current_date
        self._repo = None

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-agenda-view")

        self._build_ui()
        self._init_repository()
        self._load_events()

    def _build_ui(self) -> None:
        """Build agenda view UI."""
        # Header
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_margin_top(8)
        header.set_margin_bottom(8)
        header.set_margin_start(12)
        header.set_margin_end(12)

        title = Gtk.Label(label="Upcoming Events")
        title.set_xalign(0.0)
        title.add_css_class("title-2")
        title.set_hexpand(True)
        header.append(title)

        # Days selector
        days_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        days_label = Gtk.Label(label="Show next")
        days_label.add_css_class("dim-label")
        days_box.append(days_label)

        self._days_dropdown = Gtk.DropDown.new_from_strings(
            ["7 days", "14 days", "30 days", "90 days"]
        )
        self._days_dropdown.set_selected(1)  # 14 days default
        self._days_dropdown.connect("notify::selected", self._on_days_changed)
        days_box.append(self._days_dropdown)

        header.append(days_box)
        self.append(header)

        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self.append(sep)

        # Scrollable list
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)

        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("agenda-list")
        self._list_box.connect("row-activated", self._on_row_activated)

        scroll.set_child(self._list_box)
        self.append(scroll)

        # Empty state
        self._empty_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self._empty_box.set_valign(Gtk.Align.CENTER)
        self._empty_box.set_halign(Gtk.Align.CENTER)
        self._empty_box.set_vexpand(True)
        self._empty_box.set_visible(False)

        empty_icon = Gtk.Image.new_from_icon_name("x-office-calendar-symbolic")
        empty_icon.set_pixel_size(64)
        empty_icon.add_css_class("dim-label")
        self._empty_box.append(empty_icon)

        empty_label = Gtk.Label(label="No upcoming events")
        empty_label.add_css_class("title-3")
        empty_label.add_css_class("dim-label")
        self._empty_box.append(empty_label)

        self.append(self._empty_box)

    def _init_repository(self) -> None:
        """Initialize repository."""
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

    def _on_days_changed(self, dropdown: Gtk.DropDown, param: Any) -> None:
        """Handle days selection change."""
        selected = dropdown.get_selected()
        days_map = {0: 7, 1: 14, 2: 30, 3: 90}
        self._days_ahead = days_map.get(selected, 14)
        self._load_events()

    def _load_events(self) -> None:
        """Load and display events."""
        # Clear list
        while row := self._list_box.get_first_child():
            self._list_box.remove(row)

        if not self._repo:
            self._show_empty_state(True)
            return

        try:
            start_dt = datetime.combine(self._start_date, time.min)
            end_dt = datetime.combine(
                self._start_date + timedelta(days=self._days_ahead),
                time.min,
            )

            events = self._repo.list_events(start=start_dt, end=end_dt)

            if not events:
                self._show_empty_state(True)
                return

            self._show_empty_state(False)

            # Add category info
            for event in events:
                cat_id = event.get("category_id")
                if cat_id:
                    cat = self._repo.get_category(cat_id)
                    if cat:
                        event["color"] = cat.get("color", "#4285F4")
                        event["category_name"] = cat.get("name")

            # Group by date
            events_by_date: Dict[date, List[Dict[str, Any]]] = {}
            for event in events:
                start_time = event.get("start_time")
                if isinstance(start_time, datetime):
                    event_date = start_time.date()
                else:
                    continue

                if event_date not in events_by_date:
                    events_by_date[event_date] = []
                events_by_date[event_date].append(event)

            # Sort dates
            sorted_dates = sorted(events_by_date.keys())

            # Build list
            for event_date in sorted_dates:
                is_today = event_date == self._current_date

                # Date header
                header = DateHeaderRow(event_date, is_today)
                self._list_box.append(header)

                # Events for this date
                day_events = sorted(
                    events_by_date[event_date],
                    key=lambda e: (
                        not e.get("is_all_day", False),
                        e.get("start_time", datetime.max),
                    ),
                )

                for event in day_events:
                    row = AgendaEventRow(event, self._on_event_clicked)
                    self._list_box.append(row)

        except Exception as e:
            logger.warning("Failed to load agenda events: %s", e)
            self._show_empty_state(True)

    def _show_empty_state(self, show: bool) -> None:
        """Toggle empty state visibility."""
        self._empty_box.set_visible(show)

    def _on_row_activated(self, list_box: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        """Handle row activation."""
        if isinstance(row, AgendaEventRow) and self._on_event_clicked:
            event_id = row.event_id
            if event_id:
                self._on_event_clicked(event_id)

    def refresh(self) -> None:
        """Refresh events."""
        self._current_date = date.today()
        self._start_date = self._current_date
        self._load_events()

    def set_days_ahead(self, days: int) -> None:
        """Set number of days to show."""
        self._days_ahead = days
        self._load_events()
