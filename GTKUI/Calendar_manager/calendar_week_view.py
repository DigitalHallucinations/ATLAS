"""GTK Calendar Week View widget.

Displays a week view with hourly columns and event blocks.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, Gtk, Pango # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Hour range to display (0-24)
START_HOUR = 0
END_HOUR = 24
HOUR_HEIGHT = 60  # pixels per hour


class EventBlock(Gtk.Box):
    """A visual block representing an event on the week/day view."""

    def __init__(
        self,
        event: Dict[str, Any],
        on_click: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._event = event
        self._on_click = on_click

        self.add_css_class("event-block")
        self.set_margin_start(2)
        self.set_margin_end(2)

        # Set background color
        color = event.get("color", "#4285F4")
        self._apply_color(color)

        self._build_ui()

    def _apply_color(self, color: str) -> None:
        """Apply background color via CSS."""
        try:
            css_provider = Gtk.CssProvider()
            css = f"""
                .event-block {{
                    background-color: {color};
                    border-radius: 4px;
                    padding: 4px;
                }}
                .event-block label {{
                    color: white;
                }}
            """
            css_provider.load_from_data(css.encode())
            self.get_style_context().add_provider(
                css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        except Exception as e:
            logger.debug("Failed to apply event color: %s", e)

    def _build_ui(self) -> None:
        """Build the event block UI."""
        # Title
        title = self._event.get("title", "Untitled")
        title_label = Gtk.Label(label=title)
        title_label.set_xalign(0.0)
        title_label.set_ellipsize(Pango.EllipsizeMode.END)
        title_label.add_css_class("caption")
        title_label.add_css_class("heading")
        self.append(title_label)

        # Time
        start_time = self._event.get("start_time")
        if isinstance(start_time, datetime):
            time_str = start_time.strftime("%H:%M")
            time_label = Gtk.Label(label=time_str)
            time_label.set_xalign(0.0)
            time_label.add_css_class("caption")
            self.append(time_label)

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
        """Handle click on event block."""
        if self._on_click:
            event_id = self._event.get("id")
            if event_id:
                self._on_click(event_id)

    @property
    def event(self) -> Dict[str, Any]:
        """Get the event data."""
        return self._event


class DayColumn(Gtk.Box):
    """A single day column in the week view."""

    def __init__(
        self,
        column_date: date,
        is_today: bool = False,
        on_hour_clicked: Optional[Callable[[date, int], None]] = None,
        on_event_clicked: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.column_date = column_date
        self.is_today = is_today
        self._on_hour_clicked = on_hour_clicked
        self._on_event_clicked = on_event_clicked
        self._events: List[Dict[str, Any]] = []
        self._event_blocks: List[EventBlock] = []

        self.set_hexpand(True)
        self.add_css_class("day-column")

        if is_today:
            self.add_css_class("today-column")

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the day column UI."""
        # Day header
        header = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        header.add_css_class("day-column-header")
        header.set_margin_top(4)
        header.set_margin_bottom(4)

        # Weekday name
        weekday = self.column_date.strftime("%a")
        weekday_label = Gtk.Label(label=weekday)
        weekday_label.add_css_class("dim-label")
        header.append(weekday_label)

        # Day number
        day_label = Gtk.Label(label=str(self.column_date.day))
        if self.is_today:
            day_label.add_css_class("accent")
            day_label.add_css_class("title-2")
        else:
            day_label.add_css_class("title-3")
        header.append(day_label)

        self.append(header)

        # Scrollable hour grid with events overlay
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.EXTERNAL)
        scroll.set_vexpand(True)

        # Overlay for events on top of hour grid
        self._overlay = Gtk.Overlay()

        # Hour grid (background)
        self._hour_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        for hour in range(START_HOUR, END_HOUR):
            hour_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            hour_box.set_size_request(-1, HOUR_HEIGHT)
            hour_box.add_css_class("hour-slot")

            # Click handler for each hour
            click = Gtk.GestureClick()
            click.connect("pressed", self._on_hour_slot_clicked, hour)
            hour_box.add_controller(click)

            self._hour_grid.append(hour_box)

        self._overlay.set_child(self._hour_grid)

        # Events container (overlay)
        self._events_container = Gtk.Fixed()
        self._overlay.add_overlay(self._events_container)

        scroll.set_child(self._overlay)
        self.append(scroll)

    def _on_hour_slot_clicked(
        self,
        gesture: Gtk.GestureClick,
        n_press: int,
        x: float,
        y: float,
        hour: int,
    ) -> None:
        """Handle click on hour slot."""
        if self._on_hour_clicked and n_press == 2:
            self._on_hour_clicked(self.column_date, hour)

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        """Set events for this day column."""
        self._events = events

        # Clear existing event blocks
        for block in self._event_blocks:
            self._events_container.remove(block)
        self._event_blocks.clear()

        # Position event blocks
        for event in events:
            if event.get("is_all_day"):
                continue  # All-day events handled separately

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
            block = EventBlock(event, self._on_event_clicked)
            block.set_size_request(120, int(height))

            self._events_container.put(block, 0, top)
            self._event_blocks.append(block)


class CalendarWeekView(Gtk.Box):
    """Week view calendar widget showing 7 day columns with events.

    Displays a week view with hourly time slots and positioned event blocks.
    """

    def __init__(
        self,
        atlas: Any = None,
        on_date_selected: Optional[Callable[[date], None]] = None,
        on_hour_clicked: Optional[Callable[[date, int], None]] = None,
        on_event_clicked: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._atlas = atlas
        self._on_date_selected = on_date_selected
        self._on_hour_clicked = on_hour_clicked
        self._on_event_clicked = on_event_clicked

        self._current_date = date.today()
        self._week_start = self._get_week_start(self._current_date)
        self._day_columns: Dict[date, DayColumn] = {}
        self._repo = None

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-week-view")

        self._build_ui()
        self._init_repository()
        self._populate_columns()

    def _get_week_start(self, d: date) -> date:
        """Get the Monday of the week containing date d."""
        return d - timedelta(days=d.weekday())

    def _build_ui(self) -> None:
        """Build the week view UI."""
        # Navigation header
        nav_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        nav_box.set_margin_top(8)
        nav_box.set_margin_bottom(8)
        nav_box.set_margin_start(8)
        nav_box.set_margin_end(8)

        # Previous week button
        prev_btn = Gtk.Button()
        prev_btn.set_icon_name("go-previous-symbolic")
        prev_btn.add_css_class("flat")
        prev_btn.connect("clicked", lambda _: self._navigate(-1))
        nav_box.append(prev_btn)

        # Week label
        self._header_label = Gtk.Label()
        self._header_label.set_hexpand(True)
        self._header_label.add_css_class("title-2")
        nav_box.append(self._header_label)

        # Today button
        today_btn = Gtk.Button(label="Today")
        today_btn.add_css_class("flat")
        today_btn.connect("clicked", lambda _: self.go_to_today())
        nav_box.append(today_btn)

        # Next week button
        next_btn = Gtk.Button()
        next_btn.set_icon_name("go-next-symbolic")
        next_btn.add_css_class("flat")
        next_btn.connect("clicked", lambda _: self._navigate(1))
        nav_box.append(next_btn)

        self.append(nav_box)

        # Main content area
        content = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        content.set_hexpand(True)
        content.set_vexpand(True)

        # Time labels column
        time_col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        time_col.set_size_request(50, -1)
        time_col.add_css_class("time-labels")

        # Header spacer
        spacer = Gtk.Box()
        spacer.set_size_request(-1, 52)  # Match day column header
        time_col.append(spacer)

        # Hour labels in scrollable area
        time_scroll = Gtk.ScrolledWindow()
        time_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        time_scroll.set_vexpand(True)

        time_grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        for hour in range(START_HOUR, END_HOUR):
            hour_label = Gtk.Label(label=f"{hour:02d}:00")
            hour_label.set_size_request(-1, HOUR_HEIGHT)
            hour_label.set_valign(Gtk.Align.START)
            hour_label.add_css_class("dim-label")
            hour_label.add_css_class("caption")
            time_grid.append(hour_label)

        time_scroll.set_child(time_grid)
        time_col.append(time_scroll)
        content.append(time_col)

        # Days container with scroll sync
        self._days_scroll = Gtk.ScrolledWindow()
        self._days_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._days_scroll.set_hexpand(True)
        self._days_scroll.set_vexpand(True)

        # Sync scrolling between time labels and days
        adj = self._days_scroll.get_vadjustment()
        time_scroll.set_vadjustment(adj)

        self._days_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=1)
        self._days_box.set_homogeneous(True)
        self._days_scroll.set_child(self._days_box)

        content.append(self._days_scroll)
        self.append(content)

    def _init_repository(self) -> None:
        """Initialize the calendar repository."""
        # Service is accessed directly from ATLAS when needed
        pass

    def _load_events(self) -> None:
        """Load events for the visible week."""
        async def fetch_and_display():
            if not self._atlas:
                return
                
            try:
                service = self._atlas.calendar_service
                actor = self._atlas.get_current_actor()
                
                start_dt = datetime.combine(self._week_start, time.min)
                end_dt = datetime.combine(self._week_start + timedelta(days=7), time.min)

                result = await service.get_events_in_range(
                    actor,
                    start_time=start_dt,
                    end_time=end_dt
                )
                
                if not result.is_success:
                    logger.warning(f"Failed to load events: {result.error}")
                    return
                
                events = result.data

                # Group by date
                events_by_date: Dict[date, List[Any]] = {}
                for event in events:
                    start_time = event.start_time
                    if isinstance(start_time, datetime):
                        event_date = start_time.date()
                    else:
                        continue

                    if event_date not in events_by_date:
                        events_by_date[event_date] = []

                    # Convert to dict format for compatibility
                    event_dict = {
                        "id": str(event.event_id),
                        "title": event.title,
                        "start_time": event.start_time,
                        "end_time": event.end_time,
                        "is_all_day": event.all_day,
                        "category_id": str(event.category_id) if event.category_id else None,
                        "color": "#4285F4"  # Default color, category colors handled elsewhere
                    }
                    events_by_date[event_date].append(event_dict)

                # Update columns on main thread
                def update_columns():
                    for col_date, column in self._day_columns.items():
                        events = events_by_date.get(col_date, [])
                        column.set_events(events)
                
                GLib.idle_add(update_columns)
                
            except Exception as e:
                logger.error(f"Error loading events: {e}")
        
        asyncio.create_task(fetch_and_display())

    def _populate_columns(self) -> None:
        """Populate the week columns."""
        # Update header
        week_end = self._week_start + timedelta(days=6)
        if self._week_start.month == week_end.month:
            header = f"{self._week_start.strftime('%B %d')} - {week_end.day}, {week_end.year}"
        else:
            header = f"{self._week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"
        self._header_label.set_label(header)

        # Clear existing columns
        while child := self._days_box.get_first_child():
            self._days_box.remove(child)
        self._day_columns.clear()

        # Create day columns
        for i in range(7):
            column_date = self._week_start + timedelta(days=i)
            is_today = column_date == self._current_date

            column = DayColumn(
                column_date,
                is_today,
                self._on_hour_clicked,
                self._on_event_clicked,
            )
            self._days_box.append(column)
            self._day_columns[column_date] = column

        # Load events
        self._load_events()

    def _navigate(self, delta_weeks: int) -> None:
        """Navigate by delta weeks."""
        self._week_start += timedelta(weeks=delta_weeks)
        self._populate_columns()

    def go_to_today(self) -> None:
        """Navigate to current week."""
        self._current_date = date.today()
        self._week_start = self._get_week_start(self._current_date)
        self._populate_columns()

    def go_to_date(self, target_date: date) -> None:
        """Navigate to week containing target date."""
        self._week_start = self._get_week_start(target_date)
        self._populate_columns()

    def refresh(self) -> None:
        """Refresh events from repository."""
        self._load_events()

    @property
    def week_start(self) -> date:
        """Get the start date of the current week."""
        return self._week_start
