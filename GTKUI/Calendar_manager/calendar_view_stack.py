"""GTK Calendar View Stack controller.

Provides view switching between month, week, day, and agenda views.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

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

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.add_css_class("calendar-view-stack")

        self._build_ui()

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
