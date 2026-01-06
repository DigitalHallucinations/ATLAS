"""GTK calendar management workspace controller."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from .calendar_list import CalendarListPanel
from .calendar_dialog import CalendarDialog
from .calendar_settings import CalendarSettingsPanel
from .calendar_view_stack import CalendarViewStack
from .category_panel import CategoryPanel
from .event_dialog import EventDialog
from .sync_status import SyncStatusPanel

logger = logging.getLogger(__name__)


class CalendarManagement:
    """Controller responsible for rendering the calendar management workspace."""

    def __init__(self, atlas: Any, parent_window: Any) -> None:
        self.ATLAS = atlas
        self.parent_window = parent_window

        self._widget: Optional[Gtk.Widget] = None
        self._view_stack: Optional[Gtk.Stack] = None
        self._view_switcher: Optional[Gtk.StackSwitcher] = None

        # View instances
        self._calendar_view: Optional[CalendarViewStack] = None
        self._calendar_list: Optional[CalendarListPanel] = None
        self._category_panel: Optional[CategoryPanel] = None
        self._sync_status: Optional[SyncStatusPanel] = None
        self._settings_panel: Optional[CalendarSettingsPanel] = None

        # MessageBus subscriptions
        self._bus_subscriptions: List[Any] = []

    def get_embeddable_widget(self) -> Gtk.Widget:
        """Return the workspace widget, creating it on first use."""
        if self._widget is None:
            self._widget = self._build_workspace()
            self._subscribe_to_bus()
        return self._widget

    def _build_workspace(self) -> Gtk.Widget:
        """Build the calendar management workspace."""
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        root.set_hexpand(True)
        root.set_vexpand(True)

        # Header with title and add button
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header.set_margin_top(12)
        header.set_margin_bottom(8)
        header.set_margin_start(16)
        header.set_margin_end(16)

        title = Gtk.Label(label="Calendar Manager")
        title.set_xalign(0.0)
        title.add_css_class("title-1")
        header.append(title)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header.append(spacer)

        # Add calendar button
        add_button = Gtk.Button()
        add_button.set_icon_name("list-add-symbolic")
        add_button.set_tooltip_text("Add Calendar Source")
        add_button.add_css_class("flat")
        add_button.connect("clicked", self._on_add_calendar_clicked)
        header.append(add_button)

        # View switcher
        self._view_stack = Gtk.Stack()
        self._view_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._view_stack.set_transition_duration(200)

        self._view_switcher = Gtk.StackSwitcher()
        self._view_switcher.set_stack(self._view_stack)
        header.append(self._view_switcher)

        root.append(header)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        root.append(separator)

        # Build views - Calendar view first (main view)
        self._calendar_view = CalendarViewStack(
            atlas=self.ATLAS,
            on_event_clicked=self._on_event_clicked,
            on_create_event=self._on_create_event,
        )
        self._view_stack.add_titled(self._calendar_view, "calendar", "Calendar")

        self._calendar_list = CalendarListPanel(self.ATLAS, self)
        self._view_stack.add_titled(self._calendar_list, "sources", "Sources")

        self._category_panel = CategoryPanel(self.ATLAS, self)
        self._view_stack.add_titled(self._category_panel, "categories", "Categories")

        self._sync_status = SyncStatusPanel(self.ATLAS, self)
        self._view_stack.add_titled(self._sync_status, "sync", "Sync")

        self._settings_panel = CalendarSettingsPanel(
            atlas=self.ATLAS,
            on_settings_changed=self._on_settings_changed,
        )
        self._view_stack.add_titled(self._settings_panel, "settings", "Settings")

        # Set icons
        self._view_stack.get_page(self._calendar_view).set_icon_name(
            "x-office-calendar-symbolic"
        )
        self._view_stack.get_page(self._calendar_list).set_icon_name(
            "folder-symbolic"
        )
        self._view_stack.get_page(self._category_panel).set_icon_name(
            "view-list-symbolic"
        )
        self._view_stack.get_page(self._sync_status).set_icon_name(
            "emblem-synchronizing-symbolic"
        )
        self._view_stack.get_page(self._settings_panel).set_icon_name(
            "preferences-system-symbolic"
        )

        self._view_stack.set_hexpand(True)
        self._view_stack.set_vexpand(True)
        root.append(self._view_stack)

        return root

    def _on_event_clicked(self, event_id: str) -> None:
        """Handle event click - open event dialog for editing."""
        dialog = EventDialog(
            parent=self.parent_window,
            atlas=self.ATLAS,
            mode="edit",
            event_id=event_id,
        )
        dialog.connect("response", self._on_event_dialog_response)
        dialog.present()

    def _on_create_event(self, event_date: date, hour: Optional[int]) -> None:
        """Handle create event request."""
        # Build initial datetime
        if hour is not None:
            start_time = datetime.combine(event_date, datetime.min.time().replace(hour=hour))
        else:
            start_time = datetime.combine(event_date, datetime.min.time().replace(hour=9))

        dialog = EventDialog(
            parent=self.parent_window,
            atlas=self.ATLAS,
            mode="add",
            initial_start=start_time,
        )
        dialog.connect("response", self._on_event_dialog_response)
        dialog.present()

    def _on_event_dialog_response(self, dialog: EventDialog, response_id: int) -> None:
        """Handle event dialog response."""
        if response_id == Gtk.ResponseType.OK:
            # Event was saved, refresh views
            self.refresh()
        dialog.destroy()

    def _subscribe_to_bus(self) -> None:
        """Subscribe to calendar-related MessageBus channels."""
        try:
            from core.messaging import MessageBus

            bus = MessageBus.get_instance()

            channels = [
                "CALENDAR_SYNC",
                "CALENDAR_CONFIG",
                "CALENDAR_ERROR",
            ]

            for channel in channels:
                try:
                    subscription = bus.subscribe(channel, self._on_bus_message)
                    self._bus_subscriptions.append(subscription)
                except Exception as exc:
                    logger.debug("Failed to subscribe to %s: %s", channel, exc)

        except ImportError:
            logger.debug("MessageBus not available for calendar subscriptions")
        except Exception as exc:
            logger.debug("Failed to subscribe to calendar channels: %s", exc)

    def _on_bus_message(self, message: Any) -> None:
        """Handle incoming MessageBus events."""
        GLib.idle_add(self._process_bus_message, message)

    def _process_bus_message(self, message: Any) -> bool:
        """Process a MessageBus message on the main thread."""
        try:
            channel = getattr(message, "channel", None)
            payload = getattr(message, "payload", {})

            if channel == "CALENDAR_SYNC":
                self._handle_sync_event(payload)
            elif channel == "CALENDAR_CONFIG":
                self._handle_config_event(payload)
            elif channel == "CALENDAR_ERROR":
                self._handle_error_event(payload)

        except Exception as exc:
            logger.warning("Error processing calendar bus message: %s", exc)

        return False

    def _handle_sync_event(self, payload: dict) -> None:
        """Handle calendar sync events."""
        if self._sync_status:
            self._sync_status.refresh()

    def _handle_config_event(self, payload: dict) -> None:
        """Handle calendar configuration changes."""
        if self._calendar_list:
            self._calendar_list.refresh()

    def _handle_error_event(self, payload: dict) -> None:
        """Handle calendar error events."""
        calendar_name = payload.get("calendar", "Unknown")
        error_msg = payload.get("error", "Unknown error")
        logger.warning("Calendar error [%s]: %s", calendar_name, error_msg)

    def _on_add_calendar_clicked(self, button: Gtk.Button) -> None:
        """Handle add calendar button click."""
        dialog = CalendarDialog(
            parent=self.parent_window,
            atlas=self.ATLAS,
            mode="add",
        )
        dialog.connect("response", self._on_add_dialog_response)
        dialog.present()

    def _on_add_dialog_response(self, dialog: CalendarDialog, response_id: int) -> None:
        """Handle add calendar dialog response."""
        if response_id == Gtk.ResponseType.OK:
            calendar_config = dialog.get_calendar_config()
            if calendar_config:
                self._add_calendar(calendar_config)
        dialog.destroy()

    def _add_calendar(self, config: dict) -> None:
        """Add a new calendar to configuration."""
        try:
            from core.config import ConfigManager

            manager = ConfigManager.get_instance()
            name = config.pop("name", "unnamed")
            backend_type = config.pop("type", "ics")

            # Add to configuration
            calendars = manager.config.get("calendars", {})
            sources = calendars.get("sources", {})
            sources[name] = {"type": backend_type, **config}
            calendars["sources"] = sources
            manager.config["calendars"] = calendars

            # Save configuration
            manager.save_config()

            # Refresh the list
            if self._calendar_list:
                self._calendar_list.refresh()

            logger.info("Added calendar: %s (%s)", name, backend_type)

        except Exception as exc:
            logger.error("Failed to add calendar: %s", exc)

    def refresh(self) -> None:
        """Refresh all views."""
        if self._calendar_view:
            self._calendar_view.refresh()
        if self._calendar_list:
            self._calendar_list.refresh()
        if self._category_panel:
            self._category_panel.refresh()
        if self._sync_status:
            self._sync_status.refresh()

    def _on_settings_changed(self) -> None:
        """Handle settings changes - refresh views to apply new settings."""
        logger.debug("Calendar settings changed, refreshing views")
        self.refresh()

    def cleanup(self) -> None:
        """Clean up resources when workspace is closed."""
        try:
            from core.messaging import MessageBus

            bus = MessageBus.get_instance()
            for subscription in self._bus_subscriptions:
                try:
                    bus.unsubscribe(subscription)
                except Exception:
                    pass
        except Exception:
            pass
        self._bus_subscriptions.clear()
