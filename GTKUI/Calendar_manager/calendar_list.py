"""Calendar list panel for calendar management UI."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk, Pango

logger = logging.getLogger(__name__)

# Backend type display info
BACKEND_INFO = {
    "ics": {"label": "ICS File", "icon": "document-open-symbolic"},
    "dbus": {"label": "System Calendar", "icon": "computer-symbolic"},
    "google": {"label": "Google Calendar", "icon": "applications-internet-symbolic"},
    "outlook": {"label": "Outlook/Microsoft 365", "icon": "mail-send-symbolic"},
    "caldav": {"label": "CalDAV", "icon": "network-server-symbolic"},
    "apple": {"label": "Apple Calendar", "icon": "phone-apple-iphone-symbolic"},
}


class CalendarListPanel(Gtk.Box):
    """Panel displaying the list of configured calendars."""

    def __init__(self, atlas: Any, parent_controller: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.ATLAS = atlas
        self._controller = parent_controller
        self._calendar_rows: Dict[str, Gtk.ListBoxRow] = {}

        self.set_hexpand(True)
        self.set_vexpand(True)

        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        """Build the panel UI."""
        # Scrolled window for calendar list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)

        # List box for calendars
        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("boxed-list")
        self._list_box.set_margin_start(16)
        self._list_box.set_margin_end(16)
        self._list_box.set_margin_top(12)
        self._list_box.set_margin_bottom(12)

        scrolled.set_child(self._list_box)
        self.append(scrolled)

        # Empty state placeholder
        self._empty_label = Gtk.Label(
            label="No calendars configured.\n\nClick the + button to add a calendar."
        )
        self._empty_label.set_justify(Gtk.Justification.CENTER)
        self._empty_label.add_css_class("dim-label")
        self._empty_label.set_margin_top(48)
        self._empty_label.set_visible(False)
        self.append(self._empty_label)

    def refresh(self) -> None:
        """Refresh the calendar list from configuration."""
        GLib.idle_add(self._do_refresh)

    def _do_refresh(self) -> bool:
        """Perform the refresh on the main thread."""
        # Clear existing rows
        while True:
            row = self._list_box.get_first_child()
            if row is None:
                break
            self._list_box.remove(row)
        self._calendar_rows.clear()

        # Load calendars from config
        calendars = self._get_calendars()

        if not calendars:
            self._list_box.set_visible(False)
            self._empty_label.set_visible(True)
        else:
            self._empty_label.set_visible(False)
            self._list_box.set_visible(True)

            for cal in calendars:
                row = self._create_calendar_row(cal)
                self._list_box.append(row)
                self._calendar_rows[cal["name"]] = row

        return False

    def _get_calendars(self) -> List[dict]:
        """Get list of configured calendars."""
        try:
            from core.config import ConfigManager

            manager = ConfigManager.get_instance()
            calendars = manager.config.get("calendars", {})
            sources = calendars.get("sources", {})
            default = calendars.get("default_calendar")

            result = []
            for name, config in sources.items():
                if isinstance(config, dict):
                    cal = dict(config)
                    cal["name"] = name
                    cal["is_default"] = name == default
                    result.append(cal)

            # Sort by priority then name
            result.sort(key=lambda c: (c.get("priority", 100), c.get("name", "")))
            return result

        except Exception as exc:
            logger.warning("Failed to get calendars: %s", exc)
            return []

    def _create_calendar_row(self, calendar: dict) -> Gtk.ListBoxRow:
        """Create a row widget for a calendar."""
        row = Gtk.ListBoxRow()
        row.set_activatable(False)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Color indicator
        color = calendar.get("color", "#3584e4")
        color_box = Gtk.Box()
        color_box.set_size_request(4, 40)
        color_box.add_css_class("calendar-color-indicator")
        # Apply color via inline style (GTK4 CSS)
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            f".calendar-color-indicator {{ background-color: {color}; border-radius: 2px; }}".encode()
        )
        color_box.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        box.append(color_box)

        # Icon
        backend_type = calendar.get("type", "ics")
        info = BACKEND_INFO.get(backend_type, BACKEND_INFO["ics"])
        icon = Gtk.Image.new_from_icon_name(info["icon"])
        icon.set_pixel_size(24)
        icon.add_css_class("dim-label")
        box.append(icon)

        # Name and details column
        text_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        text_box.set_hexpand(True)

        # Name row with default badge
        name_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        name_label = Gtk.Label(label=calendar.get("display_name", calendar["name"]))
        name_label.set_xalign(0.0)
        name_label.add_css_class("heading")
        name_label.set_ellipsize(Pango.EllipsizeMode.END)
        name_row.append(name_label)

        if calendar.get("is_default"):
            default_badge = Gtk.Label(label="Default")
            default_badge.add_css_class("caption")
            default_badge.add_css_class("accent")
            name_row.append(default_badge)

        text_box.append(name_row)

        # Details
        details_parts = [info["label"]]
        sync_mode = calendar.get("sync_mode", "on-demand")
        details_parts.append(sync_mode.replace("-", " ").title())
        if not calendar.get("write_enabled", True):
            details_parts.append("Read-only")

        details_label = Gtk.Label(label=" • ".join(details_parts))
        details_label.set_xalign(0.0)
        details_label.add_css_class("caption")
        details_label.add_css_class("dim-label")
        text_box.append(details_label)

        box.append(text_box)

        # Status indicator
        status_icon = Gtk.Image.new_from_icon_name("emblem-ok-symbolic")
        status_icon.add_css_class("success")
        status_icon.set_tooltip_text("Connected")
        box.append(status_icon)

        # Action buttons
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Set as default button (if not default)
        if not calendar.get("is_default"):
            default_btn = Gtk.Button()
            default_btn.set_icon_name("starred-symbolic")
            default_btn.set_tooltip_text("Set as Default")
            default_btn.add_css_class("flat")
            default_btn.connect(
                "clicked", self._on_set_default_clicked, calendar["name"]
            )
            action_box.append(default_btn)

        # Edit button
        edit_btn = Gtk.Button()
        edit_btn.set_icon_name("document-edit-symbolic")
        edit_btn.set_tooltip_text("Edit")
        edit_btn.add_css_class("flat")
        edit_btn.connect("clicked", self._on_edit_clicked, calendar["name"])
        action_box.append(edit_btn)

        # Sync button
        sync_btn = Gtk.Button()
        sync_btn.set_icon_name("emblem-synchronizing-symbolic")
        sync_btn.set_tooltip_text("Sync Now")
        sync_btn.add_css_class("flat")
        sync_btn.connect("clicked", self._on_sync_clicked, calendar["name"])
        action_box.append(sync_btn)

        # Delete button
        delete_btn = Gtk.Button()
        delete_btn.set_icon_name("user-trash-symbolic")
        delete_btn.set_tooltip_text("Remove")
        delete_btn.add_css_class("flat")
        delete_btn.add_css_class("destructive-action")
        delete_btn.connect("clicked", self._on_delete_clicked, calendar["name"])
        action_box.append(delete_btn)

        box.append(action_box)

        row.set_child(box)
        return row

    def _on_set_default_clicked(self, button: Gtk.Button, name: str) -> None:
        """Handle set as default button click."""
        try:
            from core.config import ConfigManager

            manager = ConfigManager.get_instance()
            calendars = manager.config.get("calendars", {})
            calendars["default_calendar"] = name
            manager.config["calendars"] = calendars
            manager.save_config()
            self.refresh()
            logger.info("Set default calendar: %s", name)
        except Exception as exc:
            logger.error("Failed to set default calendar: %s", exc)

    def _on_edit_clicked(self, button: Gtk.Button, name: str) -> None:
        """Handle edit button click."""
        from .calendar_dialog import CalendarDialog

        try:
            from core.config import ConfigManager

            manager = ConfigManager.get_instance()
            calendars = manager.config.get("calendars", {})
            sources = calendars.get("sources", {})
            config = sources.get(name, {})
            config["name"] = name

            dialog = CalendarDialog(
                parent=self._controller.parent_window,
                atlas=self.ATLAS,
                mode="edit",
                calendar_config=config,
            )
            dialog.connect("response", self._on_edit_dialog_response, name)
            dialog.present()

        except Exception as exc:
            logger.error("Failed to open edit dialog: %s", exc)

    def _on_edit_dialog_response(
        self, dialog: Any, response_id: int, original_name: str
    ) -> None:
        """Handle edit dialog response."""
        if response_id == Gtk.ResponseType.OK:
            config = dialog.get_calendar_config()
            if config:
                self._update_calendar(original_name, config)
        dialog.destroy()

    def _update_calendar(self, original_name: str, config: dict) -> None:
        """Update calendar configuration."""
        try:
            from core.config import ConfigManager

            manager = ConfigManager.get_instance()
            new_name = config.pop("name", original_name)
            backend_type = config.pop("type", "ics")

            calendars = manager.config.get("calendars", {})
            sources = calendars.get("sources", {})

            # Remove old entry if renamed
            if original_name != new_name and original_name in sources:
                del sources[original_name]
                # Update default if renamed
                if calendars.get("default_calendar") == original_name:
                    calendars["default_calendar"] = new_name

            sources[new_name] = {"type": backend_type, **config}
            calendars["sources"] = sources
            manager.config["calendars"] = calendars
            manager.save_config()
            self.refresh()

            logger.info("Updated calendar: %s", new_name)

        except Exception as exc:
            logger.error("Failed to update calendar: %s", exc)

    def _on_sync_clicked(self, button: Gtk.Button, name: str) -> None:
        """Handle sync button click."""
        logger.info("Syncing calendar: %s", name)
        # TODO: Trigger actual sync via calendar registry
        try:
            from core.messaging import MessageBus

            bus = MessageBus.get_instance()
            bus.publish("CALENDAR_SYNC_REQUEST", {"calendar": name})
        except Exception as exc:
            logger.debug("Failed to publish sync request: %s", exc)

    def _on_delete_clicked(self, button: Gtk.Button, name: str) -> None:
        """Handle delete button click."""
        # Show confirmation dialog
        dialog = Gtk.MessageDialog(
            transient_for=self._controller.parent_window,
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=f"Remove calendar '{name}'?",
        )
        dialog.format_secondary_text(
            "This will remove the calendar from ATLAS. "
            "The calendar data itself will not be deleted."
        )
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Remove", Gtk.ResponseType.OK)
        dialog.get_widget_for_response(Gtk.ResponseType.OK).add_css_class(
            "destructive-action"
        )
        dialog.connect("response", self._on_delete_confirmed, name)
        dialog.present()

    def _on_delete_confirmed(
        self, dialog: Gtk.MessageDialog, response_id: int, name: str
    ) -> None:
        """Handle delete confirmation response."""
        dialog.destroy()
        if response_id == Gtk.ResponseType.OK:
            self._delete_calendar(name)

    def _delete_calendar(self, name: str) -> None:
        """Delete a calendar from configuration."""
        try:
            from core.config import ConfigManager

            manager = ConfigManager.get_instance()
            calendars = manager.config.get("calendars", {})
            sources = calendars.get("sources", {})

            if name in sources:
                del sources[name]
                calendars["sources"] = sources

                # Clear default if deleted
                if calendars.get("default_calendar") == name:
                    calendars["default_calendar"] = None

                manager.config["calendars"] = calendars
                manager.save_config()
                self.refresh()
                logger.info("Removed calendar: %s", name)

        except Exception as exc:
            logger.error("Failed to delete calendar: %s", exc)
