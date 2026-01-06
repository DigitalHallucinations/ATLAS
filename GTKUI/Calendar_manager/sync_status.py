"""Sync status panel for calendar management."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

logger = logging.getLogger(__name__)


class SyncStatusPanel(Gtk.Box):
    """Panel showing synchronization status for all calendars."""

    def __init__(self, atlas: Any, parent_window: Gtk.Window) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas
        self._parent = parent_window
        self._status_data: Dict[str, dict] = {}
        self._row_widgets: Dict[str, Dict[str, Gtk.Widget]] = {}

        self.set_margin_top(12)
        self.set_margin_bottom(12)
        self.set_margin_start(12)
        self.set_margin_end(12)

        self._build_ui()
        self._load_status()

    def _build_ui(self) -> None:
        """Build the sync status UI."""
        # Header
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header_box.set_margin_bottom(8)

        title = Gtk.Label(label="Synchronization Status")
        title.add_css_class("title-2")
        title.set_xalign(0.0)
        title.set_hexpand(True)
        header_box.append(title)

        # Sync all button
        sync_all_btn = Gtk.Button()
        sync_all_btn.set_icon_name("emblem-synchronizing-symbolic")
        sync_all_btn.set_tooltip_text("Sync All Calendars")
        sync_all_btn.connect("clicked", self._on_sync_all)
        header_box.append(sync_all_btn)

        # Refresh button
        refresh_btn = Gtk.Button()
        refresh_btn.set_icon_name("view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Refresh Status")
        refresh_btn.connect("clicked", lambda b: self._load_status())
        header_box.append(refresh_btn)

        self.append(header_box)

        # Separator
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Status summary
        self._summary_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=24)
        self._summary_box.set_margin_top(8)
        self._summary_box.set_margin_bottom(8)
        self.append(self._summary_box)

        # Separator
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Scrolled list of calendars with sync info
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._list_box = Gtk.ListBox()
        self._list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("boxed-list")
        scroll.set_child(self._list_box)
        self.append(scroll)

        # Error log section
        error_header = Gtk.Label(label="Recent Errors")
        error_header.add_css_class("title-3")
        error_header.set_xalign(0.0)
        error_header.set_margin_top(12)
        self.append(error_header)

        error_scroll = Gtk.ScrolledWindow()
        error_scroll.set_min_content_height(100)
        error_scroll.set_max_content_height(150)
        error_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._error_list = Gtk.ListBox()
        self._error_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._error_list.add_css_class("boxed-list")
        error_scroll.set_child(self._error_list)
        self.append(error_scroll)

    def _load_status(self) -> None:
        """Load sync status from registry."""
        try:
            from modules.Tools.Base_Tools.calendar import CalendarProviderRegistry

            registry = CalendarProviderRegistry.get_instance()
            providers = registry.list_providers()

            self._status_data = {}
            for name, info in providers.items():
                backend = info.get("backend")
                self._status_data[name] = {
                    "display_name": info.get("display_name", name),
                    "backend_type": info.get("type", "unknown"),
                    "write_enabled": info.get("write_enabled", False),
                    "last_sync": getattr(backend, "_last_sync", None) if backend else None,
                    "sync_error": getattr(backend, "_last_error", None) if backend else None,
                    "event_count": getattr(backend, "_event_count", 0) if backend else 0,
                }

            self._update_display()

        except ImportError:
            logger.warning("Calendar registry not available")
            self._show_empty_state("Calendar system not initialized")
        except Exception as e:
            logger.error("Failed to load sync status: %s", e)
            self._show_empty_state(f"Error loading status: {e}")

    def _update_display(self) -> None:
        """Update the display with current status data."""
        # Clear existing rows
        while True:
            row = self._list_box.get_row_at_index(0)
            if row is None:
                break
            self._list_box.remove(row)

        self._row_widgets.clear()

        # Update summary
        self._update_summary()

        # Add rows for each calendar
        for name, status in self._status_data.items():
            row = self._create_status_row(name, status)
            self._list_box.append(row)

        # Update error list
        self._update_errors()

    def _update_summary(self) -> None:
        """Update the summary statistics."""
        # Clear summary box
        while True:
            child = self._summary_box.get_first_child()
            if child is None:
                break
            self._summary_box.remove(child)

        total = len(self._status_data)
        synced = sum(1 for s in self._status_data.values() if s.get("last_sync"))
        errors = sum(1 for s in self._status_data.values() if s.get("sync_error"))
        writable = sum(1 for s in self._status_data.values() if s.get("write_enabled"))

        # Create stat boxes
        stats = [
            ("view-list-symbolic", f"{total}", "Total Calendars"),
            ("emblem-ok-symbolic", f"{synced}", "Synced"),
            ("dialog-error-symbolic", f"{errors}", "Errors"),
            ("document-edit-symbolic", f"{writable}", "Writable"),
        ]

        for icon_name, value, label in stats:
            stat_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

            icon = Gtk.Image.new_from_icon_name(icon_name)
            stat_box.append(icon)

            value_label = Gtk.Label(label=value)
            value_label.add_css_class("title-1")
            stat_box.append(value_label)

            desc_label = Gtk.Label(label=label)
            desc_label.add_css_class("dim-label")
            stat_box.append(desc_label)

            self._summary_box.append(stat_box)

    def _create_status_row(self, name: str, status: dict) -> Gtk.ListBoxRow:
        """Create a row for a calendar's sync status."""
        row = Gtk.ListBoxRow()

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        box.set_margin_start(12)
        box.set_margin_end(12)

        # Status indicator
        has_error = status.get("sync_error") is not None
        indicator = Gtk.Image()
        if has_error:
            indicator.set_from_icon_name("dialog-error-symbolic")
            indicator.add_css_class("error")
        elif status.get("last_sync"):
            indicator.set_from_icon_name("emblem-ok-symbolic")
            indicator.add_css_class("success")
        else:
            indicator.set_from_icon_name("content-loading-symbolic")
            indicator.add_css_class("dim-label")
        box.append(indicator)

        # Info section
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_hexpand(True)

        name_label = Gtk.Label(label=status.get("display_name", name))
        name_label.set_xalign(0.0)
        name_label.add_css_class("heading")
        info_box.append(name_label)

        # Last sync time
        last_sync = status.get("last_sync")
        if last_sync:
            if isinstance(last_sync, datetime):
                sync_text = last_sync.strftime("Last sync: %Y-%m-%d %H:%M")
            else:
                sync_text = f"Last sync: {last_sync}"
        else:
            sync_text = "Not synced"

        sync_label = Gtk.Label(label=sync_text)
        sync_label.set_xalign(0.0)
        sync_label.add_css_class("dim-label")
        sync_label.add_css_class("caption")
        info_box.append(sync_label)

        # Event count
        event_count = status.get("event_count", 0)
        count_label = Gtk.Label(label=f"{event_count} events")
        count_label.set_xalign(0.0)
        count_label.add_css_class("dim-label")
        count_label.add_css_class("caption")
        info_box.append(count_label)

        box.append(info_box)

        # Type badge
        backend_type = status.get("backend_type", "unknown")
        type_label = Gtk.Label(label=backend_type.upper())
        type_label.add_css_class("caption")
        type_label.add_css_class("dim-label")
        box.append(type_label)

        # Sync button
        sync_btn = Gtk.Button()
        sync_btn.set_icon_name("emblem-synchronizing-symbolic")
        sync_btn.set_tooltip_text("Sync Now")
        sync_btn.connect("clicked", self._on_sync_calendar, name)
        box.append(sync_btn)

        row.set_child(box)

        # Store widgets for updates
        self._row_widgets[name] = {
            "indicator": indicator,
            "sync_label": sync_label,
            "count_label": count_label,
            "sync_btn": sync_btn,
        }

        return row

    def _update_errors(self) -> None:
        """Update the error log."""
        # Clear existing
        while True:
            row = self._error_list.get_row_at_index(0)
            if row is None:
                break
            self._error_list.remove(row)

        # Collect errors
        errors: List[tuple] = []
        for name, status in self._status_data.items():
            error = status.get("sync_error")
            if error:
                errors.append((name, str(error)))

        if not errors:
            # Show "no errors" message
            row = Gtk.ListBoxRow()
            label = Gtk.Label(label="No sync errors")
            label.add_css_class("dim-label")
            label.set_margin_top(8)
            label.set_margin_bottom(8)
            row.set_child(label)
            self._error_list.append(row)
            return

        for name, error_msg in errors:
            row = Gtk.ListBoxRow()
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            box.set_margin_top(4)
            box.set_margin_bottom(4)
            box.set_margin_start(8)
            box.set_margin_end(8)

            icon = Gtk.Image.new_from_icon_name("dialog-error-symbolic")
            icon.add_css_class("error")
            box.append(icon)

            name_label = Gtk.Label(label=f"{name}:")
            name_label.add_css_class("heading")
            box.append(name_label)

            error_label = Gtk.Label(label=error_msg)
            error_label.set_xalign(0.0)
            error_label.set_hexpand(True)
            error_label.set_wrap(True)
            error_label.add_css_class("dim-label")
            box.append(error_label)

            row.set_child(box)
            self._error_list.append(row)

    def _show_empty_state(self, message: str) -> None:
        """Show empty state message."""
        # Clear list
        while True:
            row = self._list_box.get_row_at_index(0)
            if row is None:
                break
            self._list_box.remove(row)

        # Add message row
        row = Gtk.ListBoxRow()
        label = Gtk.Label(label=message)
        label.add_css_class("dim-label")
        label.set_margin_top(24)
        label.set_margin_bottom(24)
        row.set_child(label)
        self._list_box.append(row)

    def _on_sync_calendar(self, button: Gtk.Button, name: str) -> None:
        """Handle sync button click for a single calendar."""
        button.set_sensitive(False)
        button.set_icon_name("content-loading-symbolic")

        def do_sync():
            try:
                from modules.Tools.Base_Tools.calendar import CalendarProviderRegistry

                registry = CalendarProviderRegistry.get_instance()
                backend = registry.get_provider(name)

                if backend and hasattr(backend, "sync"):
                    backend.sync()
                    logger.info("Synced calendar: %s", name)

                GLib.idle_add(self._sync_complete, name, None)
            except Exception as e:
                logger.error("Sync failed for %s: %s", name, e)
                GLib.idle_add(self._sync_complete, name, str(e))

        import threading
        thread = threading.Thread(target=do_sync, daemon=True)
        thread.start()

    def _sync_complete(self, name: str, error: Optional[str]) -> None:
        """Handle sync completion on main thread."""
        widgets = self._row_widgets.get(name, {})
        sync_btn = widgets.get("sync_btn")

        if sync_btn:
            sync_btn.set_sensitive(True)
            sync_btn.set_icon_name("emblem-synchronizing-symbolic")

        # Update status
        if name in self._status_data:
            if error:
                self._status_data[name]["sync_error"] = error
            else:
                self._status_data[name]["last_sync"] = datetime.now()
                self._status_data[name]["sync_error"] = None

        self._update_display()

        # Notify via message bus
        if hasattr(self.ATLAS, "message_bus"):
            self.ATLAS.message_bus.publish(
                "CALENDAR_SYNC",
                {"calendar": name, "success": error is None, "error": error},
            )

    def _on_sync_all(self, button: Gtk.Button) -> None:
        """Handle sync all button click."""
        button.set_sensitive(False)

        def do_sync_all():
            try:
                from modules.Tools.Base_Tools.calendar import CalendarProviderRegistry

                registry = CalendarProviderRegistry.get_instance()
                providers = registry.list_providers()

                for name, info in providers.items():
                    backend = info.get("backend")
                    if backend and hasattr(backend, "sync"):
                        try:
                            backend.sync()
                            logger.info("Synced calendar: %s", name)
                        except Exception as e:
                            logger.error("Failed to sync %s: %s", name, e)

                GLib.idle_add(self._sync_all_complete, button)
            except Exception as e:
                logger.error("Sync all failed: %s", e)
                GLib.idle_add(self._sync_all_complete, button)

        import threading
        thread = threading.Thread(target=do_sync_all, daemon=True)
        thread.start()

    def _sync_all_complete(self, button: Gtk.Button) -> None:
        """Handle sync all completion."""
        button.set_sensitive(True)
        self._load_status()

        if hasattr(self.ATLAS, "message_bus"):
            self.ATLAS.message_bus.publish(
                "CALENDAR_SYNC", {"calendar": "all", "success": True}
            )

    def refresh(self) -> None:
        """Public method to refresh status."""
        self._load_status()
