"""Calendar add/edit dialog."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

logger = logging.getLogger(__name__)

# Backend types with their configuration fields
BACKEND_CONFIGS = {
    "ics": {
        "label": "ICS File",
        "fields": [
            {"name": "path", "label": "File Path", "type": "file", "required": True},
            {"name": "timezone", "label": "Timezone", "type": "text", "placeholder": "America/New_York"},
        ],
    },
    "google": {
        "label": "Google Calendar",
        "fields": [
            {"name": "credentials_path", "label": "Credentials File", "type": "file", "required": True},
            {"name": "calendar_id", "label": "Calendar ID", "type": "text", "placeholder": "primary"},
            {"name": "timezone", "label": "Timezone", "type": "text", "placeholder": "America/New_York"},
        ],
    },
    "outlook": {
        "label": "Outlook / Microsoft 365",
        "fields": [
            {"name": "client_id", "label": "Application (Client) ID", "type": "text", "required": True},
            {"name": "tenant_id", "label": "Tenant ID", "type": "text", "placeholder": "common"},
            {"name": "calendar_id", "label": "Calendar ID", "type": "text", "placeholder": "default"},
            {"name": "timezone", "label": "Timezone", "type": "text", "placeholder": "America/New_York"},
        ],
    },
    "caldav": {
        "label": "CalDAV",
        "fields": [
            {"name": "url", "label": "CalDAV URL", "type": "text", "required": True},
            {"name": "username", "label": "Username", "type": "text", "required": True},
            {"name": "password_key", "label": "Password Key (in secrets)", "type": "text"},
            {"name": "calendar_id", "label": "Calendar Path", "type": "text"},
            {"name": "timezone", "label": "Timezone", "type": "text", "placeholder": "America/New_York"},
        ],
    },
    "dbus": {
        "label": "System Calendar (DBus)",
        "fields": [
            {"name": "timezone", "label": "Timezone", "type": "text", "placeholder": "America/New_York"},
        ],
    },
}

SYNC_MODES = [
    ("on-demand", "On Demand"),
    ("realtime", "Real-time"),
    ("daily", "Daily"),
    ("read-only", "Read Only"),
    ("manual", "Manual"),
]


class CalendarDialog(Gtk.Dialog):
    """Dialog for adding or editing a calendar."""

    def __init__(
        self,
        parent: Gtk.Window,
        atlas: Any,
        mode: str = "add",
        calendar_config: Optional[dict] = None,
    ) -> None:
        title = "Add Calendar" if mode == "add" else "Edit Calendar"
        super().__init__(
            title=title,
            transient_for=parent,
            modal=True,
        )
        self.ATLAS = atlas
        self._mode = mode
        self._config = calendar_config or {}
        self._field_widgets: Dict[str, Gtk.Widget] = {}

        self.set_default_size(500, -1)
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)

        action_button = self.add_button(
            "Add" if mode == "add" else "Save", Gtk.ResponseType.OK
        )
        action_button.add_css_class("suggested-action")

        self._build_form()

    def _build_form(self) -> None:
        """Build the dialog form."""
        content = self.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(12)

        # Name field
        name_row = self._create_field_row("Name", "name", "text", required=True)
        content.append(name_row)

        # Display name
        display_row = self._create_field_row(
            "Display Name", "display_name", "text", placeholder="Optional friendly name"
        )
        content.append(display_row)

        # Backend type selector
        type_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        type_label = Gtk.Label(label="Type")
        type_label.set_xalign(0.0)
        type_label.set_width_chars(15)
        type_box.append(type_label)

        self._type_combo = Gtk.ComboBoxText()
        for backend_type, info in BACKEND_CONFIGS.items():
            self._type_combo.append(backend_type, info["label"])
        self._type_combo.set_active_id(self._config.get("type", "ics"))
        self._type_combo.set_hexpand(True)
        self._type_combo.connect("changed", self._on_type_changed)
        type_box.append(self._type_combo)
        content.append(type_box)

        # Separator
        content.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Backend-specific fields container
        self._fields_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        content.append(self._fields_container)

        # Separator
        content.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Common settings
        common_label = Gtk.Label(label="Settings")
        common_label.set_xalign(0.0)
        common_label.add_css_class("heading")
        content.append(common_label)

        # Sync mode
        sync_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        sync_label = Gtk.Label(label="Sync Mode")
        sync_label.set_xalign(0.0)
        sync_label.set_width_chars(15)
        sync_box.append(sync_label)

        self._sync_combo = Gtk.ComboBoxText()
        for value, label in SYNC_MODES:
            self._sync_combo.append(value, label)
        self._sync_combo.set_active_id(self._config.get("sync_mode", "on-demand"))
        self._sync_combo.set_hexpand(True)
        sync_box.append(self._sync_combo)
        content.append(sync_box)

        # Write enabled toggle
        write_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        write_label = Gtk.Label(label="Allow Writes")
        write_label.set_xalign(0.0)
        write_label.set_width_chars(15)
        write_box.append(write_label)

        self._write_switch = Gtk.Switch()
        self._write_switch.set_active(self._config.get("write_enabled", True))
        self._write_switch.set_halign(Gtk.Align.START)
        write_box.append(self._write_switch)
        content.append(write_box)

        # Color picker
        color_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        color_label = Gtk.Label(label="Color")
        color_label.set_xalign(0.0)
        color_label.set_width_chars(15)
        color_box.append(color_label)

        self._color_button = Gtk.ColorButton()
        default_color = self._config.get("color", "#3584e4")
        try:
            from gi.repository import Gdk
            rgba = Gdk.RGBA()
            rgba.parse(default_color)
            self._color_button.set_rgba(rgba)
        except Exception:
            pass
        color_box.append(self._color_button)
        content.append(color_box)

        # Priority
        priority_row = self._create_field_row(
            "Priority", "priority", "number", placeholder="100"
        )
        content.append(priority_row)

        # Populate fields with existing config
        self._populate_values()

        # Build backend-specific fields
        self._rebuild_backend_fields()

    def _create_field_row(
        self,
        label: str,
        name: str,
        field_type: str,
        required: bool = False,
        placeholder: str = "",
    ) -> Gtk.Box:
        """Create a form field row."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        label_widget = Gtk.Label(label=label + (" *" if required else ""))
        label_widget.set_xalign(0.0)
        label_widget.set_width_chars(15)
        box.append(label_widget)

        if field_type == "file":
            entry = Gtk.Entry()
            entry.set_hexpand(True)
            entry.set_placeholder_text(placeholder or "Select file...")

            file_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            file_box.set_hexpand(True)
            file_box.append(entry)

            browse_btn = Gtk.Button()
            browse_btn.set_icon_name("folder-open-symbolic")
            browse_btn.connect("clicked", self._on_browse_clicked, entry)
            file_box.append(browse_btn)

            box.append(file_box)
            self._field_widgets[name] = entry

        elif field_type == "number":
            spin = Gtk.SpinButton()
            spin.set_range(0, 1000)
            spin.set_increments(1, 10)
            spin.set_value(int(self._config.get(name, 100)))
            spin.set_hexpand(True)
            box.append(spin)
            self._field_widgets[name] = spin

        else:  # text
            entry = Gtk.Entry()
            entry.set_hexpand(True)
            entry.set_placeholder_text(placeholder)
            box.append(entry)
            self._field_widgets[name] = entry

        return box

    def _populate_values(self) -> None:
        """Populate form fields with existing config values."""
        for name, widget in self._field_widgets.items():
            value = self._config.get(name)
            if value is not None:
                if isinstance(widget, Gtk.Entry):
                    widget.set_text(str(value))
                elif isinstance(widget, Gtk.SpinButton):
                    widget.set_value(float(value))

    def _on_type_changed(self, combo: Gtk.ComboBoxText) -> None:
        """Handle backend type change."""
        self._rebuild_backend_fields()

    def _rebuild_backend_fields(self) -> None:
        """Rebuild backend-specific fields."""
        # Clear existing fields
        while True:
            child = self._fields_container.get_first_child()
            if child is None:
                break
            self._fields_container.remove(child)

        # Get selected backend type
        backend_type = self._type_combo.get_active_id()
        if not backend_type:
            return

        config = BACKEND_CONFIGS.get(backend_type, {})
        fields = config.get("fields", [])

        for field in fields:
            name = field["name"]
            label = field["label"]
            field_type = field.get("type", "text")
            required = field.get("required", False)
            placeholder = field.get("placeholder", "")

            row = self._create_field_row(label, name, field_type, required, placeholder)
            self._fields_container.append(row)

            # Set existing value
            value = self._config.get(name)
            if value is not None and name in self._field_widgets:
                widget = self._field_widgets[name]
                if isinstance(widget, Gtk.Entry):
                    widget.set_text(str(value))

    def _on_browse_clicked(self, button: Gtk.Button, entry: Gtk.Entry) -> None:
        """Handle file browse button click."""
        dialog = Gtk.FileChooserDialog(
            title="Select File",
            transient_for=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("Open", Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self._on_file_selected, entry)
        dialog.present()

    def _on_file_selected(
        self, dialog: Gtk.FileChooserDialog, response_id: int, entry: Gtk.Entry
    ) -> None:
        """Handle file selection response."""
        if response_id == Gtk.ResponseType.ACCEPT:
            file = dialog.get_file()
            if file:
                entry.set_text(file.get_path())
        dialog.destroy()

    def get_calendar_config(self) -> Optional[dict]:
        """Get the calendar configuration from the form.

        Returns:
            Dictionary with calendar configuration, or None if validation fails.
        """
        config: Dict[str, Any] = {}

        # Name (required)
        name_widget = self._field_widgets.get("name")
        if isinstance(name_widget, Gtk.Entry):
            name = name_widget.get_text().strip()
            if not name:
                logger.warning("Calendar name is required")
                return None
            config["name"] = name

        # Display name
        display_widget = self._field_widgets.get("display_name")
        if isinstance(display_widget, Gtk.Entry):
            display = display_widget.get_text().strip()
            if display:
                config["display_name"] = display

        # Type
        config["type"] = self._type_combo.get_active_id() or "ics"

        # Backend-specific fields
        backend_config = BACKEND_CONFIGS.get(config["type"], {})
        for field in backend_config.get("fields", []):
            name = field["name"]
            required = field.get("required", False)
            widget = self._field_widgets.get(name)

            if isinstance(widget, Gtk.Entry):
                value = widget.get_text().strip()
                if value:
                    config[name] = value
                elif required:
                    logger.warning("Field '%s' is required", name)
                    return None

        # Common settings
        config["sync_mode"] = self._sync_combo.get_active_id() or "on-demand"
        config["write_enabled"] = self._write_switch.get_active()

        # Color
        rgba = self._color_button.get_rgba()
        config["color"] = f"#{int(rgba.red*255):02x}{int(rgba.green*255):02x}{int(rgba.blue*255):02x}"

        # Priority
        priority_widget = self._field_widgets.get("priority")
        if isinstance(priority_widget, Gtk.SpinButton):
            config["priority"] = int(priority_widget.get_value())

        return config
