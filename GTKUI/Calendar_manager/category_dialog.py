"""Category dialog for creating and editing calendar categories.

Provides a GTK 4 dialog for adding or editing calendar categories
with color picker, icon selection, and visibility settings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # type: ignore[import-untyped]

from .color_chooser import ColorChooser, DEFAULT_PALETTE

logger = logging.getLogger(__name__)

# Common emoji icons for calendar categories
CATEGORY_ICONS = [
    "💼",  # Work
    "👤",  # Personal
    "❤️",  # Health
    "👨‍👩‍👧",  # Family
    "🎉",  # Holidays/Celebration
    "🎂",  # Birthdays
    "🏃",  # Sports/Fitness
    "🎮",  # Gaming
    "📚",  # Education
    "✈️",  # Travel
    "💰",  # Finances
    "🎵",  # Music
    "🍽️",  # Food/Dining
    "🎬",  # Entertainment
    "⭐",  # Important
    "📋",  # Tasks
]


class CategoryDialog(Gtk.Dialog):
    """Dialog for adding or editing a calendar category."""

    def __init__(
        self,
        parent: Gtk.Window,
        atlas: Any,
        mode: str = "add",
        category: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the category dialog.

        Args:
            parent: Parent window for the dialog.
            atlas: ATLAS instance for accessing services.
            mode: Either "add" or "edit".
            category: Existing category data when editing.
        """
        title = "Add Category" if mode == "add" else "Edit Category"
        super().__init__(
            title=title,
            transient_for=parent,
            modal=True,
        )
        self.ATLAS = atlas
        self._mode = mode
        self._category = category or {}

        self.set_default_size(450, -1)
        self.add_button("Cancel", Gtk.ResponseType.CANCEL)

        action_button = self.add_button(
            "Add" if mode == "add" else "Save", Gtk.ResponseType.OK
        )
        action_button.add_css_class("suggested-action")
        self._action_button = action_button

        self._build_form()
        self._validate_form()

    def _build_form(self) -> None:
        """Build the dialog form."""
        content = self.get_content_area()
        content.set_margin_top(16)
        content.set_margin_bottom(16)
        content.set_margin_start(16)
        content.set_margin_end(16)
        content.set_spacing(16)

        # Name field
        name_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        name_label = Gtk.Label(label="Name")
        name_label.set_xalign(0.0)
        name_label.add_css_class("heading")
        name_box.append(name_label)

        self._name_entry = Gtk.Entry()
        self._name_entry.set_placeholder_text("Category name")
        self._name_entry.set_text(self._category.get("name", ""))
        self._name_entry.connect("changed", self._on_name_changed)
        name_box.append(self._name_entry)
        content.append(name_box)

        # Description field
        desc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        desc_label = Gtk.Label(label="Description (optional)")
        desc_label.set_xalign(0.0)
        desc_label.add_css_class("heading")
        desc_box.append(desc_label)

        self._desc_entry = Gtk.Entry()
        self._desc_entry.set_placeholder_text("Brief description")
        self._desc_entry.set_text(self._category.get("description", "") or "")
        desc_box.append(self._desc_entry)
        content.append(desc_box)

        # Icon selection
        icon_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        icon_label = Gtk.Label(label="Icon")
        icon_label.set_xalign(0.0)
        icon_label.add_css_class("heading")
        icon_box.append(icon_label)

        self._icon_grid = Gtk.FlowBox()
        self._icon_grid.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._icon_grid.set_max_children_per_line(8)
        self._icon_grid.set_min_children_per_line(8)
        self._icon_grid.set_row_spacing(4)
        self._icon_grid.set_column_spacing(4)
        self._icon_grid.set_homogeneous(True)

        current_icon = self._category.get("icon", "👤")
        self._icon_buttons: Dict[str, Gtk.ToggleButton] = {}

        for icon in CATEGORY_ICONS:
            btn = Gtk.ToggleButton(label=icon)
            btn.set_size_request(36, 36)
            btn.add_css_class("flat")
            if icon == current_icon:
                btn.set_active(True)
            btn.connect("toggled", self._on_icon_toggled, icon)
            self._icon_buttons[icon] = btn
            self._icon_grid.append(btn)

        icon_box.append(self._icon_grid)
        self._selected_icon = current_icon
        content.append(icon_box)

        # Color selection
        color_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        color_label = Gtk.Label(label="Color")
        color_label.set_xalign(0.0)
        color_label.add_css_class("heading")
        color_box.append(color_label)

        self._color_chooser = ColorChooser(
            initial_color=self._category.get("color", "#4285F4"),
            show_custom=True,
            columns=6,
        )
        color_box.append(self._color_chooser)
        content.append(color_box)

        # Settings section
        settings_label = Gtk.Label(label="Settings")
        settings_label.set_xalign(0.0)
        settings_label.add_css_class("heading")
        settings_label.set_margin_top(8)
        content.append(settings_label)

        # Visibility toggle
        vis_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        vis_label = Gtk.Label(label="Visible in calendar")
        vis_label.set_xalign(0.0)
        vis_label.set_hexpand(True)
        vis_row.append(vis_label)

        self._visible_switch = Gtk.Switch()
        self._visible_switch.set_active(self._category.get("is_visible", True))
        self._visible_switch.set_valign(Gtk.Align.CENTER)
        vis_row.append(self._visible_switch)
        content.append(vis_row)

        # Default category toggle
        default_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        default_label = Gtk.Label(label="Default for new events")
        default_label.set_xalign(0.0)
        default_label.set_hexpand(True)
        default_row.append(default_label)

        self._default_switch = Gtk.Switch()
        self._default_switch.set_active(self._category.get("is_default", False))
        self._default_switch.set_valign(Gtk.Align.CENTER)
        default_row.append(self._default_switch)
        content.append(default_row)

        # Read-only info for built-in categories
        if self._category.get("is_builtin") and self._mode == "edit":
            info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            info_box.set_margin_top(8)

            info_icon = Gtk.Image.new_from_icon_name("dialog-information-symbolic")
            info_icon.add_css_class("dim-label")
            info_box.append(info_icon)

            info_label = Gtk.Label(
                label="This is a built-in category. Some settings cannot be changed."
            )
            info_label.add_css_class("dim-label")
            info_label.add_css_class("caption")
            info_label.set_wrap(True)
            info_box.append(info_label)

            content.append(info_box)

            # Disable name editing for built-in categories
            if self._category.get("is_readonly"):
                self._name_entry.set_sensitive(False)

    def _on_name_changed(self, entry: Gtk.Entry) -> None:
        """Handle name entry change."""
        self._validate_form()

    def _on_icon_toggled(self, button: Gtk.ToggleButton, icon: str) -> None:
        """Handle icon button toggle."""
        if button.get_active():
            # Deselect other icons
            for other_icon, other_btn in self._icon_buttons.items():
                if other_icon != icon:
                    other_btn.set_active(False)
            self._selected_icon = icon
        elif all(not btn.get_active() for btn in self._icon_buttons.values()):
            # Prevent deselecting all - reselect this one
            button.set_active(True)

    def _validate_form(self) -> None:
        """Validate form and enable/disable action button."""
        name = self._name_entry.get_text().strip()
        valid = len(name) > 0
        self._action_button.set_sensitive(valid)

    def get_category_data(self) -> Optional[Dict[str, Any]]:
        """Get the category data from the form.

        Returns:
            Dictionary with category properties, or None if invalid.
        """
        name = self._name_entry.get_text().strip()
        if not name:
            return None

        data = {
            "name": name,
            "description": self._desc_entry.get_text().strip() or None,
            "icon": self._selected_icon,
            "color": self._color_chooser.get_color(),
            "is_visible": self._visible_switch.get_active(),
            "is_default": self._default_switch.get_active(),
        }

        # Include ID for edits
        if self._mode == "edit" and "id" in self._category:
            data["id"] = self._category["id"]

        return data


__all__ = ["CategoryDialog", "CATEGORY_ICONS"]
