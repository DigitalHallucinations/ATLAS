"""Color chooser widget with preset palette and custom picker.

Provides a GTK 4 widget for selecting colors from a preset palette
or using a custom color picker.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk

logger = logging.getLogger(__name__)

# Default color palette matching the calendar store specification
DEFAULT_PALETTE = [
    "#4285F4",  # Blue
    "#34A853",  # Green
    "#EA4335",  # Red
    "#FBBC05",  # Yellow
    "#FF6D01",  # Orange
    "#9334E6",  # Purple
    "#E91E63",  # Pink
    "#00BCD4",  # Cyan
    "#795548",  # Brown
    "#607D8B",  # Gray
    "#009688",  # Teal
    "#673AB7",  # Indigo
]


class ColorSwatch(Gtk.Button):
    """A clickable color swatch button."""

    def __init__(self, color: str, size: int = 32) -> None:
        super().__init__()
        self._color = color
        self._size = size

        self.set_size_request(size, size)
        self.add_css_class("flat")
        self.add_css_class("color-swatch")

        # Apply color via CSS
        self._apply_color_style()

    def _apply_color_style(self) -> None:
        """Apply the color as background via CSS."""
        css_provider = Gtk.CssProvider()
        css = f"""
        .color-swatch {{
            background-color: {self._color};
            border-radius: 4px;
            border: 2px solid transparent;
            min-width: {self._size}px;
            min-height: {self._size}px;
        }}
        .color-swatch:hover {{
            border-color: alpha(currentColor, 0.3);
        }}
        .color-swatch.selected {{
            border-color: @accent_color;
            border-width: 3px;
        }}
        """
        css_provider.load_from_data(css.encode())
        self.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    @property
    def color(self) -> str:
        """Get the swatch color."""
        return self._color

    def set_selected(self, selected: bool) -> None:
        """Set the selected state."""
        if selected:
            self.add_css_class("selected")
        else:
            self.remove_css_class("selected")


class ColorChooser(Gtk.Box):
    """Widget for choosing colors from palette or custom picker.

    Signals:
        color-changed: Emitted when the selected color changes.
                      Callback signature: (widget, color_hex: str)
    """

    def __init__(
        self,
        palette: Optional[list[str]] = None,
        initial_color: str = "#4285F4",
        show_custom: bool = True,
        columns: int = 6,
    ) -> None:
        """Initialize the color chooser.

        Args:
            palette: List of hex color codes for the palette. Defaults to DEFAULT_PALETTE.
            initial_color: Initial selected color.
            show_custom: Whether to show the custom color picker button.
            columns: Number of columns in the palette grid.
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)

        self._palette = palette or DEFAULT_PALETTE
        self._selected_color = initial_color.upper()
        self._show_custom = show_custom
        self._columns = columns
        self._swatches: dict[str, ColorSwatch] = {}
        self._on_color_changed: Optional[Callable[[str], None]] = None

        self._build_ui()
        self._update_selection()

    def _build_ui(self) -> None:
        """Build the color chooser UI."""
        # Palette grid
        self._palette_grid = Gtk.FlowBox()
        self._palette_grid.set_selection_mode(Gtk.SelectionMode.NONE)
        self._palette_grid.set_max_children_per_line(self._columns)
        self._palette_grid.set_min_children_per_line(self._columns)
        self._palette_grid.set_row_spacing(4)
        self._palette_grid.set_column_spacing(4)
        self._palette_grid.set_homogeneous(True)

        for color in self._palette:
            swatch = ColorSwatch(color.upper())
            swatch.connect("clicked", self._on_swatch_clicked)
            self._swatches[color.upper()] = swatch
            self._palette_grid.append(swatch)

        self.append(self._palette_grid)

        if self._show_custom:
            # Separator
            sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
            sep.set_margin_top(4)
            sep.set_margin_bottom(4)
            self.append(sep)

            # Custom color row
            custom_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

            custom_label = Gtk.Label(label="Custom:")
            custom_label.add_css_class("dim-label")
            custom_row.append(custom_label)

            # Color entry
            self._color_entry = Gtk.Entry()
            self._color_entry.set_placeholder_text("#RRGGBB")
            self._color_entry.set_max_length(7)
            self._color_entry.set_width_chars(9)
            self._color_entry.set_text(self._selected_color)
            self._color_entry.connect("changed", self._on_entry_changed)
            custom_row.append(self._color_entry)

            # Color button (opens GTK color chooser)
            self._color_button = Gtk.ColorButton()
            self._set_color_button_color(self._selected_color)
            self._color_button.set_use_alpha(False)
            self._color_button.connect("color-set", self._on_color_button_set)
            custom_row.append(self._color_button)

            # Preview swatch
            self._preview_swatch = Gtk.Box()
            self._preview_swatch.set_size_request(32, 32)
            self._preview_swatch.add_css_class("color-preview")
            self._update_preview()
            custom_row.append(self._preview_swatch)

            self.append(custom_row)

    def _set_color_button_color(self, hex_color: str) -> None:
        """Set the GTK ColorButton's color from hex string."""
        rgba = Gdk.RGBA()
        if rgba.parse(hex_color):
            self._color_button.set_rgba(rgba)

    def _update_preview(self) -> None:
        """Update the preview swatch color."""
        css_provider = Gtk.CssProvider()
        css = f"""
        .color-preview {{
            background-color: {self._selected_color};
            border-radius: 4px;
            border: 1px solid alpha(currentColor, 0.2);
        }}
        """
        css_provider.load_from_data(css.encode())
        self._preview_swatch.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _update_selection(self) -> None:
        """Update the visual selection state."""
        for color, swatch in self._swatches.items():
            swatch.set_selected(color == self._selected_color)

    def _on_swatch_clicked(self, swatch: ColorSwatch) -> None:
        """Handle palette swatch click."""
        self._set_color(swatch.color)

    def _on_entry_changed(self, entry: Gtk.Entry) -> None:
        """Handle color entry text change."""
        text = entry.get_text().strip().upper()
        if len(text) == 7 and text.startswith("#"):
            # Validate hex format
            try:
                int(text[1:], 16)
                self._set_color(text, update_entry=False)
            except ValueError:
                pass

    def _on_color_button_set(self, button: Gtk.ColorButton) -> None:
        """Handle GTK color chooser selection."""
        rgba = button.get_rgba()
        hex_color = "#{:02X}{:02X}{:02X}".format(
            int(rgba.red * 255),
            int(rgba.green * 255),
            int(rgba.blue * 255),
        )
        self._set_color(hex_color)

    def _set_color(self, color: str, update_entry: bool = True) -> None:
        """Set the selected color and emit signal."""
        color = color.upper()
        if color == self._selected_color:
            return

        self._selected_color = color
        self._update_selection()

        if self._show_custom:
            if update_entry:
                self._color_entry.set_text(color)
            self._set_color_button_color(color)
            self._update_preview()

        # Emit callback
        if self._on_color_changed:
            self._on_color_changed(color)

    def get_color(self) -> str:
        """Get the currently selected color."""
        return self._selected_color

    def set_color(self, color: str) -> None:
        """Set the selected color."""
        self._set_color(color.upper())

    def connect_color_changed(self, callback: Callable[[str], None]) -> None:
        """Connect a callback for color changes.

        Args:
            callback: Function called with the new hex color string.
        """
        self._on_color_changed = callback


__all__ = ["ColorChooser", "ColorSwatch", "DEFAULT_PALETTE"]
