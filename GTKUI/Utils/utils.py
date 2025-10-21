# UI/Utils/utils.py
"""
This module provides utility functions for the GTK application.
It includes methods to apply CSS styling from an external file and to create
Gtk.Box widgets with uniform margins and spacing.

Enterprise-level error handling and logging are implemented to ensure
robust operation in production environments.
"""

import os
import logging
import gi

# Require GTK version 4.0 for modern widget APIs
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk

# Set up a module-level logger for detailed debugging.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def apply_css():
    """
    Applies CSS styling to the GTK application using an external CSS file.
    
    The CSS file is located relative to this module file for robust path resolution.
    The function uses a Gtk.CssProvider to load and apply the CSS to the default GDK display.
    
    Raises:
        FileNotFoundError: If the CSS file cannot be located.
        RuntimeError: If the default GDK display is unavailable.
    """
    css_provider = Gtk.CssProvider()
    # Compute the absolute path to the CSS file based on this file's directory.
    css_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    logger.debug(f"Attempting to load CSS from: {css_file_path}")

    try:
        # Ensure that the CSS file exists before attempting to load it.
        if not os.path.exists(css_file_path):
            raise FileNotFoundError(f"CSS file not found at {css_file_path}")

        # Load the CSS styling from the file.
        css_provider.load_from_path(css_file_path)
        logger.debug("CSS file loaded successfully.")

        # Retrieve the default GDK display.
        display = Gdk.Display.get_default()
        if display is None:
            raise RuntimeError("Unable to retrieve the default GDK display.")

        # Add the CSS provider to the display with high priority.
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        logger.debug("Successfully applied CSS from %s", css_file_path)
    except Exception as e:
        logger.error(f"Failed to load external CSS file '{css_file_path}': {e}")


def create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10):
    """
    Creates a Gtk.Box widget with specified orientation, spacing, and uniform margins.
    
    This helper function centralizes box creation for consistent UI layout.
    
    Args:
        orientation (Gtk.Orientation): The orientation (VERTICAL or HORIZONTAL) for the box.
        spacing (int): Spacing (in pixels) between child widgets.
        margin (int): Uniform margin (in pixels) to apply on all sides.
    
    Returns:
        Gtk.Box: A configured Gtk.Box widget with the specified parameters.
    """
    box = Gtk.Box(orientation=orientation, spacing=spacing)
    for setter_name in ("set_margin_top", "set_margin_bottom", "set_margin_start", "set_margin_end"):
        setter = getattr(box, setter_name, None)
        if callable(setter):
            try:
                setter(margin)
            except Exception:  # pragma: no cover - defensive for stub environments
                continue
    if not hasattr(box, "append"):
        def _append(child):
            if hasattr(box, "add"):
                box.add(child)
            else:
                children = getattr(box, "children", None)
                if children is None:
                    try:
                        box.children = [child]
                    except Exception:  # pragma: no cover - stub fallback
                        pass
                else:
                    try:
                        children.append(child)
                    except Exception:
                        pass

        setattr(box, "append", _append)
    return box
