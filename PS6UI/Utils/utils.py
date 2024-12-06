# UI/Utils/utils.py


import gi
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk

def apply_css():
    """
    Applies CSS styling from an external file to the GTK 4 application.
    """
    css_provider = Gtk.CssProvider()
    css_file_path = "UI/Utils/style.css"  

    try:
        css_provider.load_from_path(css_file_path)
        display = Gdk.Display.get_default()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        print(f"Successfully loaded CSS from {css_file_path}")
    except Exception as e:
        print(f"Failed to load external CSS file '{css_file_path}': {e}")


def create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10):
    """
    Creates a Gtk.Box with specified orientation, spacing, and uniform margins.

    Args:
        orientation (Gtk.Orientation): Orientation of the box (VERTICAL or HORIZONTAL).
        spacing (int): Spacing between child widgets.
        margin (int): Uniform margin to apply on all sides.

    Returns:
        Gtk.Box: The configured Gtk.Box instance.
    """
    box = Gtk.Box(orientation=orientation, spacing=spacing)
    box.set_margin_top(margin)
    box.set_margin_bottom(margin)
    box.set_margin_start(margin)
    box.set_margin_end(margin)
    return box