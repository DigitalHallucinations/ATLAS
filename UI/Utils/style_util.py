# UI/Utils/style_util.py

from gi.repository import Gtk, Gdk

def apply_css():
    """
    Applies CSS styling from an external file to the GTK application.
    """
    css_provider = Gtk.CssProvider()
    css_file_path = "UI/Utils/style.css"  # Path to your external CSS file

    try:
        css_provider.load_from_path(css_file_path)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        print(f"Successfully loaded CSS from {css_file_path}")
    except Exception as e:
        print(f"Failed to load external CSS file '{css_file_path}': {e}")