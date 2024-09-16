# UI/sidebar.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
from ATLAS.ATLAS import ATLAS
from UI.persona_management import PersonaManagement

class Sidebar(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Sidebar")
        
        self.ATLAS = ATLAS()
        self.persona_management = PersonaManagement(self.ATLAS, self)

        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        monitor_height = monitor_geometry.height

        self.set_default_size(50, monitor_height)
        self.set_decorated(False)
        self.set_keep_above(False)
        self.stick()

        self.apply_css_styling()

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.add(self.box)

        top_spacer = Gtk.Box()
        self.box.pack_start(top_spacer, False, False, 5)

        self.create_icon("Icons/providers.png", self.show_providers_menu, 42)
        self.create_icon("Icons/history.png", self.handle_history_button, 42)
        self.create_icon("Icons/chat.png", self.show_chat_page, 42)
        self.create_icon("Icons/agent.png", self.persona_management.show_persona_menu, 42)

        self.box.pack_start(Gtk.Box(), True, True, 0)
        self.create_icon("Icons/settings.png", self.show_settings_page, 42)
        self.create_icon("Icons/power_button.png", self.close_application, 42)

        bottom_spacer = Gtk.Box()
        self.box.pack_start(bottom_spacer, False, False, 5)

        self.position_sidebar()

    def create_icon(self, icon_path, callback, icon_size):
        event_box = Gtk.EventBox()
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, icon_size, icon_size, True)
        icon = Gtk.Image.new_from_pixbuf(pixbuf)
        event_box.add(icon)
        event_box.connect("button-press-event", lambda widget, event: callback())
        self.box.pack_start(event_box, False, False, 0)

    def position_sidebar(self):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        geometry = monitor.get_geometry()
        self.move(geometry.width - self.get_size().width, 0)

    def show_providers_menu(self):
        print("Providers menu clicked")

    def handle_history_button(self):
        self.ATLAS.log_history()

    def show_chat_page(self):
        print("Chat page clicked")

    def show_settings_page(self):
        settings_window = Gtk.Window(title="Settings")
        settings_window.set_default_size(300, 400)
        settings_window.set_keep_above(True)
        self.persona_management.position_window_next_to_sidebar(settings_window, 300)
        settings_window.show_all()

    def close_application(self):
        print("Closing application")
        Gtk.main_quit()

    def apply_css_styling(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background-color: #2b2b2b;
            }
            label {
                color: white;
                padding: 5px;
                font-size: 14px;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )