# UI/Provider_manager/provider_management.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

class ProviderManagement:
    def __init__(self, ATLAS, parent_window):
        self.ATLAS = ATLAS
        self.parent_window = parent_window
        self.provider_window = None

    def show_provider_menu(self):
        self.provider_window = Gtk.Window(title="Select Provider")
        self.provider_window.set_default_size(150, 400)
        self.provider_window.set_keep_above(True)

        self.position_window_next_to_sidebar(self.provider_window, 150)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.provider_window.add(box)

        provider_names = self.ATLAS.get_available_providers()

        for provider_name in provider_names:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            label = Gtk.Label(label=provider_name)
            label.set_xalign(0.0)
            label.set_yalign(0.5)

            label_event_box = Gtk.EventBox()
            label_event_box.add(label)
            label_event_box.connect(
                "button-press-event",
                lambda widget, event, provider_name=provider_name: self.select_provider(provider_name)
            )

            settings_icon_path = "Icons/settings.png"
            settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            settings_icon = Gtk.Image.new_from_pixbuf(settings_pixbuf)

            settings_event_box = Gtk.EventBox()
            settings_event_box.add(settings_icon)
            settings_event_box.set_margin_start(20)

            settings_event_box.connect(
                "button-press-event",
                lambda widget, event, provider_name=provider_name: self.open_provider_settings(provider_name)
            )

            hbox.pack_start(label_event_box, True, True, 0)
            hbox.pack_end(settings_event_box, False, False, 0)

            box.pack_start(hbox, False, False, 0)

        self.provider_window.show_all()

    def select_provider(self, provider):
        self.ATLAS.set_current_provider(provider)
        print(f"Provider '{provider}' selected.")

    def open_provider_settings(self, provider_name):
        if self.provider_window:
            self.provider_window.close()

        # Here you would load the specific provider settings
        self.show_provider_settings(provider_name)

    def show_provider_settings(self, provider_name):
        settings_window = Gtk.Window(title=f"Settings for {provider_name}")
        settings_window.set_default_size(400, 300)
        settings_window.set_keep_above(True)

        self.apply_css_styling()

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        settings_window.add(main_vbox)

        # Example settings (customize based on actual provider settings)
        provider_label = Gtk.Label(label="Provider:")
        provider_value = Gtk.Label(label=provider_name)
        
        main_vbox.pack_start(provider_label, False, False, 0)
        main_vbox.pack_start(provider_value, False, False, 0)

        # Add any other specific settings for the provider here...

        settings_window.show_all()
        self.position_window_next_to_sidebar(settings_window, 400)

    def position_window_next_to_sidebar(self, window, window_width):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        screen_width = monitor_geometry.width

        window_height = window.get_preferred_height()[1]

        window_x = screen_width - 50 - 10 - window_width
        window.set_default_size(window_width, window_height if window_height < monitor_geometry.height else monitor_geometry.height - 50)

        window.move(window_x, 0)

    def apply_css_styling(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            * { background-color: #2b2b2b; color: white; }
            label { margin: 5px; }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
