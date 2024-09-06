# UI/side_bar.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
from ATLAS.ATLAS import ATLAS

class Sidebar(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Sidebar")
        
        # ATLAS instance 
        self.ATLAS = ATLAS()

        # Set window properties to span the full screen height
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        monitor_height = monitor_geometry.height

        self.set_default_size(40, monitor_height)
        self.set_decorated(False)
        self.set_keep_above(False)
        self.stick()

        # Set up CSS for styling
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

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.add(self.box)

        top_spacer = Gtk.Box()
        self.box.pack_start(top_spacer, False, False, 5)

        # Replace buttons with event boxes containing icons
        self.create_icon("Icons/providers.png", self.show_providers_menu, 32)
        self.create_icon("Icons/history.png", self.handle_history_button, 32)
        self.create_icon("Icons/chat.png", self.show_chat_page, 32)
        self.create_icon("Icons/agent.png", self.show_persona_menu, 32)

        self.box.pack_start(Gtk.Box(), True, True, 0)
        self.create_icon("Icons/settings.png", self.show_settings_page, 32)
        self.create_icon("Icons/power_button.png", self.close_application, 32)

        bottom_spacer = Gtk.Box()
        self.box.pack_start(bottom_spacer, False, False, 10)

        self.position_sidebar()

    def create_icon(self, icon_path, callback, icon_size):
        # Create an event box to capture click events
        event_box = Gtk.EventBox()

        # Load the icon image
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, icon_size, icon_size, True)
        icon = Gtk.Image.new_from_pixbuf(pixbuf)

        # Add the image to the event box
        event_box.add(icon)

        # Connect the click event to the event box
        event_box.connect("button-press-event", lambda widget, event: callback(widget))

        # Pack the event box into the vertical box
        self.box.pack_start(event_box, False, False, 0)

    def position_sidebar(self):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        geometry = monitor.get_geometry()
        self.move(geometry.width - self.get_size().width, 0)

    def position_window_next_to_sidebar(self, window, window_width):
        """Helper function to place a window next to the sidebar."""
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        screen_width = monitor_geometry.width
        
        # Calculate the X position to place the window next to the sidebar
        window_x = screen_width - 40 - 10 - window_width  # 40px sidebar, 10px gap
        window.move(window_x, 0)  # Y = 0 (top of the screen)

    def show_providers_menu(self, widget):
        print("Providers menu clicked")

    def handle_history_button(self, widget):
        self.ATLAS.log_history()

    def show_chat_page(self, widget):
        print("Chat page clicked")

    def show_persona_menu(self, widget):
        # Create persona window
        persona_window = Gtk.Window(title="Select Persona")
        persona_window.set_default_size(150, 600)
        persona_window.set_keep_above(True)

        # Use helper method to position the window
        self.position_window_next_to_sidebar(persona_window, 150)

        # Create box to hold persona labels
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        persona_window.add(box)

        persona_names = self.ATLAS.get_persona_names()

        for persona in persona_names:
            # Create a horizontal box to hold the label and the settings icon
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            # Create and configure the persona label
            label = Gtk.Label(label=persona)
            label.set_xalign(0.0)  # Align the label to the left
            label.set_yalign(0.5)

            # Add a click event to load the persona when the label is clicked
            label_event_box = Gtk.EventBox()
            label_event_box.add(label)
            label_event_box.connect("button-press-event", lambda widget, event, persona=persona: self.ATLAS.load_persona(persona))

            # Create the settings icon
            settings_icon_path = "Icons/settings.png"  # Path to your settings icon
            settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            settings_icon = Gtk.Image.new_from_pixbuf(settings_pixbuf)

            # Create an event box for the settings icon
            settings_event_box = Gtk.EventBox()
            settings_event_box.add(settings_icon)
            settings_event_box.set_margin_start(20)  # Add 20px margin to the right of the label

            # Optionally, connect to a callback when the settings icon is clicked
            settings_event_box.connect("button-press-event", lambda widget, event, persona=persona: self.show_persona_settings(persona))

            # Add the label and settings icon to the horizontal box
            hbox.pack_start(label_event_box, True, True, 0)  # Let the label expand
            hbox.pack_end(settings_event_box, False, False, 0)  # Pack the icon to the right

            # Add the horizontal box to the main box
            box.pack_start(hbox, False, False, 0)

        persona_window.show_all()

    def show_persona_settings(self, persona):
        print(f"Settings for {persona}")


    def show_settings_page(self, widget):
        # Create settings window
        settings_window = Gtk.Window(title="Settings")
        settings_window.set_default_size(300, 400)
        settings_window.set_keep_above(True)
        
        # Use helper method to position the window
        self.position_window_next_to_sidebar(settings_window, 300)
        
        settings_window.show_all()

    def close_application(self, widget):
        print("Closing application")
        Gtk.main_quit()
