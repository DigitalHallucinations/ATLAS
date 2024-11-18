# UI/sidebar.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

from UI.Chat.chat_page import ChatPage
from UI.Persona_manager.persona_management import PersonaManagement
from UI.Provider_manager.provider_management import ProviderManagement
from UI.Utils.style_util import apply_css  

class Sidebar(Gtk.Window):
    def __init__(self, atlas):
        super().__init__(title="Sidebar")
        
        self.ATLAS = atlas
        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)

        display = Gdk.Display.get_default()
        if display:
            print(f"Display detected: {display}")
            monitor_count = display.get_n_monitors()
            print(f"Number of monitors detected: {monitor_count}")
            monitor = display.get_primary_monitor()
            if monitor:
                print(f"Primary monitor detected: {monitor}")
                monitor_geometry = monitor.get_geometry()
                print(f"Primary monitor geometry: {monitor_geometry.width}x{monitor_geometry.height}")
            else:
                print("No primary monitor detected.")
        else:
            print("No display detected.")

        monitor_geometry = monitor.get_geometry()
        monitor_height = monitor_geometry.height

        self.set_default_size(50, monitor_height)
        self.set_decorated(False)
        self.set_keep_above(False)
        self.stick()
    
        self.get_style_context().add_class("sidebar")
        apply_css()

        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.add(self.box)

        top_spacer = Gtk.Box()
        self.box.pack_start(top_spacer, False, False, 5)

        self.create_icon("Icons/providers.png", self.show_provider_menu, 42)
        self.create_icon("Icons/history.png", self.handle_history_button, 42)
        self.create_icon("Icons/chat.png", self.show_chat_page, 42)
        self.create_icon("Icons/agent.png", self.show_persona_menu, 42)

        self.box.pack_start(Gtk.Box(), True, True, 0)
        self.create_icon("Icons/settings.png", self.show_settings_page, 42)
        self.create_icon("Icons/power_button.png", self.close_application, 42)

        bottom_spacer = Gtk.Box()
        self.box.pack_start(bottom_spacer, False, False, 5)

        self.position_sidebar()

    def create_icon(self, icon_path, callback, icon_size):
        event_box = Gtk.EventBox()
        event_box.get_style_context().add_class("icon")
        
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, icon_size, icon_size, True)
        except GLib.Error as e:
            print(f"Error loading icon {icon_path}: {e}")
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("Icons/default.png", icon_size, icon_size, True)  # Fallback icon
        icon = Gtk.Image.new_from_pixbuf(pixbuf)
        event_box.add(icon)
        event_box.connect("button-press-event", lambda widget, event: callback())
        self.box.pack_start(event_box, False, False, 0)

    def position_sidebar(self):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        geometry = monitor.get_geometry()
        sidebar_width = 50  
        self.move(geometry.width - sidebar_width, 0)

    def show_provider_menu(self):
        if self.ATLAS.is_initialized():
            self.provider_management.show_provider_menu()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def handle_history_button(self):
        if self.ATLAS.is_initialized():
            self.ATLAS.log_history()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_chat_page(self):
        if self.ATLAS.is_initialized():
            chat_page = ChatPage(self.ATLAS)
            self.ATLAS.chat_page = chat_page  # Store reference to chat_page
            chat_page.show_all()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_persona_menu(self):
        if self.ATLAS.is_initialized():
            self.persona_management.show_persona_menu()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_settings_page(self):
        if self.ATLAS.is_initialized():
            settings_window = Gtk.Window(title="Settings")
            settings_window.set_default_size(300, 400)
            settings_window.set_keep_above(True)
            self.persona_management.position_window_next_to_sidebar(settings_window, 300)
            settings_window.show_all()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def close_application(self):
        print("Closing application")
        Gtk.main_quit()

    def show_error_dialog(self, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Initialization Error",
        )
        dialog.format_secondary_text(message)
        dialog.get_style_context().add_class("message-dialog")
        
        dialog.run()
        dialog.destroy()
