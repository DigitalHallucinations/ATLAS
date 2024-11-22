# UI/sidebar.py

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib

import os

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

        # Set default size
        self.set_default_size(50, 600)  # Adjust height as needed
        self.set_decorated(False)

        self.get_style_context().add_class("sidebar")
        apply_css()

        # Create main box
        self.box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.set_child(self.box)  # In GTK 4, use set_child instead of add()

        # Add icons
        self.create_icon("Icons/providers.png", self.show_provider_menu, 42)
        self.create_icon("Icons/history.png", self.handle_history_button, 42)
        self.create_icon("Icons/chat.png", self.show_chat_page, 42)
        self.create_icon("Icons/agent.png", self.show_persona_menu, 42)

        # Spacer
        self.box.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Add bottom icons
        self.create_icon("Icons/settings.png", self.show_settings_page, 42)
        self.create_icon("Icons/power_button.png", self.close_application, 42)

    def create_icon(self, icon_path, callback, icon_size):
        try:
            # Construct the icon path relative to this file
            full_icon_path = os.path.join(os.path.dirname(__file__), "..", icon_path)
            full_icon_path = os.path.abspath(full_icon_path)
            # Load the icon using Gdk.Texture
            texture = Gdk.Texture.new_from_filename(full_icon_path)
            # Create Gtk.Picture for the icon
            icon = Gtk.Picture.new_for_paintable(texture)
            icon.set_size_request(icon_size, icon_size)
            icon.set_content_fit(Gtk.ContentFit.CONTAIN)
        except GLib.Error as e:
            print(f"Error loading icon {icon_path}: {e}")
            # Load a fallback icon
            fallback_icon_path = os.path.join(os.path.dirname(__file__), "..", "Icons/default.png")
            try:
                texture = Gdk.Texture.new_from_filename(fallback_icon_path)
                icon = Gtk.Picture.new_for_paintable(texture)
                icon.set_size_request(icon_size, icon_size)
                icon.set_content_fit(Gtk.ContentFit.CONTAIN)
            except GLib.Error as e:
                print(f"Error loading fallback icon: {e}")
                icon = Gtk.Image.new_from_icon_name("image-missing")

        icon.get_style_context().add_class("icon")

        # In GTK 4, use Gtk.GestureClick for click events
        gesture = Gtk.GestureClick()
        gesture.connect("pressed", lambda gesture, n_press, x, y: callback())
        icon.add_controller(gesture)

        self.box.append(icon)

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
            chat_page.present()
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
            settings_window.present()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def close_application(self):
        print("Closing application")
        app = self.get_application()
        if app:
            app.quit()

    def show_error_dialog(self, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Initialization Error",
        )
        dialog.format_secondary_text(message)
        dialog.show()
