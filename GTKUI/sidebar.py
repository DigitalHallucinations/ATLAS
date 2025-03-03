# UI/sidebar.py

"""
UI/sidebar.py

This module implements the Sidebar window that provides quick access to core
application functionality such as provider management, chat, persona selection,
settings, and application exit.

It uses absolute paths for loading icon assets, robust error handling, and integrates
with the rest of the ATLAS application.
"""

import os
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import logging

# Import other UI components and utility functions.
from GTKUI.Chat.chat_page import ChatPage
from GTKUI.Persona_manager.persona_management import PersonaManagement
from GTKUI.Provider_manager.provider_management import ProviderManagement
from GTKUI.Utils.utils import apply_css, create_box

# Configure logging for this module.
logger = logging.getLogger(__name__)


class Sidebar(Gtk.Window):
    """
    Sidebar window for accessing main application features.

    Provides clickable icons for provider settings, history, chat, persona management,
    settings, and application exit. Uses absolute paths for icon assets and applies CSS styling.
    """
    def __init__(self, atlas):
        """
        Initialize the Sidebar window.

        Args:
            atlas: The main ATLAS application instance.
        """
        super().__init__(title="Sidebar")
        self.ATLAS = atlas
        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)

        # Set the default size; increased width for improved usability.
        self.set_default_size(80, 600)
        # Remove window decorations for a modern, minimal look.
        self.set_decorated(False)
        self.get_style_context().add_class("sidebar")
        # Apply centralized CSS styling.
        apply_css()

        # Create the main container box.
        self.box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=5, margin=10)
        self.set_child(self.box)

        # Add primary icons at the top.
        self.create_icon("Icons/providers.png", self.show_provider_menu, 42)
        self.create_icon("Icons/history.png", self.handle_history_button, 42)
        self.create_icon("Icons/chat.png", self.show_chat_page, 42)
        self.create_icon("Icons/agent.png", self.show_persona_menu, 42)

        # Add a visual spacer.
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(10)
        separator.set_margin_bottom(10)
        self.box.append(separator)

        # Add secondary icons at the bottom.
        self.create_icon("Icons/settings.png", self.show_settings_page, 42)
        self.create_icon("Icons/power_button.png", self.close_application, 42)

    def create_icon(self, icon_path, callback, icon_size):
        """
        Creates and adds an icon widget to the sidebar.

        Tries to load the icon image from an absolute path. If the primary image cannot be loaded,
        attempts to load a fallback image. Registers a click gesture to invoke the given callback.

        Args:
            icon_path (str): Relative path to the icon image (from the UI folder).
            callback (function): The function to be called when the icon is clicked.
            icon_size (int): The desired width and height of the icon.
        """
        try:
            # Compute absolute path based on current file location.
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_icon_path = os.path.abspath(os.path.join(current_dir, "..", icon_path))
            texture = Gdk.Texture.new_from_filename(full_icon_path)
            icon = Gtk.Picture.new_for_paintable(texture)
            icon.set_size_request(icon_size, icon_size)
            icon.set_content_fit(Gtk.ContentFit.CONTAIN)
        except Exception as e:
            logger.error(f"Error loading icon '{icon_path}': {e}")
            # Attempt to load a fallback icon.
            fallback_icon_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Icons/default.png"))
            try:
                texture = Gdk.Texture.new_from_filename(fallback_icon_path)
                icon = Gtk.Picture.new_for_paintable(texture)
                icon.set_size_request(icon_size, icon_size)
                icon.set_content_fit(Gtk.ContentFit.CONTAIN)
            except Exception as e:
                logger.error(f"Error loading fallback icon: {e}")
                icon = Gtk.Image.new_from_icon_name("image-missing")

        icon.get_style_context().add_class("icon")
        # Use Gtk.GestureClick to register a click event.
        gesture = Gtk.GestureClick.new()
        gesture.connect("pressed", lambda gesture, n_press, x, y: callback())
        icon.add_controller(gesture)
        self.box.append(icon)

    def show_provider_menu(self):
        """
        Opens the provider selection menu if ATLAS is initialized.
        """
        if self.ATLAS.is_initialized():
            self.provider_management.show_provider_menu()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def handle_history_button(self):
        """
        Invokes the history logging functionality of ATLAS.
        """
        if self.ATLAS.is_initialized():
            self.ATLAS.log_history()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_chat_page(self):
        """
        Opens the chat page by instantiating ChatPage and storing its reference.
        """
        if self.ATLAS.is_initialized():
            chat_page = ChatPage(self.ATLAS)
            self.ATLAS.chat_page = chat_page
            chat_page.present()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_persona_menu(self):
        """
        Opens the persona management menu.
        """
        if self.ATLAS.is_initialized():
            self.persona_management.show_persona_menu()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_settings_page(self):
        """
        Opens the settings page in a new transient window.
        """
        if self.ATLAS.is_initialized():
            settings_window = Gtk.Window(title="Settings")
            settings_window.set_default_size(300, 400)
            settings_window.set_transient_for(self)
            settings_window.set_modal(True)
            # Apply custom CSS styling to the settings window.
            self.apply_css_styling(settings_window)
            vbox = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
            settings_window.set_child(vbox)
            label = Gtk.Label(label="Settings Page Content Here")
            vbox.append(label)
            settings_window.present()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def close_application(self):
        """
        Closes the application by retrieving the parent application and quitting it.
        """
        logger.info("Closing application")
        app = self.get_application()
        if app:
            app.quit()

    def show_error_dialog(self, message):
        """
        Displays an error dialog with the provided message.

        Args:
            message (str): The error message to display.
        """
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Initialization Error",
        )
        dialog.format_secondary_text(message)
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

    def apply_css_styling(self, window=None):
        """
        Applies CSS styling to the specified window (or this window if none is provided).

        Args:
            window (Gtk.Window, optional): The window to style.
        """
        target = window if window else self
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .sidebar {
                background-color: #2b2b2b;
            }
            .icon {
                margin: 5px;
                border-radius: 5px;
            }
            .icon:hover {
                background-color: #4a90d9;
            }
            .icon:active {
                background-color: #357ABD;
            }
        """)
        display = target.get_display()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
