# GTKUI/sidebar.py

"""
Sidebar Module
--------------
This module implements the Sidebar window, which provides quick access to core
application functionalities including provider management, chat, persona management,
and settings. A new speech settings icon is added to allow configuration of speech services.

Authors: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""

import os
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import logging

# Import UI components and utility functions.
from GTKUI.Chat.chat_page import ChatPage
from GTKUI.Persona_manager.persona_management import PersonaManagement
from GTKUI.Provider_manager.provider_management import ProviderManagement
from GTKUI.Utils.utils import apply_css, create_box

logger = logging.getLogger(__name__)

class Sidebar(Gtk.Window):
    def __init__(self, atlas):
        """
        Initializes the Sidebar window.

        Args:
            atlas: The main ATLAS application instance.
        """
        super().__init__(title="Sidebar")
        self.ATLAS = atlas
        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)
        self._chat_page = None

        # Set default window size and style.
        self.set_default_size(80, 600)
        self.set_decorated(False)
        self.get_style_context().add_class("sidebar")
        apply_css()

        # Create the main container box.
        self.box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=5, margin=10)
        self.set_child(self.box)

        # Add primary icons.
        self.create_icon("Icons/providers.png", self.show_provider_menu, 42, tooltip="Providers")
        self.create_icon("Icons/history.png", self.handle_history_button, 42, tooltip="History")
        self.create_icon("Icons/chat.png", self.show_chat_page, 42, tooltip="Chat")
        # Speech settings icon.
        self.create_icon("Icons/speech.png", self.show_speech_settings, 42, tooltip="Speech Settings")
        self.create_icon("Icons/agent.png", self.show_persona_menu, 42, tooltip="Personas")

        # Add a visual spacer.
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(10)
        separator.set_margin_bottom(10)
        self.box.append(separator)

        # Add secondary icons.
        self.create_icon("Icons/settings.png", self.show_settings_page, 42, tooltip="Settings")
        self.create_icon("Icons/power_button.png", self.close_application, 42, tooltip="Quit")

    def create_icon(self, icon_path, callback, icon_size, tooltip=None):
        """
        Creates and adds an icon widget to the sidebar.

        Args:
            icon_path (str): Relative path to the icon image.
            callback (function): Function to call when the icon is clicked.
            icon_size (int): Desired size for the icon.
            tooltip (str, optional): Tooltip text for the icon.
        """
        def _resolve_path(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.abspath(os.path.join(current_dir, "..", path))

        # Load the texture with a fallback
        texture = None
        try:
            full_icon_path = _resolve_path(icon_path)
            texture = Gdk.Texture.new_from_filename(full_icon_path)
        except Exception as e:
            logger.error(f"Error loading icon '{icon_path}': {e}")
            try:
                fallback_icon_path = _resolve_path("Icons/default.png")
                texture = Gdk.Texture.new_from_filename(fallback_icon_path)
            except Exception as e2:
                logger.error(f"Error loading fallback icon: {e2}")
                texture = None

        # Picture that displays the icon
        if texture:
            picture = Gtk.Picture.new_for_paintable(texture)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            picture.set_size_request(icon_size, icon_size)
        else:
            picture = Gtk.Image.new_from_icon_name("image-missing")
            picture.set_size_request(icon_size, icon_size)

        # Wrap the picture in a Button for proper accessibility & keyboard support
        button = Gtk.Button()
        button.add_css_class("icon")   # keep existing CSS hover/active behavior
        button.add_css_class("flat")   # minimal chrome in a sidebar
        button.set_child(picture)
        button.set_can_focus(True)

        if tooltip:
            button.set_tooltip_text(tooltip)

        # Accessible role is safe to set in GTK4
        button.set_accessible_role(Gtk.AccessibleRole.BUTTON)

        button.connect("clicked", lambda _btn: callback())
        self.box.append(button)

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
        Triggers the history logging functionality of ATLAS.
        """
        if self.ATLAS.is_initialized():
            self.ATLAS.log_history()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_chat_page(self):
        """
        Opens the chat page.
        """
        if self.ATLAS.is_initialized():
            if self._chat_page is None:
                chat_page = ChatPage(self.ATLAS)
                chat_page.connect("close-request", self._on_chat_page_close_request)
                self._chat_page = chat_page
            self._chat_page.present()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def _on_chat_page_close_request(self, *_args):
        """Reset the tracked chat page reference when the window closes."""

        self._chat_page = None
        return False

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
        Opens the general settings page.
        """
        if self.ATLAS.is_initialized():
            settings_window = Gtk.Window(title="Settings")
            settings_window.set_default_size(300, 400)
            settings_window.set_transient_for(self)
            settings_window.set_modal(True)
            self.apply_css_styling(settings_window)
            vbox = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
            settings_window.set_child(vbox)
            label = Gtk.Label(label="Settings Page Content Here")
            vbox.append(label)
            settings_window.present()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_speech_settings(self):
        """
        Opens the dedicated Speech Settings page for configuring speech services.
        """
        if self.ATLAS.is_initialized():
            from GTKUI.Settings.Speech.speech_settings import SpeechSettings
            speech_settings_page = SpeechSettings(self.ATLAS)
            speech_settings_page.present()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def close_application(self):
        """
        Closes the application by quitting the parent application.
        """
        logger.info("Closing application")
        app = self.get_application()
        if app:
            app.quit()

    def show_error_dialog(self, message):
        """
        Displays an error dialog.

        Args:
            message (str): Error message to display.
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
        Applies custom CSS styling to the provided window.

        Args:
            window (Gtk.Window, optional): Window to style.
        """
        target = window if window else self
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            .sidebar {
                background-color: #2b2b2b;
            }
            /* Applied to the Gtk.Button wrapping the picture */
            .icon {
                margin: 5px;
                border-radius: 5px;
                padding: 2px;
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
