# GTKUI/Provider_manager/provider_management.py

import threading
import gi
import asyncio
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GLib

import os
import logging

from GTKUI.Utils.utils import create_box  
from .Settings.HF_settings import HuggingFaceSettingsWindow
from modules.Providers.HuggingFace.HF_gen_response import HuggingFaceGenerator

class ProviderManagement:
    """
    Manages provider-related functionalities, including displaying available
    providers, handling provider selection, and managing provider settings.
    """

    def __init__(self, ATLAS, parent_window):
        """
        Initializes the ProviderManagement with the given ATLAS instance and parent window.

        Args:
            ATLAS (ATLAS): The main ATLAS instance.
            parent_window (Gtk.Window): The parent GTK window.
        """
        self.ATLAS = ATLAS
        self.parent_window = parent_window
        self.provider_window = None
        self.config_manager = self.ATLAS.config_manager  
        self.logger = logging.getLogger(__name__)

    def _run_async_task(self, coro):
        """
        Helper method to run an asynchronous coroutine in a separate thread.
        This ensures that the UI thread remains responsive.

        Args:
            coro (coroutine): The coroutine to run.
        """
        def task():
            try:
                asyncio.run(coro)
            except Exception as e:
                # Report errors back to the UI thread.
                GLib.idle_add(self.show_error_dialog, f"Async task error: {e}")
        threading.Thread(target=task, daemon=True).start()

    def show_provider_menu(self):
        """
        Displays the provider selection window, listing all available providers.
        Each provider is displayed with a label and a settings icon.
        """
        self.provider_window = Gtk.Window(title="Select Provider")
        self.provider_window.set_default_size(300, 400)
        self.provider_window.set_transient_for(self.parent_window)
        self.provider_window.set_modal(True)

        box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        self.provider_window.set_child(box)

        provider_names = self.ATLAS.get_available_providers()

        for provider_name in provider_names:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            # Create and configure the provider label.
            label = Gtk.Label(label=provider_name)
            label.set_xalign(0.0)
            label.set_yalign(0.5)
            label.get_style_context().add_class("provider-label")

            # Attach a click gesture to the label for provider selection.
            label_click = Gtk.GestureClick.new()
            label_click.connect(
                "released",
                lambda gesture, n_press, x, y, provider_name=provider_name: self.select_provider(provider_name)
            )
            label.add_controller(label_click)

            # Compute the absolute path for the settings icon.
            settings_icon_path = os.path.join(os.path.dirname(__file__), "../../Icons/settings.png")
            settings_icon_path = os.path.abspath(settings_icon_path)
            try:
                texture = Gdk.Texture.new_from_filename(settings_icon_path)
                settings_icon = Gtk.Picture.new_for_paintable(texture)
                settings_icon.set_size_request(16, 16)
                settings_icon.set_content_fit(Gtk.ContentFit.CONTAIN)
            except GLib.Error as e:
                self.logger.error(f"Error loading settings icon from '{settings_icon_path}': {e}")
                settings_icon = Gtk.Image.new_from_icon_name("image-missing")

            # Attach a click gesture to the settings icon to open provider settings.
            settings_click = Gtk.GestureClick.new()
            settings_click.connect(
                "released",
                lambda gesture, n_press, x, y, provider_name=provider_name: self.open_provider_settings(provider_name)
            )
            settings_icon.add_controller(settings_click)

            hbox.append(label)
            hbox.append(settings_icon)
            box.append(hbox)

        self.provider_window.present()

    def select_provider(self, provider: str):
        """
        Handles the selection of a provider. Sets the current provider in ATLAS.
        If the provider's API key is not set, it prompts the user to enter it.

        Args:
            provider (str): The name of the provider to select.
        """
        api_key = self.config_manager.get_config(f"{provider.upper()}_API_KEY")
        if not api_key:
            self.logger.info(f"No API key set for provider {provider}. Prompting user to enter it.")
            self.open_provider_settings(provider)
        else:
            # Run the provider switch asynchronously to avoid blocking the UI.
            self._run_async_task(self.ATLAS.set_current_provider(provider))
            self.logger.info(f"Provider {provider} selected.")
            GLib.idle_add(self.provider_window.close)

    def open_provider_settings(self, provider_name: str):
        """
        Opens the provider settings window for the specified provider.
        For HuggingFace, initializes its generator if needed.

        Args:
            provider_name (str): The name of the provider.
        """
        if self.provider_window:
            GLib.idle_add(self.provider_window.close)

        if provider_name == "HuggingFace":
            if self.ATLAS.provider_manager.huggingface_generator is None:
                self.logger.info("Initializing HuggingFace generator in open_provider_settings")
                self.ATLAS.provider_manager.huggingface_generator = HuggingFaceGenerator(self.config_manager)
                self.logger.info("HuggingFace generator initialized successfully")
            else:
                self.logger.info("HuggingFace generator already initialized")
            self.show_huggingface_settings()
        else:
            self.show_provider_settings(provider_name)

    def show_huggingface_settings(self):
        """
        Displays the HuggingFace settings window.
        """
        settings_window = HuggingFaceSettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        settings_window.present()

    def show_provider_settings(self, provider_name: str):
        """
        Displays the settings window for a specific provider, including an API key entry.

        Args:
            provider_name (str): The name of the provider.
        """
        settings_window = Gtk.Window(title=f"Settings for {provider_name}")
        settings_window.set_default_size(400, 300)
        settings_window.set_transient_for(self.parent_window)
        settings_window.set_modal(True)

        self.apply_css_styling()

        main_vbox = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        settings_window.set_child(main_vbox)

        # Display the provider name.
        provider_label = Gtk.Label(label="Provider:")
        provider_label.set_xalign(0.0)
        provider_label.set_margin_bottom(10)
        main_vbox.append(provider_label)

        provider_value = Gtk.Label(label=provider_name)
        provider_value.set_xalign(0.0)
        provider_value.set_margin_bottom(20)
        main_vbox.append(provider_value)

        # Create the API key entry field.
        api_key_label = Gtk.Label(label="API Key:")
        api_key_label.set_xalign(0.0)
        main_vbox.append(api_key_label)

        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_placeholder_text("Enter your API key here")
        self.api_key_entry.set_visibility(False)  
        self.api_key_entry.set_invisible_char('*')
        main_vbox.append(self.api_key_entry)

        # Pre-fill the API key if it exists.
        existing_api_key = self.get_existing_api_key(provider_name)
        if existing_api_key:
            self.api_key_entry.set_text(existing_api_key)

        # Add a Save button to update the API key.
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", self.on_save_button_clicked, provider_name, settings_window)
        main_vbox.append(save_button)

        settings_window.present()

    def get_existing_api_key(self, provider_name: str) -> str:
        """
        Retrieves the existing API key for the given provider from ConfigManager.

        Args:
            provider_name (str): The name of the provider.

        Returns:
            str: The existing API key or an empty string if not set.
        """
        api_key_methods = {
            "OpenAI": self.config_manager.get_openai_api_key,
            "Mistral": self.config_manager.get_mistral_api_key,
            "Google": self.config_manager.get_google_api_key,
            "HuggingFace": self.config_manager.get_huggingface_api_key,
            "Anthropic": self.config_manager.get_anthropic_api_key,
            "Grok": self.config_manager.get_grok_api_key,  
        }
        get_key_func = api_key_methods.get(provider_name)
        if get_key_func:
            return get_key_func() or ""
        return ""

    def on_save_button_clicked(self, button, provider_name: str, window: Gtk.Window):
        """
        Handles the Save button click event in the provider settings window.
        It updates the API key and refreshes the provider settings.

        Args:
            button (Gtk.Button): The button that was clicked.
            provider_name (str): The name of the provider.
            window (Gtk.Window): The settings window to close after updating.
        """
        new_api_key = self.api_key_entry.get_text().strip()
        if not new_api_key:
            self.show_error_dialog("API Key cannot be empty.")
            return

        # Run the API key update asynchronously.
        self._run_async_task(self._update_api_key_async(provider_name, new_api_key, window))

    async def _update_api_key_async(self, provider_name: str, new_api_key: str, window: Gtk.Window):
        """
        Asynchronously updates the API key for the given provider and refreshes the provider.
        Notifies the user upon successful update or error.

        Args:
            provider_name (str): The name of the provider.
            new_api_key (str): The new API key to set.
            window (Gtk.Window): The settings window to close after updating.
        """
        try:
            self.config_manager.update_api_key(provider_name, new_api_key)
            self.logger.info(f"API Key for {provider_name} updated.")
            await self.refresh_provider_async(provider_name)
            self.logger.info(f"Provider {provider_name} refreshed.")
            GLib.idle_add(self.show_info_dialog, f"API Key for {provider_name} saved successfully.")
            GLib.idle_add(window.close)
        except Exception as e:
            self.logger.error(f"Failed to save API Key: {str(e)}")
            GLib.idle_add(self.show_error_dialog, f"Failed to save API Key: {str(e)}")

    async def refresh_provider_async(self, provider_name: str):
        """
        Asynchronously refreshes the provider if the given provider matches the current one.
        This ensures that any changes (such as an updated API key) take effect.

        Args:
            provider_name (str): The name of the provider to refresh.
        """
        if provider_name == self.ATLAS.provider_manager.current_llm_provider:
            try:
                await self.ATLAS.provider_manager.set_current_provider(provider_name)
                self.logger.info(f"Provider {provider_name} refreshed with new API key.")
            except Exception as e:
                self.logger.error(f"Error refreshing provider {provider_name}: {e}")
                GLib.idle_add(self.show_error_dialog, f"Error refreshing provider {provider_name}: {e}")

    def show_error_dialog(self, message: str):
        """
        Displays an error dialog with the specified message.

        Args:
            message (str): The error message to display.
        """
        dialog = Gtk.MessageDialog(
            transient_for=self.provider_window or self.parent_window,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error",
        )
        dialog.format_secondary_text(message)
        dialog.connect("response", lambda dialog, response: dialog.destroy())
        dialog.present()

    def show_info_dialog(self, message: str):
        """
        Displays an information dialog with the specified message.

        Args:
            message (str): The information message to display.
        """
        dialog = Gtk.MessageDialog(
            transient_for=self.provider_window or self.parent_window,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Information",
        )
        dialog.format_secondary_text(message)
        dialog.connect("response", lambda dialog, response: dialog.destroy())
        dialog.present()

    def apply_css_styling(self):
        """
        Applies CSS styling to the provider settings window.
        This ensures a consistent look and feel across the application.
        """
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            * { background-color: #2b2b2b; color: white; }
            label { margin: 5px; }
            entry { background-color: #3c3c3c; color: white; }
            button { background-color: #555555; color: white; }
            button:hover { background-color: #4a90d9; }
            button:active { background-color: #357ABD; }
            .provider-label {
                font-weight: bold;
            }
        """)
        display = Gtk.Window().get_display()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
