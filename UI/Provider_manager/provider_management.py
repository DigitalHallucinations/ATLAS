# ATLAS/UI/Provider_manager/provider_management.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib

import asyncio

class ProviderManagement:
    """
    Manages provider-related functionalities, including displaying available
    providers, handling provider selection, and managing provider settings.
    
    Attributes:
        ATLAS (ATLAS): The main ATLAS instance.
        parent_window (Gtk.Window): The parent GTK window.
        provider_window (Gtk.Window): The window displaying the list of providers.
        config_manager (ConfigManager): Instance of ConfigManager for configuration access.
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

    def show_provider_menu(self):
        """
        Displays the provider selection window, listing all available providers.
        Each provider has a label and a settings icon.
        """
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
            try:
                settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(settings_icon_path, 16, 16, True)
            except GLib.Error as e:
                self.ATLAS.logger.error(f"Error loading settings icon: {e}")
                settings_pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale("Icons/default.png", 16, 16, True)
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

    def select_provider(self, provider: str):
        """
        Handles the selection of a provider. Sets the current provider in ATLAS.
        If the provider's API key is not set, prompts the user to enter it.

        Args:
            provider (str): The name of the provider to select.
        """
        api_key = self.config_manager.get_config(f"{provider.upper()}_API_KEY")
        if not api_key:
            self.ATLAS.logger.info(f"No API key set for provider {provider}. Prompting user to enter it.")
            self.open_provider_settings(provider)
        else:
            asyncio.create_task(self.ATLAS.set_current_provider(provider))
            self.ATLAS.logger.info(f"Provider {provider} selected.")
            self.provider_window.close()

    def open_provider_settings(self, provider_name: str):
        """
        Opens the settings window for the specified provider.

        Args:
            provider_name (str): The name of the provider whose settings are to be opened.
        """
        if self.provider_window:
            self.provider_window.close()

        self.show_provider_settings(provider_name)

    def show_provider_settings(self, provider_name: str):
        """
        Displays the settings window for a specific provider, including the API key entry.

        Args:
            provider_name (str): The name of the provider.
        """
        settings_window = Gtk.Window(title=f"Settings for {provider_name}")
        settings_window.set_default_size(400, 300)
        settings_window.set_keep_above(True)

        self.apply_css_styling()
        self.position_window_next_to_sidebar(settings_window, 400)

        main_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        settings_window.add(main_vbox)

        # Provider Name
        provider_label = Gtk.Label(label="Provider:")
        provider_label.set_xalign(0.0)
        provider_label.set_margin_bottom(10)
        main_vbox.pack_start(provider_label, False, False, 0)

        provider_value = Gtk.Label(label=provider_name)
        provider_value.set_xalign(0.0)
        provider_value.set_margin_bottom(20)
        main_vbox.pack_start(provider_value, False, False, 0)

        # API Key Entry
        api_key_label = Gtk.Label(label="API Key:")
        api_key_label.set_xalign(0.0)
        main_vbox.pack_start(api_key_label, False, False, 0)

        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_placeholder_text("Enter your API key here")
        self.api_key_entry.set_visibility(False)  # Hide the API key input
        self.api_key_entry.set_invisible_char('*')  # Mask the input
        main_vbox.pack_start(self.api_key_entry, False, False, 0)

        # Pre-fill the API key if it exists
        existing_api_key = self.get_existing_api_key(provider_name)
        if existing_api_key:
            self.api_key_entry.set_text(existing_api_key)

        # Save Button
        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", self.on_save_button_clicked, provider_name, settings_window)
        main_vbox.pack_start(save_button, False, False, 20)

        settings_window.show_all()

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

        Args:
            button (Gtk.Button): The button that was clicked.
            provider_name (str): The name of the provider.
            window (Gtk.Window): The settings window to close after saving.
        """
        new_api_key = self.api_key_entry.get_text().strip()
        if not new_api_key:
            self.show_error_dialog("API Key cannot be empty.")
            return

        try:
            self.config_manager.update_api_key(provider_name, new_api_key)
            self.ATLAS.logger.info(f"API Key for {provider_name} updated.")
            self.refresh_provider(provider_name)
            self.show_info_dialog(f"API Key for {provider_name} saved successfully.")
            window.close()
        except Exception as e:
            self.show_error_dialog(f"Failed to save API Key: {str(e)}")

    def refresh_provider(self, provider_name: str):
        """
        Refreshes the provider configuration to use the new API key.

        Args:
            provider_name (str): The name of the provider to refresh.
        """
        if provider_name == self.ATLAS.provider_manager.current_llm_provider:
            asyncio.create_task(self.ATLAS.provider_manager.set_current_provider(provider_name))
            self.ATLAS.logger.info(f"Provider {provider_name} refreshed with new API key.")

    def show_error_dialog(self, message: str):
        """
        Displays an error dialog with the specified message.

        Args:
            message (str): The error message to display.
        """
        dialog = Gtk.MessageDialog(
            transient_for=self.provider_window,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def show_info_dialog(self, message: str):
        """
        Displays an information dialog with the specified message.

        Args:
            message (str): The information message to display.
        """
        dialog = Gtk.MessageDialog(
            transient_for=self.provider_window,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Information",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def position_window_next_to_sidebar(self, window: Gtk.Window, window_width: int):
        """
        Positions the given window next to the sidebar.

        Args:
            window (Gtk.Window): The window to position.
            window_width (int): The width of the window.
        """
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        screen_width = monitor_geometry.width

        window_height = window.get_preferred_height()[1]

        window_x = screen_width - 50 - 10 - window_width
        window.set_default_size(
            window_width,
            window_height if window_height < monitor_geometry.height else monitor_geometry.height - 50
        )

        window.move(window_x, 0)

    def apply_css_styling(self):
        """
        Applies CSS styling to the provider settings window.
        """
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            * { background-color: #2b2b2b; color: white; }
            label { margin: 5px; }
            entry { background-color: #3c3c3c; color: white; }
            button { background-color: #555555; color: white; }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
