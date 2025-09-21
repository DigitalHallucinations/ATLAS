# GTKUI/Provider_manager/provider_management.py

import threading
import gi
import asyncio
from typing import Any, Dict
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GLib

import os
import logging

from GTKUI.Utils.utils import create_box
from .Settings.HF_settings import HuggingFaceSettingsWindow

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

        # Reused in settings window scope
        self.api_key_entry: Gtk.Entry | None = None
        self.api_key_toggle: Gtk.ToggleButton | None = None
        self.api_key_visible = False

    # ------------------------ Utilities ------------------------

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

    def _abs_icon(self, relative_path: str) -> str:
        """Resolve absolute path for an icon relative to this file."""
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        return os.path.join(base, relative_path)

    def _load_icon_picture(self, primary_path: str, fallback_icon_name: str, size: int = 16) -> Gtk.Widget:
        """
        Try to load a paintable from a file path; fall back to a themed icon name.
        Returns a Gtk.Picture (file) or Gtk.Image (themed) as a Widget.
        """
        try:
            texture = Gdk.Texture.new_from_filename(primary_path)
            pic = Gtk.Picture.new_for_paintable(texture)
            pic.set_size_request(size, size)
            pic.set_content_fit(Gtk.ContentFit.CONTAIN)
            return pic
        except Exception:
            img = Gtk.Image.new_from_icon_name(fallback_icon_name)
            img.set_pixel_size(size)
            return img

    # ------------------------ Provider Menu ------------------------

    def show_provider_menu(self):
        """
        Displays the provider selection window, listing all available providers.
        Each provider is displayed with a label and a settings icon.
        """
        self.provider_window = Gtk.Window(title="Select Provider")
        self.provider_window.set_default_size(300, 400)
        self.provider_window.set_transient_for(self.parent_window)
        self.provider_window.set_modal(True)
        self.provider_window.set_tooltip_text("Choose a default LLM provider or open its settings.")

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_propagate_natural_height(True)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        self.provider_window.set_child(scrolled)

        box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        box.set_tooltip_text("Available providers registered in ATLAS.")
        box.set_valign(Gtk.Align.START)
        scrolled.set_child(box)

        provider_names = self.ATLAS.get_available_providers()

        for provider_name in provider_names:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

            # Create and configure the provider label.
            label = Gtk.Label(label=provider_name)
            label.set_xalign(0.0)
            label.set_yalign(0.5)
            label.get_style_context().add_class("provider-label")
            label.set_tooltip_text(f"Click to select {provider_name} as the active provider.")

            # Attach a click gesture to the label for provider selection.
            label_click = Gtk.GestureClick.new()
            label_click.connect(
                "released",
                lambda gesture, n_press, x, y, provider_name=provider_name: self.select_provider(provider_name)
            )
            label.add_controller(label_click)

            # Settings icon (file -> themed fallback).
            settings_icon_path = self._abs_icon("Icons/settings.png")
            settings_widget = self._load_icon_picture(settings_icon_path, "emblem-system-symbolic", 16)
            settings_widget.set_tooltip_text(f"Open {provider_name} settings (API keys, options).")

            settings_click = Gtk.GestureClick.new()
            settings_click.connect(
                "released",
                lambda gesture, n_press, x, y, provider_name=provider_name: self.open_provider_settings(provider_name)
            )
            settings_widget.add_controller(settings_click)

            hbox.append(label)
            hbox.append(settings_widget)
            box.append(hbox)

        self.provider_window.present()

    def select_provider(self, provider: str):
        """Attempt to switch the active provider and display any errors."""
        async def switch_provider():
            try:
                await self.ATLAS.set_current_provider(provider)
            except Exception as exc:
                self.logger.error(f"Failed to select provider {provider}: {exc}")
                GLib.idle_add(self.show_error_dialog, str(exc))
                return

            self.logger.info(f"Provider {provider} selected.")
            if self.provider_window:
                GLib.idle_add(self.provider_window.close)

        # Run the provider switch asynchronously to avoid blocking the UI.
        self._run_async_task(switch_provider())

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
            result = self.ATLAS.provider_manager.ensure_huggingface_ready()
            if not result.get("success"):
                message = result.get("error", "Unable to open HuggingFace settings.")
                self.logger.error("Failed to prepare HuggingFace settings window: %s", message)
                self.show_error_dialog(message)
                return
            self.show_huggingface_settings()
        else:
            self.show_provider_settings(provider_name)

    def show_huggingface_settings(self):
        """
        Displays the HuggingFace settings window.
        """
        settings_window = HuggingFaceSettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        settings_window.set_tooltip_text("Configure HuggingFace provider options and credentials.")
        settings_window.present()

    # ------------------------ Settings Window ------------------------

    def show_provider_settings(self, provider_name: str):
        """
        Displays the settings window for a specific provider, including an API key entry
        with a visibility toggle (eye).
        """
        settings_window = Gtk.Window(title=f"Settings for {provider_name}")
        settings_window.set_default_size(400, 300)
        settings_window.set_transient_for(self.parent_window)
        settings_window.set_modal(True)
        settings_window.set_tooltip_text(f"Update API key and settings for {provider_name}.")

        self.apply_css_styling()

        main_vbox = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        settings_window.set_child(main_vbox)

        # Display the provider name.
        provider_label = Gtk.Label(label="Provider:")
        provider_label.set_xalign(0.0)
        provider_label.set_margin_bottom(10)
        provider_label.set_tooltip_text("The provider whose settings you are editing.")
        main_vbox.append(provider_label)

        provider_value = Gtk.Label(label=provider_name)
        provider_value.set_xalign(0.0)
        provider_value.set_margin_bottom(20)
        provider_value.set_tooltip_text(f"{provider_name}")
        main_vbox.append(provider_value)

        # API key row with eye toggle
        api_row = self._build_api_row(provider_name)
        main_vbox.append(api_row)

        # Save button
        save_button = Gtk.Button(label="Save")
        save_button.set_tooltip_text("Save the API key and refresh the provider if it is active.")
        save_button.connect("clicked", self.on_save_button_clicked, provider_name, settings_window)
        main_vbox.append(save_button)

        settings_window.present()

    def _build_api_row(self, provider_name: str) -> Gtk.Widget:
        """
        Build a labeled API key row: label, password entry, and eye toggle to show/hide.
        """
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        api_key_label = Gtk.Label(label="API Key:")
        api_key_label.set_xalign(0.0)
        api_key_label.set_tooltip_text("Enter your API key. Use the eye to show/hide.")
        vbox.append(api_key_label)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Entry
        self.api_key_entry = Gtk.Entry()
        base_placeholder = "Enter your API key here"
        base_tooltip = "API key is hidden. Click the eye to reveal."
        self.api_key_entry.set_placeholder_text(base_placeholder)
        self.api_key_entry.set_visibility(False)  # password-style default
        self.api_key_entry.set_invisible_char('*')
        self.api_key_entry.set_tooltip_text(base_tooltip)
        self.api_key_entry.set_hexpand(True)
        hbox.append(self.api_key_entry)

        # Eye toggle button
        self.api_key_toggle = Gtk.ToggleButton()
        self.api_key_toggle.set_can_focus(True)
        self.api_key_toggle.set_tooltip_text("Show API key")
        # NOTE: No explicit AccessibleRole.TOGGLE_BUTTON in GTK4. Let GTK choose, or set BUTTON if desired:
        role = getattr(Gtk.AccessibleRole, "BUTTON", None)
        if role is not None:
            self.api_key_toggle.set_accessible_role(role)

        # Load icons (project file -> themed fallback)
        self._eye_icon_path = self._abs_icon("Icons/eye.png")
        self._eye_off_icon_path = self._abs_icon("Icons/eye-off.png")

        eye_widget = self._load_icon_picture(self._eye_icon_path, "view-reveal-symbolic", 18)
        self.api_key_toggle.set_child(eye_widget)

        # Connect toggle
        self.api_key_toggle.connect("toggled", self._on_api_eye_toggled)
        hbox.append(self.api_key_toggle)

        vbox.append(hbox)

        # Pre-fill placeholder when key exists
        self._apply_provider_status_to_entry(
            self.api_key_entry,
            provider_name,
            base_placeholder,
            base_tooltip,
        )
        return vbox

    def _on_api_eye_toggled(self, toggle_btn: Gtk.ToggleButton):
        """
        Toggle visibility of the API key entry and swap the icon/tooltip accordingly.
        """
        self.api_key_visible = toggle_btn.get_active()
        if self.api_key_entry:
            self.api_key_entry.set_visibility(self.api_key_visible)
        # Swap icon
        icon_name = "view-conceal-symbolic" if self.api_key_visible else "view-reveal-symbolic"
        icon_path = self._eye_off_icon_path if self.api_key_visible else self._eye_icon_path
        new_widget = self._load_icon_picture(icon_path, icon_name, 18)
        toggle_btn.set_child(new_widget)
        # Update tooltip
        toggle_btn.set_tooltip_text("Hide API key" if self.api_key_visible else "Show API key")

    # ------------------------ Helpers & Async ------------------------

    def _apply_provider_status_to_entry(
        self,
        entry: Gtk.Entry,
        provider_name: str,
        base_placeholder: str,
        base_tooltip: str,
    ) -> None:
        """Update placeholder/tooltip to reflect saved provider credential state."""

        entry.set_text("")
        entry.set_placeholder_text(base_placeholder)
        entry.set_tooltip_text(base_tooltip)

        status = self._get_provider_key_status(provider_name)
        if not status.get("has_key"):
            return

        metadata: Dict[str, Any] = status.get("metadata") or {}
        placeholder = "Saved"
        hint = metadata.get("hint")
        if isinstance(hint, str) and hint:
            placeholder = f"{placeholder} {hint}"

        entry.set_placeholder_text(placeholder)

        tooltip = f"{base_tooltip} Stored key detected."
        length = metadata.get("length")
        if isinstance(length, int) and length > 0:
            tooltip = f"{tooltip} Length: {length} characters."
        entry.set_tooltip_text(tooltip)

    def _get_provider_key_status(self, provider_name: str) -> Dict[str, Any]:
        """Fetch sanitized provider credential details from the provider manager."""

        manager = getattr(self.ATLAS, "provider_manager", None)
        getter = getattr(manager, "get_provider_api_key_status", None)
        if not callable(getter):
            self.logger.debug("Provider manager does not expose credential status helper.")
            return {"has_key": False, "metadata": {}}

        try:
            status = getter(provider_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Failed to fetch API key status for %s: %s", provider_name, exc, exc_info=True
            )
            return {"has_key": False, "metadata": {}}

        if isinstance(status, dict):
            return status

        self.logger.debug(
            "Provider key status for %s returned unexpected payload type %s.",
            provider_name,
            type(status).__name__,
        )
        return {"has_key": False, "metadata": {}}

    def on_save_button_clicked(self, button, provider_name: str, window: Gtk.Window):
        """
        Handles the Save button click event in the provider settings window.
        It updates the API key and refreshes the provider settings.

        Args:
            button (Gtk.Button): The button that was clicked.
            provider_name (str): The name of the provider.
            window (Gtk.Window): The settings window to close after updating.
        """
        new_api_key = (self.api_key_entry.get_text().strip() if self.api_key_entry else "")
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
            result = await self.ATLAS.provider_manager.update_provider_api_key(provider_name, new_api_key)
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.error(f"Failed to save API Key: {str(e)}", exc_info=True)
            GLib.idle_add(self.show_error_dialog, f"Failed to save API Key: {str(e)}")
            return

        if result.get("success"):
            message = result.get("message") or f"API Key for {provider_name} saved successfully."
            self.logger.info(message)
            GLib.idle_add(self.show_info_dialog, message)
            GLib.idle_add(window.close)
        else:
            error_message = result.get("error") or f"Failed to save API Key for {provider_name}."
            self.logger.error(f"Failed to save API Key: {error_message}")
            GLib.idle_add(self.show_error_dialog, error_message)

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

    # ------------------------ Dialogs & CSS ------------------------

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
        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:  # Fallback for API variations
            dialog.props.secondary_text = message
        dialog.connect("response", lambda dialog, response: dialog.destroy())
        dialog.set_tooltip_text("Close to dismiss this error message.")
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
        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:  # Fallback for API variations
            dialog.props.secondary_text = message
        dialog.connect("response", lambda dialog, response: dialog.destroy())
        dialog.set_tooltip_text("Close to continue.")
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
        # Use the app's display if possible; fall back safely.
        display = None
        try:
            if self.parent_window:
                display = self.parent_window.get_display()
        except Exception:
            display = None
        if display is None:
            display = Gtk.Window().get_display()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
