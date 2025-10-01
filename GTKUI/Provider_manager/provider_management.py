# GTKUI/Provider_manager/provider_management.py

import gi
from typing import Any, Dict
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Gdk, GLib

import os
import logging

from GTKUI.Utils.utils import apply_css, create_box
from .Settings.HF_settings import HuggingFaceSettingsWindow
from .Settings.OA_settings import OpenAISettingsWindow
from .Settings.Anthropic_settings import AnthropicSettingsWindow
from .Settings.Google_settings import GoogleSettingsWindow

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

        apply_css()
        style_context = self.provider_window.get_style_context()
        style_context.add_class("chat-page")
        style_context.add_class("sidebar")

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
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
        def handle_success(_result: Any) -> None:
            self.logger.info(f"Provider {provider} selected.")
            GLib.idle_add(self._close_provider_window)

        def handle_error(exc: Exception) -> None:
            message = str(exc) or f"Failed to select provider {provider}."
            self.logger.error(
                "Failed to select provider %s: %s", provider, exc, exc_info=True
            )
            GLib.idle_add(self.show_error_dialog, message)

        try:
            self.ATLAS.set_current_provider_in_background(
                provider,
                on_success=handle_success,
                on_error=handle_error,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Unable to schedule provider switch to %s: %s", provider, exc, exc_info=True
            )
            self.show_error_dialog(str(exc))

    def open_provider_settings(self, provider_name: str):
        """
        Opens the provider settings window for the specified provider.
        For HuggingFace, initializes its generator if needed.

        Args:
            provider_name (str): The name of the provider.
        """
        if self.provider_window:
            GLib.idle_add(self.provider_window.close)

        if provider_name == "OpenAI":
            self.show_openai_settings()
        elif provider_name == "Google":
            self.show_google_settings()
        elif provider_name == "HuggingFace":
            try:
                result = self.ATLAS.ensure_huggingface_ready()
            except AttributeError:
                message = "HuggingFace support is unavailable in this build."
                self.logger.error(message)
                self.show_error_dialog(message)
                return
            except Exception as exc:
                message = str(exc) or "Unable to open HuggingFace settings."
                self.logger.error(
                    "Failed to prepare HuggingFace settings window: %s", message, exc_info=True
                )
                self.show_error_dialog(message)
                return

            if not isinstance(result, dict) or not result.get("success"):
                message = "Unable to open HuggingFace settings."
                if isinstance(result, dict):
                    message = result.get("error", message)
                self.logger.error(
                    "Failed to prepare HuggingFace settings window: %s", message
                )
                self.show_error_dialog(message)
                return

            self.show_huggingface_settings()
        elif provider_name == "Anthropic":
            self.show_anthropic_settings()
        else:
            self.show_provider_settings(provider_name)

    def show_openai_settings(self):
        """Display the OpenAI provider configuration dialog."""

        settings_window = OpenAISettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        settings_window.set_tooltip_text("Configure OpenAI default parameters and endpoints.")
        settings_window.present()

    def show_huggingface_settings(self):
        """
        Displays the HuggingFace settings window.
        """
        settings_window = HuggingFaceSettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        settings_window.set_tooltip_text("Configure HuggingFace provider options and credentials.")
        settings_window.present()

    def show_anthropic_settings(self):
        """Display the Anthropic provider configuration dialog."""

        settings_window = AnthropicSettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        settings_window.set_tooltip_text("Configure Anthropic defaults and retry behaviour.")
        settings_window.present()

    def show_google_settings(self):
        """Display the Google provider configuration dialog."""

        settings_window = GoogleSettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        settings_window.set_tooltip_text("Configure Google Gemini defaults and safety filters.")
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

        apply_css()
        style_context = settings_window.get_style_context()
        style_context.add_class("chat-page")
        style_context.add_class("sidebar")

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

        # Action buttons (credentials + settings)
        controls_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        controls_box.set_halign(Gtk.Align.END)
        controls_box.set_tooltip_text(
            "Use these controls to persist credentials or save provider configuration."
        )
        main_vbox.append(controls_box)

        save_key_button = Gtk.Button(label="Save API Key/Token")
        save_key_button.set_tooltip_text(
            "Persist the API key/token and refresh the provider if it is currently active."
        )
        save_key_button.connect(
            "clicked",
            self.on_save_api_key_clicked,
            provider_name,
            settings_window,
        )
        controls_box.append(save_key_button)

        save_settings_button = Gtk.Button(label="Save Settings")
        save_settings_button.set_tooltip_text(
            "Save non-credential provider preferences without changing the stored API key/token."
        )
        save_settings_button.connect(
            "clicked",
            self.on_save_settings_clicked,
            provider_name,
            settings_window,
        )
        controls_box.append(save_settings_button)

        settings_window.present()

    def _build_api_row(self, provider_name: str) -> Gtk.Widget:
        """
        Build a labeled API key row: label, password entry, and eye toggle to show/hide.
        """
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        api_key_label = Gtk.Label(label="API Key / Token:")
        api_key_label.set_xalign(0.0)
        api_key_label.set_tooltip_text(
            "Enter your API key or token. Use the eye to show/hide and the credential button to save."
        )
        vbox.append(api_key_label)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Entry
        self.api_key_entry = Gtk.Entry()
        base_placeholder = "Enter your API key or token here"
        base_tooltip = (
            "Credential input is hidden. Click the eye to reveal and 'Save API Key/Token' to persist."
        )
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
        """Fetch sanitized provider credential details via the ATLAS facade."""

        getter = getattr(self.ATLAS, "get_provider_api_key_status", None)
        if not callable(getter):
            self.logger.debug("ATLAS facade does not expose credential status helper.")
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

    def on_save_api_key_clicked(self, button, provider_name: str, window: Gtk.Window):
        """Persist the API key/token entered by the user and refresh the provider if needed."""

        new_api_key = (self.api_key_entry.get_text().strip() if self.api_key_entry else "")
        if not new_api_key:
            self.show_error_dialog("API Key / Token cannot be empty when saving credentials.")
            return

        self._begin_api_key_update(provider_name, new_api_key, window)

    def on_save_settings_clicked(self, button, provider_name: str, window: Gtk.Window):
        """Save non-credential provider settings while optionally delegating credential updates."""

        new_api_key = (self.api_key_entry.get_text().strip() if self.api_key_entry else "")
        if new_api_key:
            # User entered a new credential but pressed Save Settings; honor the intent.
            self.logger.info(
                "Save Settings triggered with credential input for %s; delegating to API key handler.",
                provider_name,
            )
            self._begin_api_key_update(provider_name, new_api_key, window)
            return

        if self._provider_has_saved_key(provider_name):
            self.logger.info(
                "Provider settings for %s saved without modifying stored credentials.",
                provider_name,
            )
            self.show_info_dialog(
                f"Settings for {provider_name} saved. Existing credentials remain in place."
            )
            window.close()
            return

        self.show_error_dialog(
            "No stored API key/token detected. Please provide a credential before saving settings."
        )

    def _provider_has_saved_key(self, provider_name: str) -> bool:
        """Return whether the provider manager reports an existing credential for the provider."""

        status = self._get_provider_key_status(provider_name)
        return bool(status.get("has_key"))

    def _begin_api_key_update(
        self, provider_name: str, new_api_key: str, window: Gtk.Window
    ) -> None:
        """Kick off a provider credential update using the ATLAS background helper."""

        def handle_success(result: Dict[str, Any]) -> None:
            GLib.idle_add(
                self._handle_api_key_update_result, provider_name, window, result
            )

        def handle_error(exc: Exception) -> None:
            self.logger.error(
                "Failed to save API Key for %s: %s", provider_name, exc, exc_info=True
            )
            GLib.idle_add(
                self._handle_api_key_update_result,
                provider_name,
                window,
                {"success": False, "error": str(exc)},
            )

        try:
            self.ATLAS.update_provider_api_key_in_background(
                provider_name,
                new_api_key,
                on_success=handle_success,
                on_error=handle_error,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(
                "Unable to schedule API key update for %s: %s",
                provider_name,
                exc,
                exc_info=True,
            )
            self.show_error_dialog(f"Failed to save API Key: {str(exc)}")

    def _handle_api_key_update_result(
        self, provider_name: str, window: Gtk.Window, result: Dict[str, Any]
    ) -> bool:
        """Display the outcome of an API key update on the main thread."""

        if isinstance(result, dict) and result.get("success"):
            message = (
                result.get("message")
                or f"API Key for {provider_name} saved successfully."
            )
            self.logger.info(message)
            self.show_info_dialog(message)
            window.close()
        else:
            error_message = "Failed to save API Key."
            if isinstance(result, dict):
                error_message = result.get(
                    "error",
                    result.get("message", error_message),
                )
            self.logger.error("Failed to save API Key for %s: %s", provider_name, error_message)
            self.show_error_dialog(error_message)

        return False

    def _close_provider_window(self) -> bool:
        """Close the provider selection window if it is open."""

        if self.provider_window:
            self.provider_window.close()
        return False

    async def refresh_provider_async(self, provider_name: str):
        """
        Asynchronously refreshes the provider if the given provider matches the current one.
        This ensures that any changes (such as an updated API key) take effect.

        Args:
            provider_name (str): The name of the provider to refresh.
        """
        refresher = getattr(self.ATLAS, "refresh_current_provider", None)
        if not callable(refresher):
            self.logger.error("ATLAS facade does not expose refresh_current_provider.")
            return

        try:
            result = await refresher(provider_name)
        except Exception as e:  # pragma: no cover - defensive logging
            self.logger.error(f"Error refreshing provider {provider_name}: {e}", exc_info=True)
            GLib.idle_add(
                self.show_error_dialog,
                f"Error refreshing provider {provider_name}: {e}",
            )
            return

        if isinstance(result, dict) and not result.get("success"):
            message = result.get("error") or f"Provider {provider_name} was not refreshed."
            self.logger.info(message)
            GLib.idle_add(self.show_info_dialog, message)
            return

        self.logger.info(f"Provider {provider_name} refreshed with new API key.")

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
        self._style_dialog(dialog)
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
        self._style_dialog(dialog)
        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:  # Fallback for API variations
            dialog.props.secondary_text = message
        dialog.connect("response", lambda dialog, response: dialog.destroy())
        dialog.set_tooltip_text("Close to continue.")
        dialog.present()

    def _style_dialog(self, dialog: Gtk.Widget) -> None:
        """Ensure dialogs adopt the same dark theme styling as the parent UI."""

        try:
            apply_css()
        except (AttributeError, FileNotFoundError, RuntimeError) as exc:
            self.logger.debug("Skipping CSS application for dialog: %s", exc)
        get_style_context = getattr(dialog, "get_style_context", None)
        if callable(get_style_context):
            style_context = get_style_context()
            if hasattr(style_context, "add_class"):
                style_context.add_class("chat-page")
                style_context.add_class("sidebar")

