"""Dedicated GTK window for configuring OpenAI provider defaults."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.utils import create_box

logger = logging.getLogger(__name__)


class OpenAISettingsWindow(Gtk.Window):
    """Collect OpenAI-specific preferences such as default model and streaming mode."""

    def __init__(self, ATLAS, config_manager, parent_window):
        super().__init__(title="OpenAI Settings")
        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window
        if parent_window is not None:
            self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_default_size(420, 320)

        self._last_message: Optional[Tuple[str, str, Gtk.MessageType]] = None

        main_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        self.set_child(main_box)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        main_box.append(grid)

        row = 0
        model_label = Gtk.Label(label="Default Model:")
        model_label.set_xalign(0.0)
        grid.attach(model_label, 0, row, 1, 1)
        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        grid.attach(self.model_combo, 1, row, 1, 1)

        row += 1
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_xalign(0.0)
        grid.attach(temp_label, 0, row, 1, 1)
        self.temperature_adjustment = Gtk.Adjustment(lower=0.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0)
        self.temperature_spin = Gtk.SpinButton(adjustment=self.temperature_adjustment, digits=2)
        self.temperature_spin.set_increments(0.05, 0.1)
        self.temperature_spin.set_hexpand(True)
        grid.attach(self.temperature_spin, 1, row, 1, 1)

        row += 1
        tokens_label = Gtk.Label(label="Max Tokens:")
        tokens_label.set_xalign(0.0)
        grid.attach(tokens_label, 0, row, 1, 1)
        self.max_tokens_adjustment = Gtk.Adjustment(lower=1, upper=128000, step_increment=128, page_increment=512, value=4000)
        self.max_tokens_spin = Gtk.SpinButton(adjustment=self.max_tokens_adjustment, digits=0)
        self.max_tokens_spin.set_increments(128, 512)
        self.max_tokens_spin.set_hexpand(True)
        grid.attach(self.max_tokens_spin, 1, row, 1, 1)

        row += 1
        self.stream_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.stream_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.stream_toggle, 0, row, 2, 1)

        row += 1
        base_url_label = Gtk.Label(label="Base URL (optional):")
        base_url_label.set_xalign(0.0)
        grid.attach(base_url_label, 0, row, 1, 1)
        self.base_url_entry = Gtk.Entry()
        self.base_url_entry.set_hexpand(True)
        self.base_url_entry.set_placeholder_text("https://api.openai.com/v1")
        grid.attach(self.base_url_entry, 1, row, 1, 1)

        row += 1
        org_label = Gtk.Label(label="Organization (optional):")
        org_label.set_xalign(0.0)
        grid.attach(org_label, 0, row, 1, 1)
        self.organization_entry = Gtk.Entry()
        self.organization_entry.set_hexpand(True)
        self.organization_entry.set_placeholder_text("org-1234")
        grid.attach(self.organization_entry, 1, row, 1, 1)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        button_box.set_halign(Gtk.Align.END)
        main_box.append(button_box)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", lambda *_: self.close())
        button_box.append(cancel_button)

        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)
        button_box.append(save_button)

        self._populate_defaults()

    def _populate_defaults(self) -> None:
        settings = self._get_settings_snapshot()
        models = self._load_available_models()

        self.model_combo.remove_all()
        if settings.get("model") and settings["model"] not in models:
            models = [settings["model"]] + [name for name in models if name != settings["model"]]

        if not models:
            models = [settings.get("model") or "gpt-4o"]

        active_index = 0
        for idx, name in enumerate(models):
            self.model_combo.append_text(name)
            if name == settings.get("model"):
                active_index = idx
        self.model_combo.set_active(active_index)

        self.temperature_spin.set_value(float(settings.get("temperature", 0.0)))
        self.max_tokens_spin.set_value(float(settings.get("max_tokens", 4000)))
        self.stream_toggle.set_active(bool(settings.get("stream", True)))
        self.base_url_entry.set_text(settings.get("base_url") or "")
        self.organization_entry.set_text(settings.get("organization") or "")

    def _get_settings_snapshot(self) -> Dict[str, Any]:
        try:
            if hasattr(self.ATLAS, "get_openai_llm_settings"):
                return dict(self.ATLAS.get_openai_llm_settings() or {})
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read OpenAI settings from ATLAS: %s", exc, exc_info=True)

        try:
            return dict(self.config_manager.get_openai_llm_settings() or {})
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read OpenAI settings from config manager: %s", exc, exc_info=True)
            return {}

    def _load_available_models(self) -> List[str]:
        provider_manager = getattr(self.ATLAS, "provider_manager", None)
        if provider_manager and getattr(provider_manager, "model_manager", None):
            try:
                payload = provider_manager.model_manager.get_available_models("OpenAI")
                models = payload.get("OpenAI", [])
                if isinstance(models, list):
                    return list(models)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to load OpenAI model list: %s", exc, exc_info=True)
        return []

    def on_save_clicked(self, _button: Gtk.Button):
        model = self.model_combo.get_active_text()
        if not model:
            self._show_message("Error", "Please choose a default model before saving.", Gtk.MessageType.ERROR)
            return

        payload = {
            "model": model,
            "temperature": self.temperature_spin.get_value(),
            "max_tokens": self.max_tokens_spin.get_value_as_int(),
            "stream": self.stream_toggle.get_active(),
            "base_url": self.base_url_entry.get_text().strip() or None,
            "organization": self.organization_entry.get_text().strip() or None,
        }

        try:
            result = self.ATLAS.set_openai_llm_settings(**payload)
        except Exception as exc:
            logger.error("Error saving OpenAI settings: %s", exc, exc_info=True)
            self._show_message("Error", str(exc), Gtk.MessageType.ERROR)
            return

        if isinstance(result, dict) and result.get("success"):
            message = result.get("message", "OpenAI settings saved.")
            self._show_message("Success", message, Gtk.MessageType.INFO)
            self.close()
            return

        if isinstance(result, dict):
            detail = result.get("error") or result.get("message") or "Unable to save OpenAI settings."
        elif result is None:
            detail = "Unable to save OpenAI settings."
        else:
            detail = str(result)

        self._show_message("Error", detail, Gtk.MessageType.ERROR)

    def _show_message(self, title: str, message: str, message_type: Gtk.MessageType):
        self._last_message = (title, message, message_type)

        try:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=message_type,
                buttons=Gtk.ButtonsType.OK,
                text=title,
            )
        except Exception:  # pragma: no cover - fallback when running in stubbed environments
            logger.debug("GTK message dialog unavailable; skipping display for '%s'.", title)
            return

        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:
            dialog.props.secondary_text = message

        dialog.connect("response", lambda dlg, *_: dlg.destroy())
        GLib.idle_add(dialog.present)
