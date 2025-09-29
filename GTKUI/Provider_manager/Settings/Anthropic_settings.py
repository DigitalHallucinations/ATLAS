from __future__ import annotations

"""GTK window for configuring Anthropic provider defaults."""

import logging
from typing import Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.utils import create_box

logger = logging.getLogger(__name__)


class AnthropicSettingsWindow(Gtk.Window):
    """Collect Anthropic-specific preferences such as default model and retries."""

    def __init__(self, ATLAS, config_manager, parent_window):
        super().__init__(title="Anthropic Settings")
        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window
        if parent_window is not None:
            self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_default_size(420, 260)

        self._last_message: Optional[Tuple[str, str, Gtk.MessageType]] = None

        container = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        self.set_child(container)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        container.append(grid)

        row = 0
        model_label = Gtk.Label(label="Default Model:")
        model_label.set_xalign(0.0)
        grid.attach(model_label, 0, row, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        grid.attach(self.model_combo, 1, row, 1, 1)

        row += 1
        self.streaming_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.streaming_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.streaming_toggle, 0, row, 2, 1)

        row += 1
        self.function_call_toggle = Gtk.CheckButton(label="Enable tool/function calling")
        self.function_call_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.function_call_toggle, 0, row, 2, 1)

        row += 1
        timeout_label = Gtk.Label(label="Request timeout (seconds):")
        timeout_label.set_xalign(0.0)
        grid.attach(timeout_label, 0, row, 1, 1)
        self.timeout_adjustment = Gtk.Adjustment(lower=5, upper=600, step_increment=5, page_increment=10, value=60)
        self.timeout_spin = Gtk.SpinButton(adjustment=self.timeout_adjustment, digits=0)
        self.timeout_spin.set_hexpand(True)
        grid.attach(self.timeout_spin, 1, row, 1, 1)

        row += 1
        retries_label = Gtk.Label(label="Max retries:")
        retries_label.set_xalign(0.0)
        grid.attach(retries_label, 0, row, 1, 1)
        self.retries_adjustment = Gtk.Adjustment(lower=0, upper=10, step_increment=1, page_increment=1, value=3)
        self.max_retries_spin = Gtk.SpinButton(adjustment=self.retries_adjustment, digits=0)
        self.max_retries_spin.set_hexpand(True)
        grid.attach(self.max_retries_spin, 1, row, 1, 1)

        row += 1
        delay_label = Gtk.Label(label="Retry delay (seconds):")
        delay_label.set_xalign(0.0)
        grid.attach(delay_label, 0, row, 1, 1)
        self.delay_adjustment = Gtk.Adjustment(lower=0, upper=120, step_increment=1, page_increment=5, value=5)
        self.retry_delay_spin = Gtk.SpinButton(adjustment=self.delay_adjustment, digits=0)
        self.retry_delay_spin.set_hexpand(True)
        grid.attach(self.retry_delay_spin, 1, row, 1, 1)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        container.append(button_box)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", self.on_cancel_clicked)
        button_box.append(cancel_button)

        save_button = Gtk.Button(label="Save")
        save_button.connect("clicked", self.on_save_clicked)
        button_box.append(save_button)

        self._available_models: List[str] = []
        self._load_models()
        self._load_settings()

    def _load_models(self) -> None:
        models: List[str] = []
        fetcher = getattr(self.ATLAS, "get_models_for_provider", None)
        if callable(fetcher):
            try:
                models = fetcher("Anthropic") or []
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to fetch Anthropic models: %s", exc, exc_info=True)
        self._available_models = []
        self.model_combo.remove_all()
        for name in models:
            if isinstance(name, str) and name.strip():
                cleaned = name.strip()
                if cleaned not in self._available_models:
                    self._available_models.append(cleaned)
                    self.model_combo.append_text(cleaned)

    def _load_settings(self) -> None:
        settings: Dict[str, object] = {}
        getter = getattr(self.ATLAS, "get_anthropic_settings", None)
        if callable(getter):
            try:
                settings = getter() or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to load Anthropic settings: %s", exc, exc_info=True)

        model = settings.get("model") if isinstance(settings, dict) else None
        if isinstance(model, str) and model.strip():
            active_model = model.strip()
        else:
            active_model = ""

        if active_model and active_model not in self._available_models:
            self.model_combo.append_text(active_model)
            self._available_models.append(active_model)

        if self._available_models:
            try:
                index = self._available_models.index(active_model) if active_model else 0
            except ValueError:
                index = 0
            self.model_combo.set_active(index)

        stream = settings.get("stream") if isinstance(settings, dict) else True
        self.streaming_toggle.set_active(bool(stream))

        function_calling = settings.get("function_calling") if isinstance(settings, dict) else False
        self.function_call_toggle.set_active(bool(function_calling))

        timeout = settings.get("timeout") if isinstance(settings, dict) else 60
        if isinstance(timeout, (int, float)) and timeout > 0:
            self.timeout_spin.set_value(int(timeout))

        max_retries = settings.get("max_retries") if isinstance(settings, dict) else 3
        if isinstance(max_retries, (int, float)) and max_retries >= 0:
            self.max_retries_spin.set_value(int(max_retries))

        retry_delay = settings.get("retry_delay") if isinstance(settings, dict) else 5
        if isinstance(retry_delay, (int, float)) and retry_delay >= 0:
            self.retry_delay_spin.set_value(int(retry_delay))

    def on_cancel_clicked(self, *_args) -> None:
        self.close()

    def on_save_clicked(self, *_args) -> None:
        selected_model = self.model_combo.get_active_text()
        if not selected_model:
            self._show_message("Error", "Select a model before saving.", Gtk.MessageType.ERROR)
            return

        payload = {
            "model": selected_model,
            "stream": self.streaming_toggle.get_active(),
            "function_calling": self.function_call_toggle.get_active(),
            "timeout": self.timeout_spin.get_value_as_int(),
            "max_retries": self.max_retries_spin.get_value_as_int(),
            "retry_delay": self.retry_delay_spin.get_value_as_int(),
        }

        setter = getattr(self.ATLAS, "set_anthropic_settings", None)
        if not callable(setter):
            self._show_message(
                "Error",
                "Anthropic settings are not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        try:
            result = setter(**payload)
        except Exception as exc:
            message = str(exc) or "Failed to save Anthropic settings."
            self._show_message("Error", message, Gtk.MessageType.ERROR)
            return

        if isinstance(result, dict) and result.get("success"):
            message = result.get("message") or "Anthropic settings saved."
            self._show_message("Success", message, Gtk.MessageType.INFO)
            self.close()
            return

        detail = "Failed to save Anthropic settings."
        if isinstance(result, dict):
            detail = result.get("error", detail)
        elif result is not None:
            detail = str(result)
        self._show_message("Error", detail, Gtk.MessageType.ERROR)

    def _show_message(self, title: str, message: str, message_type: Gtk.MessageType) -> None:
        self._last_message = (title, message, message_type)

        try:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                modal=True,
                message_type=message_type,
                buttons=Gtk.ButtonsType.OK,
                text=title,
            )
        except Exception:  # pragma: no cover - defensive fallback for stubbed Gtk
            logger.debug("Unable to create GTK message dialog for '%s'.", title)
            return

        if hasattr(dialog, "set_secondary_text"):
            dialog.set_secondary_text(message)
        else:
            dialog.secondary_text = message

        dialog.connect("response", lambda dlg, *_: dlg.destroy())
        GLib.idle_add(dialog.present)
