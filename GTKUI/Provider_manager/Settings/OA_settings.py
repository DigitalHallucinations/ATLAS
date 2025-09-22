"""Dedicated GTK window for configuring OpenAI provider defaults."""

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency for tests
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover - gracefully degrade if SDK missing
    OpenAI = None

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
        self.set_default_size(460, 360)

        self._last_message: Optional[Tuple[str, str, Gtk.MessageType]] = None
        self._stored_base_url: Optional[str] = None
        self._current_settings: Dict[str, Any] = {}
        self._available_models: List[str] = []
        self._api_key_visible = False
        self._base_url_is_valid = True

        main_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        self.set_child(main_box)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        main_box.append(grid)

        row = 0
        api_label = Gtk.Label(label="OpenAI API Key:")
        api_label.set_xalign(0.0)
        grid.attach(api_label, 0, row, 1, 1)

        api_entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        grid.attach(api_entry_box, 1, row, 1, 1)

        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_hexpand(True)
        if hasattr(self.api_key_entry, "set_visibility"):
            self.api_key_entry.set_visibility(False)
        if hasattr(self.api_key_entry, "set_invisible_char"):
            self.api_key_entry.set_invisible_char("•")
        self.api_key_entry.set_placeholder_text("Enter your OpenAI API key")
        api_entry_box.append(self.api_key_entry)

        self.api_key_toggle = Gtk.Button(label="Show")
        self.api_key_toggle.connect("clicked", self._on_api_key_toggle_clicked)
        api_entry_box.append(self.api_key_toggle)

        row += 1
        self.api_key_status_label = Gtk.Label(label="")
        self.api_key_status_label.set_xalign(0.0)
        grid.attach(self.api_key_status_label, 0, row, 2, 1)

        row += 1
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
        self.temperature_adjustment = Gtk.Adjustment(
            lower=0.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.temperature_spin = Gtk.SpinButton(adjustment=self.temperature_adjustment, digits=2)
        self.temperature_spin.set_increments(0.05, 0.1)
        self.temperature_spin.set_hexpand(True)
        grid.attach(self.temperature_spin, 1, row, 1, 1)

        row += 1
        top_p_label = Gtk.Label(label="Top-p:")
        top_p_label.set_xalign(0.0)
        grid.attach(top_p_label, 0, row, 1, 1)
        self.top_p_adjustment = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.01, page_increment=0.05, value=1.0
        )
        self.top_p_spin = Gtk.SpinButton(adjustment=self.top_p_adjustment, digits=2)
        self.top_p_spin.set_increments(0.01, 0.05)
        self.top_p_spin.set_hexpand(True)
        grid.attach(self.top_p_spin, 1, row, 1, 1)

        row += 1
        freq_label = Gtk.Label(label="Frequency Penalty:")
        freq_label.set_xalign(0.0)
        grid.attach(freq_label, 0, row, 1, 1)
        self.frequency_penalty_adjustment = Gtk.Adjustment(
            lower=-2.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.frequency_penalty_spin = Gtk.SpinButton(
            adjustment=self.frequency_penalty_adjustment, digits=2
        )
        self.frequency_penalty_spin.set_increments(0.05, 0.1)
        self.frequency_penalty_spin.set_hexpand(True)
        grid.attach(self.frequency_penalty_spin, 1, row, 1, 1)

        row += 1
        presence_label = Gtk.Label(label="Presence Penalty:")
        presence_label.set_xalign(0.0)
        grid.attach(presence_label, 0, row, 1, 1)
        self.presence_penalty_adjustment = Gtk.Adjustment(
            lower=-2.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.presence_penalty_spin = Gtk.SpinButton(
            adjustment=self.presence_penalty_adjustment, digits=2
        )
        self.presence_penalty_spin.set_increments(0.05, 0.1)
        self.presence_penalty_spin.set_hexpand(True)
        grid.attach(self.presence_penalty_spin, 1, row, 1, 1)

        row += 1
        tokens_label = Gtk.Label(label="Max Tokens:")
        tokens_label.set_xalign(0.0)
        grid.attach(tokens_label, 0, row, 1, 1)
        self.max_tokens_adjustment = Gtk.Adjustment(
            lower=1, upper=128000, step_increment=128, page_increment=512, value=4000
        )
        self.max_tokens_spin = Gtk.SpinButton(adjustment=self.max_tokens_adjustment, digits=0)
        self.max_tokens_spin.set_increments(128, 512)
        self.max_tokens_spin.set_hexpand(True)
        grid.attach(self.max_tokens_spin, 1, row, 1, 1)

        row += 1
        self.stream_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.stream_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.stream_toggle, 0, row, 2, 1)

        row += 1
        org_label = Gtk.Label(label="Organization (optional):")
        org_label.set_xalign(0.0)
        grid.attach(org_label, 0, row, 1, 1)
        self.organization_entry = Gtk.Entry()
        self.organization_entry.set_hexpand(True)
        self.organization_entry.set_placeholder_text("org-1234")
        grid.attach(self.organization_entry, 1, row, 1, 1)

        expander_cls = getattr(Gtk, "Expander", None)
        advanced_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        advanced_box.set_margin_start(6)
        advanced_box.set_margin_end(6)
        advanced_box.set_margin_bottom(6)

        if expander_cls is not None:
            advanced_container = expander_cls(label="Advanced")
            advanced_container.set_hexpand(True)
            advanced_container.set_margin_top(6)
            advanced_container.set_child(advanced_box)
            main_box.append(advanced_container)
        else:  # pragma: no cover - fallback for GTK stubs used in testing
            fallback_wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            fallback_wrapper.set_margin_top(6)
            main_box.append(fallback_wrapper)

            header = Gtk.Label(label="Advanced")
            header.set_xalign(0.0)
            fallback_wrapper.append(header)
            fallback_wrapper.append(advanced_box)

        advanced_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        advanced_box.append(advanced_grid)

        base_url_label = Gtk.Label(label="Custom Base URL:")
        base_url_label.set_xalign(0.0)
        advanced_grid.attach(base_url_label, 0, 0, 1, 1)

        self.base_url_entry = Gtk.Entry()
        self.base_url_entry.set_hexpand(True)
        self.base_url_entry.set_placeholder_text("https://api.openai.com/v1")
        if hasattr(self.base_url_entry, "connect"):
            self.base_url_entry.connect("changed", self._on_base_url_changed)
        advanced_grid.attach(self.base_url_entry, 1, 0, 1, 1)

        self.base_url_feedback_label = Gtk.Label(label="Leave blank to use the official endpoint.")
        self.base_url_feedback_label.set_xalign(0.0)
        if hasattr(self.base_url_feedback_label, "add_css_class"):
            self.base_url_feedback_label.add_css_class("dim-label")
        advanced_box.append(self.base_url_feedback_label)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        button_box.set_halign(Gtk.Align.END)
        main_box.append(button_box)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", lambda *_: self.close())
        button_box.append(cancel_button)

        save_key_button = Gtk.Button(label="Save API Key")
        save_key_button.connect("clicked", self.on_save_api_key_clicked)
        button_box.append(save_key_button)

        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)
        button_box.append(save_button)

        self._populate_defaults()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _populate_defaults(self) -> None:
        settings = self._get_settings_snapshot()
        self._current_settings = dict(settings)
        self._stored_base_url = settings.get("base_url")
        self._apply_base_url_to_entry(self._stored_base_url)

        self._refresh_api_key_status()

        self.temperature_spin.set_value(float(settings.get("temperature", 0.0)))
        self.top_p_spin.set_value(float(settings.get("top_p", 1.0)))
        self.frequency_penalty_spin.set_value(float(settings.get("frequency_penalty", 0.0)))
        self.presence_penalty_spin.set_value(float(settings.get("presence_penalty", 0.0)))
        self.max_tokens_spin.set_value(float(settings.get("max_tokens", 4000)))
        self.stream_toggle.set_active(bool(settings.get("stream", True)))
        self.organization_entry.set_text(settings.get("organization") or "")

        self._begin_model_refresh(settings)

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

    def _refresh_api_key_status(self) -> None:
        try:
            has_key = self.config_manager.has_provider_api_key("OpenAI")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to determine OpenAI API key status: %s", exc, exc_info=True)
            status_text = "Unable to determine API key status."
        else:
            status_text = (
                "An API key is saved for OpenAI." if has_key else "No API key saved for OpenAI."
            )

        if hasattr(self.api_key_status_label, "set_label"):
            self.api_key_status_label.set_label(status_text)
        else:  # pragma: no cover - testing stubs
            self.api_key_status_label.label = status_text

    def _apply_base_url_to_entry(self, value: Optional[str]) -> None:
        if hasattr(self, "base_url_entry"):
            if value:
                self.base_url_entry.set_text(value)
            else:
                self.base_url_entry.set_text("")
        self._sync_base_url_state()

    def _sanitize_base_url(self, raw: str) -> Tuple[Optional[str], bool]:
        text = (raw or "").strip()
        if not text:
            return None, True

        parsed = urlparse(text)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return text, True

        return None, False

    def _update_base_url_feedback(self, valid: bool) -> None:
        hint = (
            "Leave blank to use the official endpoint."
            if valid
            else "Enter a valid HTTP(S) URL or leave blank."
        )

        if hasattr(self.base_url_entry, "set_tooltip_text"):
            self.base_url_entry.set_tooltip_text(hint)

        if hasattr(self.base_url_entry, "add_css_class"):
            if valid:
                self.base_url_entry.remove_css_class("error")
            else:
                self.base_url_entry.add_css_class("error")

        if hasattr(self.base_url_feedback_label, "set_label"):
            self.base_url_feedback_label.set_label(hint)
        else:  # pragma: no cover - testing stubs
            self.base_url_feedback_label.label = hint

        if hasattr(self.base_url_feedback_label, "add_css_class"):
            if valid:
                self.base_url_feedback_label.remove_css_class("error")
            else:
                self.base_url_feedback_label.add_css_class("error")

        self._base_url_is_valid = valid

    def _sync_base_url_state(self) -> Tuple[Optional[str], bool]:
        if hasattr(self, "base_url_entry"):
            raw = self.base_url_entry.get_text()
        else:
            raw = ""

        sanitized, valid = self._sanitize_base_url(raw)
        self._stored_base_url = sanitized
        self._update_base_url_feedback(valid)
        return sanitized, valid

    def _on_base_url_changed(self, _entry: Gtk.Entry) -> None:
        self._sync_base_url_state()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _begin_model_refresh(self, settings: Dict[str, Any]) -> None:
        placeholder = settings.get("model") or "Loading models…"
        self.model_combo.remove_all()
        self.model_combo.append_text(placeholder)
        self.model_combo.set_active(0)

        self._start_background_task(self._load_models_thread, dict(settings))

    def _start_background_task(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return thread

    def _load_models_thread(self, settings: Dict[str, Any]) -> None:
        models, error = self._query_available_models(settings)
        GLib.idle_add(self._apply_model_results, models, error, settings.get("model"))

    def _query_available_models(self, settings: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
        if OpenAI is None:
            return [], "The OpenAI Python SDK is not installed."

        api_key = self.config_manager.get_openai_api_key()
        if not api_key:
            return [], "Enter and save your OpenAI API key to load models."

        client_kwargs: Dict[str, Any] = {"api_key": api_key}

        base_url = self._stored_base_url
        if base_url:
            client_kwargs["base_url"] = base_url

        organization = settings.get("organization")
        if organization:
            client_kwargs["organization"] = organization

        try:
            client = OpenAI(**client_kwargs)
            response = client.models.list()
        except Exception as exc:
            logger.error("Failed to retrieve OpenAI models: %s", exc, exc_info=True)
            return [], str(exc)

        models: List[str] = []
        for entry in getattr(response, "data", []):
            model_id = getattr(entry, "id", None)
            if model_id:
                models.append(model_id)

        models = sorted(set(models))

        prioritized = [
            name
            for name in models
            if any(token in name for token in ("gpt", "omni", "o1", "o3", "chat"))
        ]
        if prioritized:
            models = prioritized

        return models, None

    def _apply_model_results(
        self, models: List[str], error: Optional[str], preferred_model: Optional[str]
    ) -> bool:
        if error:
            logger.warning("Falling back to stored model because fetching failed: %s", error)
            fallback = preferred_model or "gpt-4o"
            self.model_combo.remove_all()
            self.model_combo.append_text(fallback)
            self.model_combo.set_active(0)
            self._show_message("Model Load Failed", error, Gtk.MessageType.ERROR)
            self._available_models = [fallback]
            return False

        if preferred_model and preferred_model not in models:
            models = [preferred_model] + [name for name in models if name != preferred_model]

        if not models:
            models = [preferred_model or "gpt-4o"]

        self.model_combo.remove_all()
        active_index = 0
        for idx, name in enumerate(models):
            self.model_combo.append_text(name)
            if preferred_model and name == preferred_model:
                active_index = idx

        self.model_combo.set_active(active_index)
        self._available_models = list(models)
        return False

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_api_key_toggle_clicked(self, _button: Gtk.Button) -> None:
        self._api_key_visible = not self._api_key_visible

        if hasattr(self.api_key_entry, "set_visibility"):
            self.api_key_entry.set_visibility(self._api_key_visible)
        else:  # pragma: no cover - testing stubs
            self.api_key_entry.visibility = self._api_key_visible

        label = "Hide" if self._api_key_visible else "Show"
        if hasattr(self.api_key_toggle, "set_label"):
            self.api_key_toggle.set_label(label)
        else:  # pragma: no cover - testing stubs
            self.api_key_toggle.label = label

    def on_save_api_key_clicked(self, _button: Gtk.Button) -> None:
        api_key = (self.api_key_entry.get_text() or "").strip()
        if not api_key:
            self._show_message("Error", "Enter an API key before saving.", Gtk.MessageType.ERROR)
            return

        self._start_background_task(self._persist_api_key, api_key)

    def _persist_api_key(self, api_key: str) -> None:
        try:
            result = asyncio.run(self.ATLAS.update_provider_api_key("OpenAI", api_key))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error saving OpenAI API key: %s", exc, exc_info=True)
            GLib.idle_add(self._handle_api_key_save_result, {"success": False, "error": str(exc)})
            return

        GLib.idle_add(self._handle_api_key_save_result, result)

    def _handle_api_key_save_result(self, result: Dict[str, Any]) -> bool:
        if isinstance(result, dict) and result.get("success"):
            message = result.get("message", "API key saved.")
            self._show_message("Success", message, Gtk.MessageType.INFO)
            self.api_key_entry.set_text("")
            self._refresh_api_key_status()
            refreshed = self._get_settings_snapshot()
            self._stored_base_url = refreshed.get("base_url")
            self._apply_base_url_to_entry(self._stored_base_url)
            self._begin_model_refresh(refreshed)
        else:
            if isinstance(result, dict):
                detail = result.get("error") or result.get("message") or "Unable to save API key."
            else:
                detail = str(result)
            self._show_message("Error", detail, Gtk.MessageType.ERROR)

        return False

    def on_save_clicked(self, _button: Gtk.Button):
        model = self.model_combo.get_active_text()
        if not model:
            self._show_message(
                "Error", "Please choose a default model before saving.", Gtk.MessageType.ERROR
            )
            return

        base_url, base_valid = self._sync_base_url_state()
        if not base_valid:
            self._show_message(
                "Error",
                "Enter a valid HTTP(S) base URL or leave the field blank.",
                Gtk.MessageType.ERROR,
            )
            return

        payload = {
            "model": model,
            "temperature": self.temperature_spin.get_value(),
            "top_p": self.top_p_spin.get_value(),
            "frequency_penalty": self.frequency_penalty_spin.get_value(),
            "presence_penalty": self.presence_penalty_spin.get_value(),
            "max_tokens": self.max_tokens_spin.get_value_as_int(),
            "stream": self.stream_toggle.get_active(),
            "base_url": base_url,
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
