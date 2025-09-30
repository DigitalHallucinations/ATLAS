from __future__ import annotations

"""GTK window for configuring Anthropic provider defaults."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.utils import create_box
from modules.background_tasks import run_async_in_thread

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
        self.set_default_size(480, 360)

        self._last_message: Optional[Tuple[str, str, Gtk.MessageType]] = None

        container = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        self.set_child(container)

        notebook = Gtk.Notebook()
        notebook.set_hexpand(True)
        notebook.set_vexpand(True)
        container.append(notebook)

        general_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        general_grid.set_margin_top(6)
        general_grid.set_margin_bottom(6)
        general_grid.set_margin_start(6)
        general_grid.set_margin_end(6)
        notebook.append_page(general_grid, Gtk.Label(label="General"))

        advanced_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        advanced_grid.set_margin_top(6)
        advanced_grid.set_margin_bottom(6)
        advanced_grid.set_margin_start(6)
        advanced_grid.set_margin_end(6)
        advanced_scroller = Gtk.ScrolledWindow()
        advanced_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        advanced_scroller.set_child(advanced_grid)
        notebook.append_page(advanced_scroller, Gtk.Label(label="Advanced"))

        self._api_key_visible = False
        self._default_api_key_placeholder = "Enter your Anthropic API key"
        self._defaults: Dict[str, object] = {
            "model": "claude-3-opus-20240229",
            "stream": True,
            "function_calling": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "max_output_tokens": None,
            "timeout": 60,
            "max_retries": 3,
            "retry_delay": 5,
            "stop_sequences": [],
            "tool_choice": "auto",
            "tool_choice_name": None,
            "metadata": {},
            "thinking": False,
            "thinking_budget": None,
        }
        self._initial_settings: Dict[str, object] = {}

        row = 0
        api_label = Gtk.Label(label="Anthropic API Key:")
        api_label.set_xalign(0.0)
        general_grid.attach(api_label, 0, row, 1, 1)

        api_entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        general_grid.attach(api_entry_box, 1, row, 1, 1)

        self.api_key_entry = Gtk.Entry()
        self.api_key_entry.set_hexpand(True)
        if hasattr(self.api_key_entry, "set_visibility"):
            self.api_key_entry.set_visibility(False)
        if hasattr(self.api_key_entry, "set_invisible_char"):
            self.api_key_entry.set_invisible_char("•")
        self.api_key_entry.set_placeholder_text(self._default_api_key_placeholder)
        self.api_key_entry.set_tooltip_text("Enter your Anthropic API key.")
        api_entry_box.append(self.api_key_entry)

        self.api_key_toggle = Gtk.Button(label="Show")
        self.api_key_toggle.connect("clicked", self._on_api_key_toggle_clicked)
        api_entry_box.append(self.api_key_toggle)

        row += 1
        self.api_key_status_label = Gtk.Label(label="")
        self.api_key_status_label.set_xalign(0.0)
        general_grid.attach(self.api_key_status_label, 0, row, 2, 1)

        row += 1
        model_label = Gtk.Label(label="Default Model:")
        model_label.set_xalign(0.0)
        general_grid.attach(model_label, 0, row, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        if hasattr(self.model_combo, "connect"):
            self.model_combo.connect("changed", self._update_save_button_state)
        general_grid.attach(self.model_combo, 1, row, 1, 1)

        row += 1
        self.streaming_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.streaming_toggle.set_halign(Gtk.Align.START)
        if hasattr(self.streaming_toggle, "connect"):
            self.streaming_toggle.connect("toggled", self._update_save_button_state)
        general_grid.attach(self.streaming_toggle, 0, row, 2, 1)

        row += 1
        self.function_call_toggle = Gtk.CheckButton(label="Enable tool/function calling")
        self.function_call_toggle.set_halign(Gtk.Align.START)
        if hasattr(self.function_call_toggle, "connect"):
            self.function_call_toggle.connect("toggled", self._update_save_button_state)
            self.function_call_toggle.connect("toggled", self._update_tool_choice_state)
        general_grid.attach(self.function_call_toggle, 0, row, 2, 1)

        row += 1
        tool_choice_label = Gtk.Label(label="Tool behaviour:")
        tool_choice_label.set_xalign(0.0)
        general_grid.attach(tool_choice_label, 0, row, 1, 1)

        self.tool_choice_combo = Gtk.ComboBoxText()
        self.tool_choice_combo.set_hexpand(True)
        self._tool_choice_map = [
            ("auto", "Automatic (Anthropic decides)"),
            ("any", "Require any tool call"),
            ("none", "Never call tools"),
            ("tool", "Request specific tool"),
        ]
        self._cached_tool_choice_value = "auto"
        for key, label in self._tool_choice_map:
            self.tool_choice_combo.append_text(label)
        if hasattr(self.tool_choice_combo, "connect"):
            self.tool_choice_combo.connect("changed", self._update_tool_choice_state)
            self.tool_choice_combo.connect("changed", self._update_save_button_state)
        general_grid.attach(self.tool_choice_combo, 1, row, 1, 1)

        row += 1
        preferred_tool_label = Gtk.Label(label="Preferred tool name:")
        preferred_tool_label.set_xalign(0.0)
        general_grid.attach(preferred_tool_label, 0, row, 1, 1)

        self.tool_name_entry = Gtk.Entry()
        self.tool_name_entry.set_hexpand(True)
        self.tool_name_entry.set_placeholder_text("Optional tool name")
        if hasattr(self.tool_name_entry, "connect"):
            self.tool_name_entry.connect("changed", self._update_save_button_state)
            self.tool_name_entry.connect("changed", self._update_tool_choice_state)
        general_grid.attach(self.tool_name_entry, 1, row, 1, 1)

        advanced_row = 0
        temperature_label = Gtk.Label(label="Sampling temperature:")
        temperature_label.set_xalign(0.0)
        advanced_grid.attach(temperature_label, 0, advanced_row, 1, 1)
        self.temperature_adjustment = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.temperature_spin = Gtk.SpinButton(
            adjustment=self.temperature_adjustment, digits=2
        )
        self.temperature_spin.set_increments(0.05, 0.1)
        self.temperature_spin.set_hexpand(True)
        if hasattr(self.temperature_spin, "connect"):
            self.temperature_spin.connect("value-changed", self._update_save_button_state)
        advanced_grid.attach(self.temperature_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        top_p_label = Gtk.Label(label="Top-p (nucleus sampling):")
        top_p_label.set_xalign(0.0)
        advanced_grid.attach(top_p_label, 0, advanced_row, 1, 1)
        self.top_p_adjustment = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.05, page_increment=0.1, value=1.0
        )
        self.top_p_spin = Gtk.SpinButton(adjustment=self.top_p_adjustment, digits=2)
        self.top_p_spin.set_increments(0.05, 0.1)
        self.top_p_spin.set_hexpand(True)
        if hasattr(self.top_p_spin, "connect"):
            self.top_p_spin.connect("value-changed", self._update_save_button_state)
        advanced_grid.attach(self.top_p_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        top_k_label = Gtk.Label(label="Top-k (0 = default):")
        top_k_label.set_xalign(0.0)
        advanced_grid.attach(top_k_label, 0, advanced_row, 1, 1)
        self.top_k_adjustment = Gtk.Adjustment(
            lower=0, upper=500, step_increment=1, page_increment=10, value=0
        )
        self.top_k_spin = Gtk.SpinButton(
            adjustment=self.top_k_adjustment, digits=0
        )
        self.top_k_spin.set_hexpand(True)
        if hasattr(self.top_k_spin, "connect"):
            self.top_k_spin.connect("value-changed", self._update_save_button_state)
        advanced_grid.attach(self.top_k_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        max_output_label = Gtk.Label(label="Max output tokens (0 = auto):")
        max_output_label.set_xalign(0.0)
        advanced_grid.attach(max_output_label, 0, advanced_row, 1, 1)
        self.max_output_adjustment = Gtk.Adjustment(
            lower=0, upper=32000, step_increment=64, page_increment=512, value=0
        )
        self.max_output_tokens_spin = Gtk.SpinButton(
            adjustment=self.max_output_adjustment, digits=0
        )
        self.max_output_tokens_spin.set_hexpand(True)
        if hasattr(self.max_output_tokens_spin, "connect"):
            self.max_output_tokens_spin.connect(
                "value-changed", self._update_save_button_state
            )
        advanced_grid.attach(self.max_output_tokens_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        stop_sequences_label = Gtk.Label(label="Stop sequences:")
        stop_sequences_label.set_xalign(0.0)
        advanced_grid.attach(stop_sequences_label, 0, advanced_row, 1, 1)

        self.stop_sequences_entry = Gtk.Entry()
        self.stop_sequences_entry.set_hexpand(True)
        self.stop_sequences_entry.set_placeholder_text(
            "Separate sequences with commas (max 4)."
        )
        if hasattr(self.stop_sequences_entry, "connect"):
            self.stop_sequences_entry.connect("changed", self._update_save_button_state)
        advanced_grid.attach(self.stop_sequences_entry, 1, advanced_row, 1, 1)

        advanced_row += 1
        stop_sequences_help = Gtk.Label(
            label="Example: END, <|stop|>"
        )
        stop_sequences_help.set_xalign(0.0)
        if hasattr(stop_sequences_help, "add_css_class"):
            stop_sequences_help.add_css_class("dim-label")
        advanced_grid.attach(stop_sequences_help, 0, advanced_row, 2, 1)

        advanced_row += 1
        metadata_label = Gtk.Label(label="Request metadata:")
        metadata_label.set_xalign(0.0)
        advanced_grid.attach(metadata_label, 0, advanced_row, 1, 1)

        self.metadata_entry = Gtk.Entry()
        self.metadata_entry.set_hexpand(True)
        self.metadata_entry.set_placeholder_text("key=value, team=research")
        if hasattr(self.metadata_entry, "connect"):
            self.metadata_entry.connect("changed", self._update_save_button_state)
        advanced_grid.attach(self.metadata_entry, 1, advanced_row, 1, 1)

        advanced_row += 1
        metadata_help = Gtk.Label(
            label="Comma separated pairs or JSON. Max 16 entries."
        )
        metadata_help.set_xalign(0.0)
        if hasattr(metadata_help, "add_css_class"):
            metadata_help.add_css_class("dim-label")
        advanced_grid.attach(metadata_help, 0, advanced_row, 2, 1)

        advanced_row += 1
        timeout_label = Gtk.Label(label="Request timeout (seconds):")
        timeout_label.set_xalign(0.0)
        advanced_grid.attach(timeout_label, 0, advanced_row, 1, 1)
        self.timeout_adjustment = Gtk.Adjustment(lower=5, upper=600, step_increment=5, page_increment=10, value=60)
        self.timeout_spin = Gtk.SpinButton(adjustment=self.timeout_adjustment, digits=0)
        self.timeout_spin.set_hexpand(True)
        if hasattr(self.timeout_spin, "connect"):
            self.timeout_spin.connect("value-changed", self._update_save_button_state)
        advanced_grid.attach(self.timeout_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        retries_label = Gtk.Label(label="Additional retries (after first attempt):")
        retries_label.set_xalign(0.0)
        advanced_grid.attach(retries_label, 0, advanced_row, 1, 1)
        self.retries_adjustment = Gtk.Adjustment(lower=0, upper=10, step_increment=1, page_increment=1, value=3)
        self.max_retries_spin = Gtk.SpinButton(adjustment=self.retries_adjustment, digits=0)
        self.max_retries_spin.set_hexpand(True)
        if hasattr(self.max_retries_spin, "connect"):
            self.max_retries_spin.connect("value-changed", self._update_save_button_state)
        advanced_grid.attach(self.max_retries_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        delay_label = Gtk.Label(label="Retry delay (seconds):")
        delay_label.set_xalign(0.0)
        advanced_grid.attach(delay_label, 0, advanced_row, 1, 1)
        self.delay_adjustment = Gtk.Adjustment(lower=0, upper=120, step_increment=1, page_increment=5, value=5)
        self.retry_delay_spin = Gtk.SpinButton(adjustment=self.delay_adjustment, digits=0)
        self.retry_delay_spin.set_hexpand(True)
        if hasattr(self.retry_delay_spin, "connect"):
            self.retry_delay_spin.connect("value-changed", self._update_save_button_state)
        advanced_grid.attach(self.retry_delay_spin, 1, advanced_row, 1, 1)

        advanced_row += 1
        thinking_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        thinking_box.set_halign(Gtk.Align.START)
        advanced_grid.attach(thinking_box, 0, advanced_row, 2, 1)

        self.thinking_toggle = Gtk.CheckButton(label="Enable Claude thinking outputs (beta)")
        if hasattr(self.thinking_toggle, "connect"):
            self.thinking_toggle.connect("toggled", self._update_thinking_controls_state)
            self.thinking_toggle.connect("toggled", self._update_save_button_state)
        thinking_box.append(self.thinking_toggle)

        self.thinking_budget_adjustment = Gtk.Adjustment(
            lower=0,
            upper=32768,
            step_increment=128,
            page_increment=512,
            value=0,
        )
        self.thinking_budget_spin = Gtk.SpinButton(
            adjustment=self.thinking_budget_adjustment,
            digits=0,
        )
        self.thinking_budget_spin.set_tooltip_text("Optional budget tokens when thinking is enabled.")
        if hasattr(self.thinking_budget_spin, "connect"):
            self.thinking_budget_spin.connect("value-changed", self._update_save_button_state)
        thinking_box.append(self.thinking_budget_spin)

        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        if hasattr(button_box, "set_halign"):
            button_box.set_halign(Gtk.Align.END)
        container.append(button_box)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", self.on_cancel_clicked)
        button_box.append(cancel_button)

        reset_button = Gtk.Button(label="Reset to defaults")
        reset_button.connect("clicked", self._on_reset_clicked)
        button_box.append(reset_button)
        self.reset_button = reset_button

        save_key_button = Gtk.Button(label="Save API Key")
        save_key_button.connect("clicked", self.on_save_api_key_clicked)
        button_box.append(save_key_button)
        self.save_key_button = save_key_button

        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)
        if hasattr(save_button, "set_sensitive"):
            save_button.set_sensitive(False)
        else:  # pragma: no cover - testing stubs
            save_button.sensitive = False
        button_box.append(save_button)
        self.save_button = save_button

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
        active_model = model.strip() if isinstance(model, str) and model.strip() else ""
        self._activate_model(active_model or str(self._defaults["model"]))

        stream = settings.get("stream") if isinstance(settings, dict) else True
        self.streaming_toggle.set_active(bool(stream))

        function_calling = settings.get("function_calling") if isinstance(settings, dict) else False
        self.function_call_toggle.set_active(bool(function_calling))

        temperature = settings.get("temperature") if isinstance(settings, dict) else self._defaults["temperature"]
        if isinstance(temperature, (int, float)):
            self.temperature_spin.set_value(float(temperature))

        top_p = settings.get("top_p") if isinstance(settings, dict) else self._defaults["top_p"]
        if isinstance(top_p, (int, float)):
            self.top_p_spin.set_value(float(top_p))

        top_k = settings.get("top_k") if isinstance(settings, dict) else None
        if isinstance(top_k, (int, float)) and top_k > 0:
            self.top_k_spin.set_value(int(top_k))
        else:
            self.top_k_spin.set_value(0)

        max_output_tokens = (
            settings.get("max_output_tokens") if isinstance(settings, dict) else None
        )
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            self.max_output_tokens_spin.set_value(int(max_output_tokens))
        else:
            self.max_output_tokens_spin.set_value(0)

        stop_sequences = settings.get("stop_sequences") if isinstance(settings, dict) else []
        if isinstance(stop_sequences, str):
            stop_sequences_list = [part.strip() for part in stop_sequences.split(",") if part.strip()]
        elif isinstance(stop_sequences, (list, tuple, set)):
            stop_sequences_list = [str(item).strip() for item in stop_sequences if str(item).strip()]
        else:
            stop_sequences_list = []
        self.stop_sequences_entry.set_text(", ".join(stop_sequences_list))

        tool_choice = settings.get("tool_choice") if isinstance(settings, dict) else self._defaults["tool_choice"]
        tool_choice_name = (
            settings.get("tool_choice_name") if isinstance(settings, dict) else self._defaults["tool_choice_name"]
        )
        self._set_tool_choice_widgets(str(tool_choice or "auto"), tool_choice_name)

        metadata = settings.get("metadata") if isinstance(settings, dict) else self._defaults["metadata"]
        self.metadata_entry.set_text(self._format_metadata_for_entry(metadata))

        timeout = settings.get("timeout") if isinstance(settings, dict) else 60
        if isinstance(timeout, (int, float)) and timeout > 0:
            self.timeout_spin.set_value(int(timeout))

        max_retries = settings.get("max_retries") if isinstance(settings, dict) else 3
        if isinstance(max_retries, (int, float)) and max_retries >= 0:
            self.max_retries_spin.set_value(int(max_retries))

        retry_delay = settings.get("retry_delay") if isinstance(settings, dict) else 5
        if isinstance(retry_delay, (int, float)) and retry_delay >= 0:
            self.retry_delay_spin.set_value(int(retry_delay))

        thinking_enabled = settings.get("thinking") if isinstance(settings, dict) else self._defaults["thinking"]
        self.thinking_toggle.set_active(bool(thinking_enabled))

        thinking_budget = settings.get("thinking_budget") if isinstance(settings, dict) else self._defaults["thinking_budget"]
        if isinstance(thinking_budget, (int, float)) and thinking_budget > 0:
            self.thinking_budget_spin.set_value(int(thinking_budget))
        else:
            self.thinking_budget_spin.set_value(0)
        self._update_thinking_controls_state()

        self._initial_settings = {
            "model": self.model_combo.get_active_text(),
            "stream": self.streaming_toggle.get_active(),
            "function_calling": self.function_call_toggle.get_active(),
            "temperature": round(self.temperature_spin.get_value(), 2),
            "top_p": round(self.top_p_spin.get_value(), 2),
            "top_k": (
                self.top_k_spin.get_value_as_int()
                if self.top_k_spin.get_value_as_int() > 0
                else None
            ),
            "max_output_tokens": (
                self.max_output_tokens_spin.get_value_as_int()
                if self.max_output_tokens_spin.get_value_as_int() > 0
                else None
            ),
            "stop_sequences": stop_sequences_list,
            "timeout": self.timeout_spin.get_value_as_int(),
            "max_retries": self.max_retries_spin.get_value_as_int(),
            "retry_delay": self.retry_delay_spin.get_value_as_int(),
            "tool_choice": self._get_tool_choice_value(),
            "tool_choice_name": self._get_tool_choice_name(),
            "metadata": self._parse_metadata_entry(self.metadata_entry.get_text()),
            "thinking": self.thinking_toggle.get_active(),
            "thinking_budget": (
                self.thinking_budget_spin.get_value_as_int()
                if self.thinking_budget_spin.get_value_as_int() > 0
                else None
            ),
        }

        self._refresh_api_key_status()
        self._update_save_button_state()

    def on_cancel_clicked(self, *_args) -> None:
        self.close()

    def on_save_api_key_clicked(self, *_args) -> None:
        api_key = (self.api_key_entry.get_text() or "").strip()
        if not api_key:
            self._show_message("Error", "Enter an API key before saving.", Gtk.MessageType.ERROR)
            return

        self._set_save_key_button_sensitive(False)

        def handle_success(result):
            GLib.idle_add(self._handle_api_key_save_result, result)

        def handle_error(exc: Exception) -> None:
            logger.error("Error saving Anthropic API key: %s", exc, exc_info=True)
            GLib.idle_add(
                self._handle_api_key_save_result,
                {"success": False, "error": str(exc)},
            )

        helper = getattr(self.ATLAS, "update_provider_api_key_in_background", None)
        if callable(helper):
            try:
                helper(
                    "Anthropic",
                    api_key,
                    on_success=handle_success,
                    on_error=handle_error,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Unable to schedule Anthropic API key save: %s", exc, exc_info=True)
                self._show_message(
                    "Error",
                    f"Failed to save API key: {str(exc)}",
                    Gtk.MessageType.ERROR,
                )
                self._set_save_key_button_sensitive(True)
            return

        updater = getattr(self.ATLAS, "update_provider_api_key", None)
        if not callable(updater):
            self._show_message(
                "Error",
                "Saving API keys is not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            self._set_save_key_button_sensitive(True)
            return

        try:
            run_async_in_thread(
                lambda: updater("Anthropic", api_key),
                on_success=handle_success,
                on_error=handle_error,
                logger=logger,
                thread_name="anthropic-api-key-fallback",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to start fallback API key save task: %s", exc, exc_info=True)
            self._show_message(
                "Error",
                f"Failed to save API key: {str(exc)}",
                Gtk.MessageType.ERROR,
            )
            self._set_save_key_button_sensitive(True)

    def on_save_clicked(self, *_args) -> None:
        payload = self._collect_payload()
        if payload is None:
            self._show_message("Error", "Select a model before saving.", Gtk.MessageType.ERROR)
            return

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
            self._initial_settings = dict(payload)
            self._update_save_button_state()
            self.close()
            return

        detail = "Failed to save Anthropic settings."
        if isinstance(result, dict):
            detail = result.get("error", detail)
        elif result is not None:
            detail = str(result)
        self._show_message("Error", detail, Gtk.MessageType.ERROR)

    def _handle_api_key_save_result(self, result: Dict[str, object]) -> bool:
        self._set_save_key_button_sensitive(True)
        if isinstance(result, dict) and result.get("success"):
            message = result.get("message") or "API key saved."
            self._show_message("Success", message, Gtk.MessageType.INFO)
            self.api_key_entry.set_text("")
            self._refresh_api_key_status()
        else:
            if isinstance(result, dict):
                detail = result.get("error") or result.get("message") or "Unable to save API key."
            else:
                detail = str(result)
            self._show_message("Error", detail, Gtk.MessageType.ERROR)

        return False

    def _set_save_key_button_sensitive(self, enabled: bool) -> None:
        button = getattr(self, "save_key_button", None)
        if button is None:
            return

        if hasattr(button, "set_sensitive"):
            button.set_sensitive(enabled)
        try:  # pragma: no cover - testing stubs and GTK proxies
            button.sensitive = enabled
        except Exception:
            pass

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

    def _refresh_api_key_status(self) -> None:
        status_text = "Credential status is unavailable."
        placeholder = self._default_api_key_placeholder
        tooltip = "Enter your Anthropic API key."

        atlas = getattr(self, "ATLAS", None)
        if atlas is not None and hasattr(atlas, "get_provider_api_key_status"):
            try:
                payload = atlas.get_provider_api_key_status("Anthropic") or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Unable to determine Anthropic API key status: %s", exc, exc_info=True)
            else:
                has_key = bool(payload.get("has_key"))
                metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                hint = ""
                if isinstance(metadata, dict):
                    hint = metadata.get("hint") or ""

                if has_key:
                    suffix = f" ({hint})" if hint else ""
                    status_text = f"An API key is saved for Anthropic.{suffix}"
                    if hint:
                        placeholder = f"Saved key: {hint}"
                        tooltip = f"A saved Anthropic key is active ({hint}). Enter a new key to replace it."
                    else:
                        tooltip = "An Anthropic API key is saved. Enter a new key to replace it."
                else:
                    status_text = "No API key saved for Anthropic."
                    tooltip = "Enter your Anthropic API key."

        if hasattr(self.api_key_entry, "set_placeholder_text"):
            self.api_key_entry.set_placeholder_text(placeholder)
        if hasattr(self.api_key_entry, "set_tooltip_text"):
            self.api_key_entry.set_tooltip_text(tooltip)

        if hasattr(self.api_key_status_label, "set_label"):
            self.api_key_status_label.set_label(status_text)
        else:  # pragma: no cover - testing stubs
            self.api_key_status_label.label = status_text

    def _collect_payload(self) -> Optional[Dict[str, object]]:
        selected_model = self.model_combo.get_active_text()
        if not selected_model:
            return None

        return {
            "model": selected_model,
            "stream": self.streaming_toggle.get_active(),
            "function_calling": self.function_call_toggle.get_active(),
            "temperature": round(self.temperature_spin.get_value(), 2),
            "top_p": round(self.top_p_spin.get_value(), 2),
            "top_k": (
                self.top_k_spin.get_value_as_int()
                if self.top_k_spin.get_value_as_int() > 0
                else None
            ),
            "max_output_tokens": (
                self.max_output_tokens_spin.get_value_as_int()
                if self.max_output_tokens_spin.get_value_as_int() > 0
                else None
            ),
            "stop_sequences": self._parse_stop_sequences_entry(),
            "timeout": self.timeout_spin.get_value_as_int(),
            "max_retries": self.max_retries_spin.get_value_as_int(),
            "retry_delay": self.retry_delay_spin.get_value_as_int(),
            "tool_choice": self._get_tool_choice_value(),
            "tool_choice_name": self._get_tool_choice_name(),
            "metadata": self._parse_metadata_entry(self.metadata_entry.get_text()),
            "thinking": self.thinking_toggle.get_active(),
            "thinking_budget": (
                self.thinking_budget_spin.get_value_as_int()
                if self.thinking_budget_spin.get_value_as_int() > 0
                else None
            ),
        }

    def _activate_model(self, model: str) -> None:
        if model and model not in self._available_models:
            self.model_combo.append_text(model)
            self._available_models.append(model)

        if self._available_models:
            try:
                index = self._available_models.index(model) if model else 0
            except ValueError:
                index = 0
            self.model_combo.set_active(index)

    def _on_reset_clicked(self, *_args) -> None:
        defaults = dict(self._defaults)
        self._activate_model(str(defaults.get("model", "")))
        self.streaming_toggle.set_active(bool(defaults.get("stream", True)))
        self.function_call_toggle.set_active(bool(defaults.get("function_calling", False)))
        self.temperature_spin.set_value(float(defaults.get("temperature", 0.0)))
        self.top_p_spin.set_value(float(defaults.get("top_p", 1.0)))
        default_top_k = defaults.get("top_k")
        if isinstance(default_top_k, (int, float)) and default_top_k:
            self.top_k_spin.set_value(int(default_top_k))
        else:
            self.top_k_spin.set_value(0)
        default_max_output = defaults.get("max_output_tokens")
        if isinstance(default_max_output, (int, float)) and default_max_output:
            self.max_output_tokens_spin.set_value(int(default_max_output))
        else:
            self.max_output_tokens_spin.set_value(0)
        default_stop_sequences = defaults.get("stop_sequences")
        if isinstance(default_stop_sequences, str):
            defaults_list = [part.strip() for part in default_stop_sequences.split(",") if part.strip()]
        elif isinstance(default_stop_sequences, (list, tuple, set)):
            defaults_list = [str(item).strip() for item in default_stop_sequences if str(item).strip()]
        else:
            defaults_list = []
        self.stop_sequences_entry.set_text(", ".join(defaults_list))
        self.timeout_spin.set_value(int(defaults.get("timeout", 60)))
        self.max_retries_spin.set_value(int(defaults.get("max_retries", 3)))
        self.retry_delay_spin.set_value(int(defaults.get("retry_delay", 5)))
        self._set_tool_choice_widgets(str(defaults.get("tool_choice", "auto")), defaults.get("tool_choice_name"))
        self.metadata_entry.set_text(self._format_metadata_for_entry(defaults.get("metadata", {})))
        self.thinking_toggle.set_active(bool(defaults.get("thinking", False)))
        budget = defaults.get("thinking_budget")
        if isinstance(budget, (int, float)) and budget > 0:
            self.thinking_budget_spin.set_value(int(budget))
        else:
            self.thinking_budget_spin.set_value(0)
        self._update_thinking_controls_state()
        self._update_save_button_state()

    def _parse_stop_sequences_entry(self) -> List[str]:
        raw = self.stop_sequences_entry.get_text() if hasattr(self.stop_sequences_entry, "get_text") else ""
        if not raw:
            return []
        return [segment.strip() for segment in raw.split(",") if segment.strip()]

    def _parse_metadata_entry(self, raw: str) -> Dict[str, str]:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        metadata: Dict[str, str] = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if not str(key).strip():
                    continue
                metadata[str(key).strip()] = "" if value is None else str(value).strip()
        elif isinstance(parsed, list):
            for entry in parsed:
                if isinstance(entry, dict):
                    for key, value in entry.items():
                        if not str(key).strip():
                            continue
                        metadata[str(key).strip()] = "" if value is None else str(value).strip()
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    key, value = entry
                    if not str(key).strip():
                        continue
                    metadata[str(key).strip()] = "" if value is None else str(value).strip()
        else:
            segments = [segment.strip() for segment in text.replace("\n", ",").split(",")]
            for segment in segments:
                if not segment:
                    continue
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                metadata[key] = value.strip()

        if len(metadata) > 16:
            metadata = dict(list(metadata.items())[:16])
        return metadata

    def _format_metadata_for_entry(self, metadata: object) -> str:
        if not isinstance(metadata, dict) or not metadata:
            return ""
        return ", ".join(f"{key}={value}" for key, value in metadata.items())

    def _update_save_button_state(self, *_args) -> None:
        if not hasattr(self, "save_button"):
            return

        payload = self._collect_payload()
        dirty = payload is not None and payload != self._initial_settings
        enabled = bool(payload) and dirty
        if hasattr(self.save_button, "set_sensitive"):
            self.save_button.set_sensitive(enabled)
        else:  # pragma: no cover - testing stubs
            self.save_button.sensitive = enabled

    def _get_tool_choice_value(self) -> str:
        result = None
        if hasattr(self.tool_choice_combo, "get_active"):
            try:
                index = self.tool_choice_combo.get_active()
            except TypeError:
                index = None
            if isinstance(index, int) and 0 <= index < len(self._tool_choice_map):
                result = self._tool_choice_map[index][0]

        if result is None:
            active_text = ""
            if hasattr(self.tool_choice_combo, "get_active_text"):
                active_text = self.tool_choice_combo.get_active_text() or ""
            else:
                active_text = getattr(self.tool_choice_combo, "active_text", "")

            for key, label in self._tool_choice_map:
                if label == active_text:
                    result = key
                    break

        if result is None:
            result = getattr(self, "_cached_tool_choice_value", "auto")

        self._cached_tool_choice_value = result
        return result

    def _get_tool_choice_name(self) -> Optional[str]:
        name = self.tool_name_entry.get_text() if hasattr(self.tool_name_entry, "get_text") else ""
        cleaned = (name or "").strip()
        return cleaned or None

    def _set_tool_choice_widgets(self, choice: str, name: Optional[str]) -> None:
        key = str(choice or "auto").lower()
        try:
            index = [entry[0] for entry in self._tool_choice_map].index(key)
        except ValueError:
            index = 0
        if hasattr(self.tool_choice_combo, "set_active"):
            self.tool_choice_combo.set_active(index)
        elif hasattr(self.tool_choice_combo, "set_active_id"):
            self.tool_choice_combo.set_active_id(str(index))
        elif hasattr(self.tool_choice_combo, "set_active_text"):
            self.tool_choice_combo.set_active_text(self._tool_choice_map[index][1])
        else:
            setattr(self.tool_choice_combo, "active_text", self._tool_choice_map[index][1])
            setattr(self.tool_choice_combo, "active_index", index)
        if isinstance(name, str):
            self.tool_name_entry.set_text(name)
        else:
            self.tool_name_entry.set_text("")
        self._cached_tool_choice_value = self._tool_choice_map[index][0]
        self._update_tool_choice_state()

    def _update_tool_choice_state(self, *_args) -> None:
        value = self._get_tool_choice_value()
        self._cached_tool_choice_value = value
        enabled = self.function_call_toggle.get_active() and value == "tool"
        if hasattr(self.tool_name_entry, "set_sensitive"):
            self.tool_name_entry.set_sensitive(enabled)
        if not enabled:
            self.tool_name_entry.set_placeholder_text("Optional tool name")
        else:
            self.tool_name_entry.set_placeholder_text("Enter tool name")

    def _update_thinking_controls_state(self, *_args) -> None:
        enabled = self.thinking_toggle.get_active()
        if hasattr(self.thinking_budget_spin, "set_sensitive"):
            self.thinking_budget_spin.set_sensitive(enabled)

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
