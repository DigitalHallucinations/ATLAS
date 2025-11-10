from __future__ import annotations

"""GTK window for configuring default settings of the Mistral provider."""

import json
import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import create_box

logger = logging.getLogger(__name__)


class MistralSettingsWindow(AtlasWindow):
    """Collect and persist preferences specific to the Mistral chat provider."""

    def __init__(self, ATLAS, config_manager, parent_window):
        super().__init__(
            title="Mistral Settings",
            modal=True,
            transient_for=parent_window,
            default_size=(420, 360),
        )
        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window

        self._last_message: Optional[Tuple[str, str, Gtk.MessageType]] = None
        self._current_settings: Dict[str, Any] = {}
        self._json_schema_is_valid = True
        self._json_schema_text_cache = ""
        self._json_schema_buffer = None
        self._json_schema_widget = None
        self._schema_default_message = (
            "Optional: provide a JSON schema to enforce structured responses."
        )

        self._custom_option_text = "Custom…"
        self._available_models: List[str] = []
        self._model_options_initialized = False
        self._custom_entry_visible = False
        self._api_key_visible = False
        self._default_api_key_placeholder = "Enter your Mistral API key"
        self._refresh_in_progress = False

        self._suppress_tool_choice_sync = False
        self._stored_base_url: Optional[str] = None
        self._base_url_is_valid = True

        self._prompt_mode_options = [
            ("default", None, "Standard prompt (default)"),
            ("reasoning", "reasoning", "Reasoning mode"),
        ]
        self._prompt_mode_value_by_id = {
            option_id: value for option_id, value, _ in self._prompt_mode_options
        }
        self._prompt_mode_index_by_id = {
            option_id: index for index, (option_id, _value, _label) in enumerate(self._prompt_mode_options)
        }
        self._prompt_mode_id_by_value = {
            value: option_id
            for option_id, value, _label in self._prompt_mode_options
            if value is not None
        }
        self._prompt_mode_default_id = "default"
        self.prompt_mode_combo: Optional[Gtk.ComboBoxText] = None

        self._build_ui()
        self.refresh_settings(clear_message=True)

    def present(self):  # pragma: no cover - GTK runtime integration
        try:
            self.refresh_settings(clear_message=False)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to refresh Mistral defaults: %s", exc, exc_info=True)
        return super().present()

    def _build_ui(self) -> None:
        container = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        self.set_child(container)

        scrolled_cls = getattr(Gtk, "ScrolledWindow", None)
        if scrolled_cls is not None:
            scroller = scrolled_cls()
            if hasattr(scroller, "set_policy"):
                scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        else:
            scroller = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        if hasattr(scroller, "set_hexpand"):
            scroller.set_hexpand(True)
        if hasattr(scroller, "set_vexpand"):
            scroller.set_vexpand(True)

        if not hasattr(scroller, "set_child"):
            def _set_child(widget):
                if hasattr(scroller, "append"):
                    scroller.append(widget)
                else:  # pragma: no cover - fallback for simple stubs
                    scroller.child = widget

            setattr(scroller, "set_child", _set_child)

        container.append(scroller)

        content_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=0)
        scroller.set_child(content_box)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        grid.set_margin_top(6)
        grid.set_margin_bottom(6)
        grid.set_margin_start(6)
        grid.set_margin_end(6)
        content_box.append(grid)

        row = 0
        api_label = Gtk.Label(label="Mistral API Key:")
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
        self.api_key_entry.set_placeholder_text(self._default_api_key_placeholder)
        api_entry_box.append(self.api_key_entry)

        self.api_key_toggle = Gtk.Button(label="Show")
        self.api_key_toggle.connect("clicked", self._on_api_key_toggle_clicked)
        api_entry_box.append(self.api_key_toggle)

        row += 1
        self.api_key_status_label = Gtk.Label(label="")
        self.api_key_status_label.set_xalign(0.0)
        grid.attach(self.api_key_status_label, 0, row, 2, 1)

        row += 1
        base_url_label = Gtk.Label(label="Custom Base URL:")
        base_url_label.set_xalign(0.0)
        grid.attach(base_url_label, 0, row, 1, 1)

        self.base_url_entry = Gtk.Entry()
        self.base_url_entry.set_hexpand(True)
        self.base_url_entry.set_placeholder_text("https://api.mistral.ai/v1")
        if hasattr(self.base_url_entry, "connect"):
            self.base_url_entry.connect("changed", self._on_base_url_changed)
        grid.attach(self.base_url_entry, 1, row, 1, 1)

        row += 1
        self.base_url_feedback_label = Gtk.Label(
            label="Leave blank to use the official endpoint."
        )
        self.base_url_feedback_label.set_xalign(0.0)
        if hasattr(self.base_url_feedback_label, "add_css_class"):
            self.base_url_feedback_label.add_css_class("dim-label")
        grid.attach(self.base_url_feedback_label, 0, row, 2, 1)

        row += 1
        model_label = Gtk.Label(label="Default model:")
        model_label.set_xalign(0.0)
        grid.attach(model_label, 0, row, 1, 1)

        model_row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        model_row_box.set_hexpand(True)
        grid.attach(model_row_box, 1, row, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        if hasattr(self.model_combo, "connect"):
            self.model_combo.connect("changed", self._on_model_combo_changed)
        model_row_box.append(self.model_combo)

        self.model_refresh_button = Gtk.Button(label="Refresh")
        if hasattr(self.model_refresh_button, "set_tooltip_text"):
            self.model_refresh_button.set_tooltip_text(
                "Fetch the latest models available from Mistral"
            )
        if hasattr(self.model_refresh_button, "set_hexpand"):
            self.model_refresh_button.set_hexpand(False)
        if hasattr(self.model_refresh_button, "connect"):
            self.model_refresh_button.connect("clicked", self._on_refresh_models_clicked)
        model_row_box.append(self.model_refresh_button)

        row += 1
        self.custom_model_label = Gtk.Label(label="Custom model ID:")
        self.custom_model_label.set_xalign(0.0)
        grid.attach(self.custom_model_label, 0, row, 1, 1)

        self.custom_model_entry = Gtk.Entry()
        self.custom_model_entry.set_hexpand(True)
        self.custom_model_entry.set_placeholder_text(
            "Enter custom model identifier"
        )
        grid.attach(self.custom_model_entry, 1, row, 1, 1)

        self._toggle_custom_entry(False)

        row += 1
        temperature_label = Gtk.Label(label="Temperature:")
        temperature_label.set_xalign(0.0)
        grid.attach(temperature_label, 0, row, 1, 1)

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
        freq_label = Gtk.Label(label="Frequency penalty:")
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
        presence_label = Gtk.Label(label="Presence penalty:")
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
        max_tokens_label = Gtk.Label(label="Max tokens (0 for provider default):")
        max_tokens_label.set_xalign(0.0)
        grid.attach(max_tokens_label, 0, row, 1, 1)

        self.max_tokens_adjustment = Gtk.Adjustment(
            lower=0, upper=128000, step_increment=128, page_increment=512, value=0
        )
        self.max_tokens_spin = Gtk.SpinButton(adjustment=self.max_tokens_adjustment, digits=0)
        self.max_tokens_spin.set_increments(128, 512)
        self.max_tokens_spin.set_hexpand(True)
        grid.attach(self.max_tokens_spin, 1, row, 1, 1)

        row += 1
        safe_prompt_label = Gtk.Label(label="Safe prompt mode:")
        safe_prompt_label.set_xalign(0.0)
        grid.attach(safe_prompt_label, 0, row, 1, 1)

        self.safe_prompt_toggle = Gtk.CheckButton(label="Enable safe prompt")
        self.safe_prompt_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.safe_prompt_toggle, 1, row, 1, 1)

        row += 1
        prompt_mode_label = Gtk.Label(label="Prompt mode:")
        prompt_mode_label.set_xalign(0.0)
        grid.attach(prompt_mode_label, 0, row, 1, 1)

        self.prompt_mode_combo = Gtk.ComboBoxText()
        self.prompt_mode_combo.set_hexpand(True)
        for option_id, _value, option_label in self._prompt_mode_options:
            if hasattr(self.prompt_mode_combo, "append"):
                self.prompt_mode_combo.append(option_id, option_label)
            else:  # pragma: no cover - fallback for simplified stubs
                self.prompt_mode_combo.append_text(option_label)
        if hasattr(self.prompt_mode_combo, "set_active_id"):
            self.prompt_mode_combo.set_active_id(self._prompt_mode_default_id)
        if hasattr(self.prompt_mode_combo, "set_active"):
            self.prompt_mode_combo.set_active(
                self._prompt_mode_index_by_id.get(self._prompt_mode_default_id, 0)
            )
        grid.attach(self.prompt_mode_combo, 1, row, 1, 1)

        row += 1
        stream_label = Gtk.Label(label="Streaming:")
        stream_label.set_xalign(0.0)
        grid.attach(stream_label, 0, row, 1, 1)

        self.stream_toggle = Gtk.CheckButton(label="Stream responses by default")
        self.stream_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.stream_toggle, 1, row, 1, 1)

        row += 1
        retries_label = Gtk.Label(label="Max retries:")
        retries_label.set_xalign(0.0)
        grid.attach(retries_label, 0, row, 1, 1)

        self.max_retries_adjustment = Gtk.Adjustment(
            lower=1, upper=10, step_increment=1, page_increment=1, value=3
        )
        self.max_retries_spin = Gtk.SpinButton(
            adjustment=self.max_retries_adjustment, digits=0
        )
        self.max_retries_spin.set_hexpand(True)
        grid.attach(self.max_retries_spin, 1, row, 1, 1)

        row += 1
        retry_min_label = Gtk.Label(label="Retry backoff minimum (seconds):")
        retry_min_label.set_xalign(0.0)
        grid.attach(retry_min_label, 0, row, 1, 1)

        self.retry_max_adjustment = None

        self.retry_min_adjustment = Gtk.Adjustment(
            lower=1, upper=300, step_increment=1, page_increment=5, value=4
        )
        self.retry_min_spin = Gtk.SpinButton(
            adjustment=self.retry_min_adjustment, digits=0
        )
        self.retry_min_spin.set_hexpand(True)
        if hasattr(self.retry_min_spin, "connect"):
            self.retry_min_spin.connect("value-changed", self._on_retry_min_changed)
        grid.attach(self.retry_min_spin, 1, row, 1, 1)

        row += 1
        retry_max_label = Gtk.Label(label="Retry backoff maximum (seconds):")
        retry_max_label.set_xalign(0.0)
        grid.attach(retry_max_label, 0, row, 1, 1)

        self.retry_max_adjustment = Gtk.Adjustment(
            lower=1, upper=600, step_increment=1, page_increment=10, value=10
        )
        self.retry_max_spin = Gtk.SpinButton(
            adjustment=self.retry_max_adjustment, digits=0
        )
        self.retry_max_spin.set_hexpand(True)
        grid.attach(self.retry_max_spin, 1, row, 1, 1)

        row += 1
        self.json_mode_toggle = Gtk.CheckButton(label="Force JSON responses")
        self.json_mode_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.json_mode_toggle, 0, row, 2, 1)

        row += 1
        schema_label = Gtk.Label(label="Response JSON Schema (optional):")
        schema_label.set_xalign(0.0)
        grid.attach(schema_label, 0, row, 2, 1)

        row += 1

        text_view_cls = getattr(Gtk, "TextView", None)
        text_buffer_cls = getattr(Gtk, "TextBuffer", None)
        scrolled_cls = getattr(Gtk, "ScrolledWindow", None)

        if text_view_cls is not None and text_buffer_cls is not None:
            if scrolled_cls is not None:
                schema_scroller = scrolled_cls()
                if hasattr(schema_scroller, "set_policy"):
                    schema_scroller.set_policy(
                        Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
                    )
                if hasattr(schema_scroller, "set_hexpand"):
                    schema_scroller.set_hexpand(True)
                if hasattr(schema_scroller, "set_vexpand"):
                    schema_scroller.set_vexpand(True)
                if hasattr(schema_scroller, "set_min_content_height"):
                    schema_scroller.set_min_content_height(140)
            else:
                schema_scroller = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

            self._json_schema_widget = text_view_cls()
            try:
                self._json_schema_buffer = text_buffer_cls()
            except Exception:  # pragma: no cover - defensive fallback
                self._json_schema_buffer = None

            if (
                self._json_schema_buffer is not None
                and hasattr(self._json_schema_widget, "set_buffer")
            ):
                self._json_schema_widget.set_buffer(self._json_schema_buffer)

            if hasattr(self._json_schema_widget, "set_wrap_mode"):
                try:
                    from gi.repository import Pango  # type: ignore

                    self._json_schema_widget.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
                except Exception:  # pragma: no cover - optional dependency may be absent
                    pass

            if hasattr(schema_scroller, "set_child"):
                schema_scroller.set_child(self._json_schema_widget)
            elif hasattr(schema_scroller, "append"):
                schema_scroller.append(self._json_schema_widget)
            else:  # pragma: no cover - fallback for simple stubs
                schema_scroller.child = self._json_schema_widget

            grid.attach(schema_scroller, 0, row, 2, 1)
        else:
            self._json_schema_widget = Gtk.Entry()
            if hasattr(self._json_schema_widget, "set_placeholder_text"):
                self._json_schema_widget.set_placeholder_text(
                    '{"name": "atlas_response", "schema": {"type": "object"}}'
                )
            if hasattr(self._json_schema_widget, "set_hexpand"):
                self._json_schema_widget.set_hexpand(True)
            grid.attach(self._json_schema_widget, 0, row, 2, 1)

        row += 1

        self.json_schema_feedback_label = Gtk.Label(
            label=self._schema_default_message
        )
        self.json_schema_feedback_label.set_xalign(0.0)
        grid.attach(self.json_schema_feedback_label, 0, row, 2, 1)

        row += 1
        seed_label = Gtk.Label(label="Random seed (optional):")
        seed_label.set_xalign(0.0)
        grid.attach(seed_label, 0, row, 1, 1)

        self.random_seed_entry = Gtk.Entry()
        self.random_seed_entry.set_hexpand(True)
        self.random_seed_entry.set_placeholder_text("Leave empty for random behaviour")
        grid.attach(self.random_seed_entry, 1, row, 1, 1)

        row += 1
        tool_choice_label = Gtk.Label(label="Tool calling:")
        tool_choice_label.set_xalign(0.0)
        grid.attach(tool_choice_label, 0, row, 1, 1)

        tool_choice_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        tool_choice_box.set_hexpand(True)
        grid.attach(tool_choice_box, 1, row, 1, 1)

        self.tool_call_toggle = Gtk.CheckButton(label="Allow automatic tool calls")
        self.tool_call_toggle.set_halign(Gtk.Align.START)
        if hasattr(self.tool_call_toggle, "connect"):
            self.tool_call_toggle.connect("toggled", self._on_tool_call_toggle_toggled)
        tool_choice_box.append(self.tool_call_toggle)

        self.parallel_tool_calls_toggle = Gtk.CheckButton(label="Allow parallel tool calls")
        self.parallel_tool_calls_toggle.set_halign(Gtk.Align.START)
        tool_choice_box.append(self.parallel_tool_calls_toggle)

        self.require_tool_toggle = Gtk.CheckButton(label="Require a tool call before responding")
        self.require_tool_toggle.set_halign(Gtk.Align.START)
        if hasattr(self.require_tool_toggle, "connect"):
            self.require_tool_toggle.connect("toggled", self._on_require_tool_toggle_toggled)
        tool_choice_box.append(self.require_tool_toggle)

        self.tool_choice_entry = Gtk.Entry()
        self.tool_choice_entry.set_hexpand(True)
        self.tool_choice_entry.set_placeholder_text("Leave empty for auto / enter JSON for advanced control")
        if hasattr(self.tool_choice_entry, "connect"):
            self.tool_choice_entry.connect("changed", self._on_tool_choice_entry_changed)
        tool_choice_box.append(self.tool_choice_entry)

        row += 1
        stop_sequences_label = Gtk.Label(label="Stop sequences:")
        stop_sequences_label.set_xalign(0.0)
        grid.attach(stop_sequences_label, 0, row, 1, 1)

        self.stop_sequences_entry = Gtk.Entry()
        self.stop_sequences_entry.set_hexpand(True)
        self.stop_sequences_entry.set_placeholder_text(
            "Comma-separated tokens to stop generation"
        )
        self.stop_sequences_entry.set_tooltip_text(
            "Provide comma separated stop strings (maximum of four)."
        )
        grid.attach(self.stop_sequences_entry, 1, row, 1, 1)

        row += 1
        self.message_label = Gtk.Label(label="")
        self.message_label.set_xalign(0.0)
        grid.attach(self.message_label, 0, row, 2, 1)

        row += 1
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_halign(Gtk.Align.END)
        container.append(button_box)

        self.save_api_key_button = Gtk.Button(label="Save API Key")
        self.save_api_key_button.connect("clicked", self.on_save_api_key_clicked)
        button_box.append(self.save_api_key_button)

        save_button = Gtk.Button(label="Save Defaults")
        save_button.connect("clicked", self.on_save_clicked)
        button_box.append(save_button)

        close_button = Gtk.Button(label="Close")
        close_button.connect("clicked", lambda *_args: self.close())
        button_box.append(close_button)

        self._update_tool_controls_state()

    def refresh_settings(self, *, clear_message: bool = False) -> None:
        settings = self.config_manager.get_mistral_llm_settings()
        self._current_settings = dict(settings)

        self._refresh_api_key_status()
        self._stored_base_url = settings.get("base_url")
        if isinstance(self._stored_base_url, str):
            self._stored_base_url = self._stored_base_url or None
        self._apply_base_url_to_entry(self._stored_base_url)
        self._refresh_model_options()
        self._select_model(settings.get("model", "") or "")
        self.temperature_spin.set_value(float(settings.get("temperature", 0.0)))
        self.top_p_spin.set_value(float(settings.get("top_p", 1.0)))
        self.frequency_penalty_spin.set_value(float(settings.get("frequency_penalty", 0.0)))
        self.presence_penalty_spin.set_value(float(settings.get("presence_penalty", 0.0)))

        max_tokens = settings.get("max_tokens")
        self.max_tokens_spin.set_value(float(max_tokens or 0))

        self.safe_prompt_toggle.set_active(bool(settings.get("safe_prompt", False)))
        self._select_prompt_mode(settings.get("prompt_mode"))
        self.stream_toggle.set_active(bool(settings.get("stream", True)))

        self.max_retries_spin.set_value(float(settings.get("max_retries", 3)))

        retry_min = settings.get("retry_min_seconds", 4)
        retry_max = settings.get("retry_max_seconds", max(retry_min, 10))
        self.retry_min_spin.set_value(float(retry_min))
        if hasattr(self.retry_max_adjustment, "set_lower"):
            self.retry_max_adjustment.set_lower(float(max(1, retry_min)))
        self.retry_max_spin.set_value(float(max(retry_min, retry_max)))

        self.json_mode_toggle.set_active(bool(settings.get("json_mode", False)))

        schema_payload = settings.get("json_schema") if isinstance(settings, dict) else None
        formatted_schema = self._format_json_schema(schema_payload)
        self._write_json_schema_text(formatted_schema)
        if schema_payload:
            self._update_json_schema_feedback(
                True, "JSON schema loaded from saved settings."
            )
        else:
            self._update_json_schema_feedback(True, self._schema_default_message)

        random_seed = settings.get("random_seed")
        if random_seed in {None, ""}:
            self.random_seed_entry.set_text("")
        else:
            self.random_seed_entry.set_text(str(random_seed))

        tool_choice = settings.get("tool_choice")
        allow_tool_calls = True
        require_tool_call = False
        tool_choice_text = ""

        if isinstance(tool_choice, Mapping):
            tool_choice_text = json.dumps(tool_choice, sort_keys=True)
        elif isinstance(tool_choice, str):
            normalized = tool_choice.strip()
            lower = normalized.lower()
            if lower == "none":
                allow_tool_calls = False
            elif lower == "required":
                require_tool_call = True
            elif lower and lower not in {"auto", "automatic"}:
                tool_choice_text = normalized
        elif tool_choice is None:
            tool_choice_text = ""
        else:
            tool_choice_text = str(tool_choice)

        self._suppress_tool_choice_sync = True
        try:
            if hasattr(self.tool_choice_entry, "set_text"):
                self.tool_choice_entry.set_text(tool_choice_text)
            else:  # pragma: no cover - testing stubs
                self.tool_choice_entry.text = tool_choice_text

            if hasattr(self.tool_call_toggle, "set_active"):
                self.tool_call_toggle.set_active(allow_tool_calls)
            if hasattr(self.require_tool_toggle, "set_active"):
                self.require_tool_toggle.set_active(require_tool_call)
        finally:
            self._suppress_tool_choice_sync = False

        parallel_setting = bool(settings.get("parallel_tool_calls", True))
        if not allow_tool_calls:
            parallel_setting = False

        self.parallel_tool_calls_toggle.set_active(parallel_setting)
        self._update_tool_controls_state()

        tokens = settings.get("stop_sequences") or []
        self.stop_sequences_entry.set_text(", ".join(tokens))

        if clear_message:
            self._last_message = None
            self._update_label(self.message_label, "")

    def _on_retry_min_changed(self, spin_button) -> None:
        try:
            value = spin_button.get_value()
        except Exception:
            value = 1.0
        if hasattr(self.retry_max_adjustment, "set_lower"):
            self.retry_max_adjustment.set_lower(max(1.0, value))
        if (
            hasattr(self, "retry_max_spin")
            and hasattr(self.retry_max_spin, "get_value")
            and self.retry_max_spin.get_value() < value
        ):
            self.retry_max_spin.set_value(value)

    def on_save_api_key_clicked(self, _button: Gtk.Button) -> None:
        api_key = (self.api_key_entry.get_text() or "").strip()
        if not api_key:
            self._set_message(
                "Error",
                "Enter an API key before saving.",
                Gtk.MessageType.ERROR,
            )
            return

        def handle_success(result: Dict[str, Any]) -> None:
            GLib.idle_add(self._handle_api_key_save_result, result)

        def handle_error(exc: Exception) -> None:
            logger.error("Error saving Mistral API key: %s", exc, exc_info=True)
            GLib.idle_add(
                self._handle_api_key_save_result,
                {"success": False, "error": str(exc)},
            )

        helper = getattr(self.ATLAS, "update_provider_api_key_in_background", None)
        if callable(helper):
            try:
                helper(
                    "Mistral",
                    api_key,
                    on_success=handle_success,
                    on_error=handle_error,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Unable to schedule Mistral API key save: %s", exc, exc_info=True
                )
                self._set_message(
                    "Error",
                    f"Failed to save API key: {str(exc)}",
                    Gtk.MessageType.ERROR,
                )
            return

        updater = getattr(self.ATLAS, "update_provider_api_key", None)
        if not callable(updater):
            self._set_message(
                "Error",
                "Saving API keys is not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        runner = getattr(self.ATLAS, "run_provider_manager_task", None)
        if not callable(runner):
            runner = getattr(self.ATLAS, "run_in_background", None)

        if not callable(runner):
            self._set_message(
                "Error",
                "Background tasks are not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        try:
            runner(
                lambda: updater("Mistral", api_key),
                on_success=handle_success,
                on_error=handle_error,
                thread_name="mistral-api-key-fallback",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Unable to start fallback API key save task: %s", exc, exc_info=True
            )
            self._set_message(
                "Error",
                f"Failed to save API key: {str(exc)}",
                Gtk.MessageType.ERROR,
            )

    def _handle_api_key_save_result(self, result: Dict[str, Any]) -> bool:
        if isinstance(result, dict) and result.get("success"):
            message = result.get("message") or "API key saved."
            self._set_message("Success", message, Gtk.MessageType.INFO)
            self.api_key_entry.set_text("")
            self._refresh_api_key_status()
        else:
            if isinstance(result, dict):
                detail = (
                    result.get("error")
                    or result.get("message")
                    or "Unable to save API key."
                )
            else:
                detail = str(result)
            self._set_message("Error", detail, Gtk.MessageType.ERROR)

        return False

    def _set_message(self, title: str, body: str, message_type: Gtk.MessageType) -> None:
        self._last_message = (title, body, message_type)
        if body:
            display = f"{title}: {body}"
        else:
            display = title
        self._update_label(self.message_label, display)

    def _set_refresh_in_progress(self, active: bool) -> None:
        self._refresh_in_progress = bool(active)
        button = getattr(self, "model_refresh_button", None)
        if button is None:
            return
        setter = getattr(button, "set_sensitive", None)
        if callable(setter):
            setter(not active)
        else:  # pragma: no cover - fallback for simple stubs
            button.sensitive = not active

    def _on_refresh_models_clicked(self, _button) -> None:
        if self._refresh_in_progress:
            return

        checker = getattr(self.config_manager, "has_provider_api_key", None)
        has_key = False
        if callable(checker):
            try:
                has_key = bool(checker("Mistral"))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Failed to determine Mistral API key status: %s", exc)
                has_key = False

        if not has_key:
            self._set_message(
                "Error",
                "Save a Mistral API key before refreshing the model list.",
                Gtk.MessageType.ERROR,
            )
            return

        provider_manager = getattr(self.ATLAS, "provider_manager", None)
        fetcher = getattr(provider_manager, "fetch_mistral_models", None)
        if not callable(fetcher):
            self._set_message(
                "Error",
                "Refreshing Mistral models is not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        base_url, base_valid = self._sync_base_url_state()
        if not base_valid:
            self._set_message(
                "Error",
                "Enter a valid HTTP(S) base URL or leave the field blank before refreshing.",
                Gtk.MessageType.ERROR,
            )
            return

        def handle_success(result):
            GLib.idle_add(self._handle_model_refresh_result, result)

        def handle_error(exc):
            GLib.idle_add(self._handle_model_refresh_error, exc)

        self._set_refresh_in_progress(True)

        runner = getattr(self.ATLAS, "run_provider_manager_task", None)
        if not callable(runner):
            runner = getattr(self.ATLAS, "run_in_background", None)

        if not callable(runner):
            self._set_refresh_in_progress(False)
            self._set_message(
                "Error",
                "Background tasks are not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        try:
            runner(
                lambda: fetcher(base_url=base_url),
                on_success=handle_success,
                on_error=handle_error,
                thread_name="mistral-model-refresh",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to start Mistral model refresh: %s", exc, exc_info=True)
            self._set_refresh_in_progress(False)
            detail = str(exc) or "Unable to start refresh task."
            self._set_message("Error", detail, Gtk.MessageType.ERROR)

    def _handle_model_refresh_result(self, result: Dict[str, Any]) -> bool:
        self._set_refresh_in_progress(False)

        if isinstance(result, dict) and result.get("success"):
            data = result.get("data") or {}
            models = data.get("models") or result.get("models") or []
            self._refresh_model_options(preferred=models)
            message = result.get("message") or "Mistral models refreshed."
            self._set_message("Success", message, Gtk.MessageType.INFO)
        else:
            if isinstance(result, dict):
                detail = (
                    result.get("error")
                    or result.get("message")
                    or "Unable to refresh Mistral models."
                )
            else:
                detail = str(result)
            self._set_message("Error", detail, Gtk.MessageType.ERROR)

        return False

    def _handle_model_refresh_error(self, exc: Exception) -> bool:
        self._set_refresh_in_progress(False)
        detail = str(exc) or exc.__class__.__name__
        self._set_message(
            "Error",
            f"Failed to refresh Mistral models: {detail}",
            Gtk.MessageType.ERROR,
        )
        return False

    def _update_label(self, widget: Gtk.Widget, text: str) -> None:
        if hasattr(widget, "set_text"):
            widget.set_text(text)
        elif hasattr(widget, "set_label"):
            widget.set_label(text)
        else:  # pragma: no cover - fallback for stubs/tests
            setattr(widget, "label", text)

    def _parse_tool_choice(self, text: str) -> Any:
        allow_widget = getattr(self, "tool_call_toggle", None)
        require_widget = getattr(self, "require_tool_toggle", None)

        allow_active = bool(getattr(allow_widget, "get_active", lambda: True)())
        require_active = bool(getattr(require_widget, "get_active", lambda: False)())

        cleaned = (text or "").strip()

        if not allow_active:
            return "none"

        if cleaned:
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                normalized = cleaned.lower()
                if normalized in {"auto", "automatic"}:
                    return "auto"
                if normalized == "none":
                    return "none"
                if normalized == "required":
                    return "required"
                return cleaned

            if isinstance(parsed, (dict, list)):
                return parsed
            if isinstance(parsed, str):
                normalized = parsed.strip().lower()
                if normalized in {"auto", "automatic", "none", "required"}:
                    return normalized if normalized != "automatic" else "auto"
            return parsed

        if require_active:
            return "required"

        return "auto"

    def _on_tool_call_toggle_toggled(self, _button) -> None:
        if self._suppress_tool_choice_sync:
            return
        self._update_tool_controls_state()

    def _on_require_tool_toggle_toggled(self, _button) -> None:
        if self._suppress_tool_choice_sync:
            return
        self._update_tool_controls_state()

    def _on_tool_choice_entry_changed(self, entry) -> None:
        if self._suppress_tool_choice_sync:
            return

        text = getattr(entry, "get_text", lambda: "")() or ""
        cleaned = text.strip()
        lower = cleaned.lower()

        if not cleaned:
            self._update_tool_controls_state()
            return

        self._suppress_tool_choice_sync = True
        try:
            if lower == "none":
                if hasattr(self.tool_call_toggle, "set_active"):
                    self.tool_call_toggle.set_active(False)
                if hasattr(self.require_tool_toggle, "set_active"):
                    self.require_tool_toggle.set_active(False)
            else:
                if hasattr(self.tool_call_toggle, "set_active") and not self.tool_call_toggle.get_active():
                    self.tool_call_toggle.set_active(True)
                if lower in {"auto", "automatic"}:
                    if hasattr(self.require_tool_toggle, "set_active"):
                        self.require_tool_toggle.set_active(False)
                elif lower == "required":
                    if hasattr(self.require_tool_toggle, "set_active"):
                        self.require_tool_toggle.set_active(True)
        finally:
            self._suppress_tool_choice_sync = False

        self._update_tool_controls_state()

    def _get_tool_choice_text(self) -> str:
        entry = getattr(self, "tool_choice_entry", None)
        if entry is None:
            return ""

        getter = getattr(entry, "get_text", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - fallback for stubs/tests
                value = getattr(entry, "text", "")
        else:
            value = getattr(entry, "text", "")

        if value is None:
            return ""
        return str(value)

    def _update_tool_controls_state(self) -> None:
        allow_widget = getattr(self, "tool_call_toggle", None)
        enabled = bool(getattr(allow_widget, "get_active", lambda: True)())

        dependents = [
            getattr(self, "parallel_tool_calls_toggle", None),
            getattr(self, "require_tool_toggle", None),
            getattr(self, "tool_choice_entry", None),
        ]

        for widget in dependents:
            if widget is None:
                continue
            setter = getattr(widget, "set_sensitive", None)
            if callable(setter):
                setter(enabled)
            else:  # pragma: no cover - fallback for stubs
                setattr(widget, "sensitive", enabled)

        if not enabled:
            parallel_widget = getattr(self, "parallel_tool_calls_toggle", None)
            if parallel_widget is not None:
                getter = getattr(parallel_widget, "get_active", None)
                if callable(getter) and getter():
                    parallel_widget.set_active(False)
            require_widget = getattr(self, "require_tool_toggle", None)
            if require_widget is not None:
                getter = getattr(require_widget, "get_active", None)
                if callable(getter) and getter():
                    require_widget.set_active(False)

    def _select_prompt_mode(self, prompt_mode: Optional[str]) -> None:
        combo = getattr(self, "prompt_mode_combo", None)
        if combo is None:
            return

        normalized = None
        if isinstance(prompt_mode, str):
            normalized = prompt_mode.strip().lower() or None

        target_id = self._prompt_mode_default_id
        if normalized in self._prompt_mode_id_by_value:
            target_id = self._prompt_mode_id_by_value[normalized]

        setter = getattr(combo, "set_active_id", None)
        if callable(setter):
            setter(target_id)

        index = self._prompt_mode_index_by_id.get(target_id, 0)
        alt_setter = getattr(combo, "set_active", None)
        if callable(alt_setter):
            alt_setter(index)

    def _get_selected_prompt_mode(self) -> Optional[str]:
        combo = getattr(self, "prompt_mode_combo", None)
        if combo is None:
            return None

        active_id = None
        getter = getattr(combo, "get_active_id", None)
        if callable(getter):
            active_id = getter()

        if not active_id:
            index_getter = getattr(combo, "get_active", None)
            if callable(index_getter):
                try:
                    index = index_getter()
                except Exception:  # pragma: no cover - defensive fallback
                    index = None
                if isinstance(index, int) and index >= 0:
                    try:
                        active_id = self._prompt_mode_options[index][0]
                    except (IndexError, TypeError):  # pragma: no cover - safety net
                        active_id = None
            if not active_id and hasattr(combo, "_active"):
                index = getattr(combo, "_active")
                if isinstance(index, int) and index >= 0:
                    try:
                        active_id = self._prompt_mode_options[index][0]
                    except (IndexError, TypeError):  # pragma: no cover - safety net
                        active_id = None

        if active_id in {None, self._prompt_mode_default_id}:
            return None

        return self._prompt_mode_value_by_id.get(active_id)

    def _refresh_api_key_status(self) -> None:
        status_text = "Credential status is unavailable."
        placeholder = self._default_api_key_placeholder
        tooltip = "Enter your Mistral API key."

        has_key = False
        if hasattr(self.config_manager, "has_provider_api_key"):
            try:
                has_key = bool(self.config_manager.has_provider_api_key("Mistral"))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Unable to query stored Mistral API key: %s", exc, exc_info=True
                )

        hint = ""
        atlas = getattr(self, "ATLAS", None)
        if atlas is not None and hasattr(atlas, "get_provider_api_key_status"):
            try:
                payload = atlas.get_provider_api_key_status("Mistral") or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Unable to determine Mistral API key status: %s", exc, exc_info=True
                )
            else:
                if isinstance(payload, dict):
                    metadata = payload.get("metadata")
                    if isinstance(metadata, dict):
                        hint = metadata.get("hint") or ""

        if has_key:
            suffix = f" ({hint})" if hint else ""
            status_text = f"An API key is saved for Mistral.{suffix}"
            if hint:
                placeholder = f"Saved key: {hint}"
                tooltip = (
                    f"A saved Mistral key is active ({hint}). Enter a new key to replace it."
                )
            else:
                tooltip = "A Mistral API key is saved. Enter a new key to replace it."
        else:
            status_text = "No API key saved for Mistral."
            tooltip = "Enter your Mistral API key."

        if hasattr(self.api_key_entry, "set_placeholder_text"):
            self.api_key_entry.set_placeholder_text(placeholder)
        if hasattr(self.api_key_entry, "set_tooltip_text"):
            self.api_key_entry.set_tooltip_text(tooltip)

        self._update_label(self.api_key_status_label, status_text)

    def _on_api_key_toggle_clicked(self, _button: Gtk.Button) -> None:
        self._api_key_visible = not self._api_key_visible

        if hasattr(self.api_key_entry, "set_visibility"):
            self.api_key_entry.set_visibility(self._api_key_visible)
        else:  # pragma: no cover - testing stubs
            self.api_key_entry.visibility = self._api_key_visible

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

        label = "Hide" if self._api_key_visible else "Show"
        if hasattr(self.api_key_toggle, "set_label"):
            self.api_key_toggle.set_label(label)
        else:  # pragma: no cover - testing stubs
            self.api_key_toggle.label = label

    def _parse_stop_sequences(self) -> List[str]:
        try:
            text = self.stop_sequences_entry.get_text()
        except Exception:  # pragma: no cover - defensive for stubs/tests
            text = ""
        if not text:
            return []
        tokens: List[str] = []
        for part in text.replace("\n", ",").split(","):
            cleaned = part.strip()
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def _parse_optional_int(self, value: str) -> Optional[int]:
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError as exc:
            raise ValueError("Random seed must be an integer.") from exc

    def on_save_clicked(self, _button) -> None:
        max_tokens_value = self.max_tokens_spin.get_value_as_int()
        max_tokens = max_tokens_value if max_tokens_value > 0 else None

        try:
            model = self._get_selected_model().strip()
            if not model:
                raise ValueError(
                    "Please select a model or provide a custom model identifier."
                )
            raw_schema_text = self._read_json_schema_text().strip()
            random_seed = self._parse_optional_int(self.random_seed_entry.get_text())
            allow_tools = bool(
                getattr(self.tool_call_toggle, "get_active", lambda: True)()
            )
            parallel_tools = bool(
                getattr(self.parallel_tool_calls_toggle, "get_active", lambda: True)()
            )
            if not allow_tools:
                parallel_tools = False

            tool_choice_value = self._parse_tool_choice(self._get_tool_choice_text())
            retry_min = self.retry_min_spin.get_value_as_int()
            retry_max = self.retry_max_spin.get_value_as_int()
            if retry_max < retry_min:
                raise ValueError(
                    "Retry maximum seconds must be greater than or equal to the minimum."
                )
            base_url, base_valid = self._sync_base_url_state()
            if not base_valid:
                raise ValueError(
                    "Enter a valid HTTP(S) base URL or leave the field blank."
                )
            prompt_mode_value = self._get_selected_prompt_mode()
            saved = self.config_manager.set_mistral_llm_settings(
                model=model,
                temperature=self.temperature_spin.get_value(),
                top_p=self.top_p_spin.get_value(),
                max_tokens=max_tokens,
                safe_prompt=self.safe_prompt_toggle.get_active(),
                stream=self.stream_toggle.get_active(),
                random_seed=random_seed,
                frequency_penalty=self.frequency_penalty_spin.get_value(),
                presence_penalty=self.presence_penalty_spin.get_value(),
                tool_choice=tool_choice_value,
                parallel_tool_calls=parallel_tools,
                stop_sequences=self._parse_stop_sequences(),
                json_mode=self.json_mode_toggle.get_active(),
                json_schema=raw_schema_text if raw_schema_text else "",
                max_retries=self.max_retries_spin.get_value_as_int(),
                retry_min_seconds=retry_min,
                retry_max_seconds=retry_max,
                base_url=base_url,
                prompt_mode=prompt_mode_value,
            )
        except Exception as exc:
            logger.error("Failed to save Mistral settings: %s", exc, exc_info=True)
            message = str(exc) or "Unable to save Mistral defaults."
            self._set_message("Error", message, Gtk.MessageType.ERROR)
            self._update_json_schema_feedback(False, message)
            return

        self._current_settings = dict(saved)
        self._set_message("Success", "Mistral defaults updated.", Gtk.MessageType.INFO)
        if raw_schema_text:
            self._update_json_schema_feedback(True, "JSON schema saved successfully.")
        else:
            self._update_json_schema_feedback(True, self._schema_default_message)
        self.refresh_settings(clear_message=False)

    def _load_known_models(self, preferred: Optional[List[str]] = None) -> List[str]:
        models: List[str] = []

        if preferred:
            models = [
                entry.strip()
                for entry in preferred
                if isinstance(entry, str) and entry.strip()
            ]
            if models:
                return self._deduplicate(models)

        provider_manager = getattr(self.ATLAS, "provider_manager", None)
        model_manager = getattr(provider_manager, "model_manager", None)
        if model_manager is not None:
            try:
                available = model_manager.get_available_models("Mistral")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to load Mistral models from ModelManager: %s", exc
                )
            else:
                if isinstance(available, dict):
                    candidate = available.get("Mistral")
                else:
                    candidate = available
                if isinstance(candidate, list):
                    models = [
                        entry.strip()
                        for entry in candidate
                        if isinstance(entry, str) and entry.strip()
                    ]

        return self._deduplicate(models)

    def _deduplicate(self, models: List[str]) -> List[str]:
        seen: set[str] = set()
        unique: List[str] = []
        for name in models:
            if name not in seen:
                unique.append(name)
                seen.add(name)
        return unique

    def _populate_model_combo(self, models: List[str]) -> None:
        self._available_models = list(models)
        combo = getattr(self, "model_combo", None)
        if combo is None:
            return
        remover = getattr(combo, "remove_all", None)
        if callable(remover):
            remover()
        for entry in self._available_models:
            combo.append_text(entry)
        combo.append_text(self._custom_option_text)
        if self._available_models:
            combo.set_active(0)
        else:
            combo.set_active(0)
        self._model_options_initialized = True

    def _refresh_model_options(self, preferred: Optional[List[str]] = None) -> None:
        latest = self._load_known_models(preferred=preferred)
        if not self._model_options_initialized or latest != self._available_models:
            current_selection = self._get_selected_model()
            self._populate_model_combo(latest)
            if current_selection:
                self._select_model(current_selection)

    def _select_model(self, model_id: str) -> None:
        model_id = model_id.strip()
        if model_id and model_id in self._available_models:
            index = self._available_models.index(model_id)
            self.model_combo.set_active(index)
            # _on_model_combo_changed will hide the custom entry.
            self.custom_model_entry.set_text("")
            self._toggle_custom_entry(False)
        else:
            custom_index = len(self._available_models)
            self.model_combo.set_active(custom_index)
            self.custom_model_entry.set_text(model_id)
            self._toggle_custom_entry(True)

    def _toggle_custom_entry(self, visible: bool) -> None:
        self._custom_entry_visible = bool(visible)
        for widget in (self.custom_model_label, self.custom_model_entry):
            if widget is None:
                continue
            setter = getattr(widget, "set_visible", None)
            if callable(setter):
                setter(visible)
            else:  # pragma: no cover - fallback for stubs/tests
                setattr(widget, "visible", visible)

    def _on_model_combo_changed(self, _combo) -> None:
        text = ""
        try:
            text = self.model_combo.get_active_text() or ""
        except Exception:  # pragma: no cover - defensive fallback
            text = ""
        is_custom = not text or text == self._custom_option_text
        self._toggle_custom_entry(is_custom)
        if not is_custom:
            self.custom_model_entry.set_text("")

    def _get_selected_model(self) -> str:
        try:
            selected = self.model_combo.get_active_text() or ""
        except Exception:  # pragma: no cover - defensive fallback
            selected = ""
        if selected and selected != self._custom_option_text:
            return selected.strip()
        custom_value = self.custom_model_entry.get_text().strip()
        if custom_value:
            return custom_value
        current_model = self._current_settings.get("model")
        if isinstance(current_model, str):
            return current_model.strip()
        return ""

    def _format_json_schema(self, payload: Any) -> str:
        if not payload:
            return ""
        try:
            return json.dumps(payload, indent=2, sort_keys=True)
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            return ""

    def _write_json_schema_text(self, text: str) -> None:
        self._json_schema_text_cache = text or ""
        buffer = getattr(self, "_json_schema_buffer", None)
        if buffer is not None and hasattr(buffer, "set_text"):
            try:
                buffer.set_text(self._json_schema_text_cache)
                return
            except Exception:  # pragma: no cover - defensive fallback
                pass

        widget = getattr(self, "_json_schema_widget", None)
        if widget is not None and hasattr(widget, "set_text"):
            widget.set_text(self._json_schema_text_cache)
        self._json_schema_text_cache = text or ""

    def _read_json_schema_text(self) -> str:
        buffer = getattr(self, "_json_schema_buffer", None)
        if buffer is not None and hasattr(buffer, "get_text"):
            try:
                start = buffer.get_start_iter()
                end = buffer.get_end_iter()
                return buffer.get_text(start, end, True)
            except Exception:  # pragma: no cover - fallback for stubs
                pass

        widget = getattr(self, "_json_schema_widget", None)
        if widget is not None and hasattr(widget, "get_text"):
            try:
                return widget.get_text()
            except Exception:  # pragma: no cover - fallback
                pass

        return getattr(self, "_json_schema_text_cache", "")

    def _update_json_schema_feedback(self, valid: bool, message: str) -> None:
        self._json_schema_is_valid = valid
        label = getattr(self, "json_schema_feedback_label", None)
        if label is None:
            return
        text = message or self._schema_default_message
        if hasattr(label, "set_text"):
            label.set_text(text)
        elif hasattr(label, "set_label"):
            label.set_label(text)
        else:  # pragma: no cover - fallback for stubs
            setattr(label, "label", text)
