from __future__ import annotations

"""Dedicated GTK window for configuring OpenAI provider defaults."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.utils import create_box
from modules.background_tasks import run_async_in_thread

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
        self._model_placeholder: str = "Loading models…"
        self._api_key_visible = False
        self._base_url_is_valid = True
        self._reasoning_effort_values: Tuple[str, ...] = ("low", "medium", "high")
        self._json_schema_is_valid = True
        self._json_schema_text_cache = ""
        self._audio_voice_options: Tuple[str, ...] = (
            "alloy",
            "verse",
            "aria",
            "lumen",
            "sage",
        )
        self._audio_format_options: Tuple[str, ...] = (
            "wav",
            "mp3",
            "ogg",
            "flac",
            "aac",
        )

        scroller_cls = getattr(Gtk, "ScrolledWindow", None)
        if scroller_cls is not None:
            scroller = scroller_cls()
            if hasattr(scroller, "set_policy"):
                scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        else:
            scroller = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        if hasattr(scroller, "set_hexpand"):
            scroller.set_hexpand(True)
        if hasattr(scroller, "set_vexpand"):
            scroller.set_vexpand(True)

        if not hasattr(scroller, "set_child"):
            def _set_child(child):
                if hasattr(scroller, "append"):
                    scroller.append(child)
                else:  # pragma: no cover - defensive fallback for test stubs
                    scroller.child = child

            setattr(scroller, "set_child", _set_child)

        self.set_child(scroller)

        main_box = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=12)
        scroller.set_child(main_box)

        self.settings_notebook = Gtk.Notebook()
        if hasattr(self.settings_notebook, "set_hexpand"):
            self.settings_notebook.set_hexpand(True)
        if hasattr(self.settings_notebook, "set_vexpand"):
            self.settings_notebook.set_vexpand(True)
        main_box.append(self.settings_notebook)

        general_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin=6,
        )
        general_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        general_box.append(general_grid)

        row = 0
        api_label = Gtk.Label(label="OpenAI API Key:")
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
        self.api_key_entry.set_placeholder_text("Enter your OpenAI API key")
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
        general_grid.attach(self.model_combo, 1, row, 1, 1)

        row += 1
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_xalign(0.0)
        general_grid.attach(temp_label, 0, row, 1, 1)
        self.temperature_adjustment = Gtk.Adjustment(
            lower=0.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.temperature_spin = Gtk.SpinButton(adjustment=self.temperature_adjustment, digits=2)
        self.temperature_spin.set_increments(0.05, 0.1)
        self.temperature_spin.set_hexpand(True)
        general_grid.attach(self.temperature_spin, 1, row, 1, 1)

        row += 1
        top_p_label = Gtk.Label(label="Top-p:")
        top_p_label.set_xalign(0.0)
        general_grid.attach(top_p_label, 0, row, 1, 1)
        self.top_p_adjustment = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.01, page_increment=0.05, value=1.0
        )
        self.top_p_spin = Gtk.SpinButton(adjustment=self.top_p_adjustment, digits=2)
        self.top_p_spin.set_increments(0.01, 0.05)
        self.top_p_spin.set_hexpand(True)
        general_grid.attach(self.top_p_spin, 1, row, 1, 1)

        row += 1
        freq_label = Gtk.Label(label="Frequency Penalty:")
        freq_label.set_xalign(0.0)
        general_grid.attach(freq_label, 0, row, 1, 1)
        self.frequency_penalty_adjustment = Gtk.Adjustment(
            lower=-2.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.frequency_penalty_spin = Gtk.SpinButton(
            adjustment=self.frequency_penalty_adjustment, digits=2
        )
        self.frequency_penalty_spin.set_increments(0.05, 0.1)
        self.frequency_penalty_spin.set_hexpand(True)
        general_grid.attach(self.frequency_penalty_spin, 1, row, 1, 1)

        row += 1
        presence_label = Gtk.Label(label="Presence Penalty:")
        presence_label.set_xalign(0.0)
        general_grid.attach(presence_label, 0, row, 1, 1)
        self.presence_penalty_adjustment = Gtk.Adjustment(
            lower=-2.0, upper=2.0, step_increment=0.05, page_increment=0.1, value=0.0
        )
        self.presence_penalty_spin = Gtk.SpinButton(
            adjustment=self.presence_penalty_adjustment, digits=2
        )
        self.presence_penalty_spin.set_increments(0.05, 0.1)
        self.presence_penalty_spin.set_hexpand(True)
        general_grid.attach(self.presence_penalty_spin, 1, row, 1, 1)

        row += 1
        tokens_label = Gtk.Label(label="Max Tokens:")
        tokens_label.set_xalign(0.0)
        general_grid.attach(tokens_label, 0, row, 1, 1)
        self.max_tokens_adjustment = Gtk.Adjustment(
            lower=1, upper=128000, step_increment=128, page_increment=512, value=4000
        )
        self.max_tokens_spin = Gtk.SpinButton(adjustment=self.max_tokens_adjustment, digits=0)
        self.max_tokens_spin.set_increments(128, 512)
        self.max_tokens_spin.set_hexpand(True)
        general_grid.attach(self.max_tokens_spin, 1, row, 1, 1)

        row += 1
        org_label = Gtk.Label(label="Organization (optional):")
        org_label.set_xalign(0.0)
        general_grid.attach(org_label, 0, row, 1, 1)
        self.organization_entry = Gtk.Entry()
        self.organization_entry.set_hexpand(True)
        self.organization_entry.set_placeholder_text("org-1234")
        general_grid.attach(self.organization_entry, 1, row, 1, 1)

        row += 1
        self.stream_toggle = Gtk.CheckButton(label="Enable streaming responses")
        self.stream_toggle.set_halign(Gtk.Align.START)
        general_grid.attach(self.stream_toggle, 0, row, 2, 1)

        self._append_settings_page(general_box, "General")

        tools_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin=6,
        )
        tools_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        tools_box.append(tools_grid)

        tools_row = 0

        self.function_call_toggle = Gtk.CheckButton(label="Allow automatic tool calls")
        self.function_call_toggle.set_halign(Gtk.Align.START)
        tools_grid.attach(self.function_call_toggle, 0, tools_row, 2, 1)
        if hasattr(self.function_call_toggle, "connect"):
            self.function_call_toggle.connect("toggled", self._on_function_call_toggle_toggled)
        tools_row += 1

        self.parallel_tool_calls_toggle = Gtk.CheckButton(label="Allow parallel tool calls")
        self.parallel_tool_calls_toggle.set_halign(Gtk.Align.START)
        tools_grid.attach(self.parallel_tool_calls_toggle, 0, tools_row, 2, 1)
        tools_row += 1

        self.require_tool_toggle = Gtk.CheckButton(label="Require a tool call before responding")
        self.require_tool_toggle.set_halign(Gtk.Align.START)
        tools_grid.attach(self.require_tool_toggle, 0, tools_row, 2, 1)

        self._append_settings_page(tools_box, "Tools")

        audio_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin=6,
        )
        audio_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        audio_box.append(audio_grid)

        audio_row = 0

        self.audio_reply_toggle = Gtk.CheckButton(label="Enable audio replies")
        self.audio_reply_toggle.set_halign(Gtk.Align.START)
        if hasattr(self.audio_reply_toggle, "connect"):
            self.audio_reply_toggle.connect("toggled", self._on_audio_toggle_toggled)
        audio_grid.attach(self.audio_reply_toggle, 0, audio_row, 2, 1)
        audio_row += 1

        audio_voice_label = Gtk.Label(label="Voice:")
        audio_voice_label.set_xalign(0.0)
        audio_grid.attach(audio_voice_label, 0, audio_row, 1, 1)
        self.audio_voice_combo = Gtk.ComboBoxText()
        for option in self._audio_voice_options:
            self.audio_voice_combo.append_text(option)
        self.audio_voice_combo.set_hexpand(True)
        audio_grid.attach(self.audio_voice_combo, 1, audio_row, 1, 1)
        audio_row += 1

        audio_format_label = Gtk.Label(label="Audio format:")
        audio_format_label.set_xalign(0.0)
        audio_grid.attach(audio_format_label, 0, audio_row, 1, 1)
        self.audio_format_combo = Gtk.ComboBoxText()
        for option in self._audio_format_options:
            self.audio_format_combo.append_text(option)
        self.audio_format_combo.set_hexpand(True)
        audio_grid.attach(self.audio_format_combo, 1, audio_row, 1, 1)

        self._append_settings_page(audio_box, "Audio")

        advanced_box = create_box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin=6,
        )
        advanced_grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        advanced_box.append(advanced_grid)

        advanced_row = 0

        max_output_label = Gtk.Label(label="Max Output Tokens (reasoning):")
        max_output_label.set_xalign(0.0)
        advanced_grid.attach(max_output_label, 0, advanced_row, 1, 1)

        self.max_output_tokens_adjustment = Gtk.Adjustment(
            lower=0, upper=128000, step_increment=128, page_increment=512, value=0
        )
        self.max_output_tokens_spin = Gtk.SpinButton(
            adjustment=self.max_output_tokens_adjustment, digits=0
        )
        self.max_output_tokens_spin.set_increments(128, 512)
        self.max_output_tokens_spin.set_hexpand(True)
        if hasattr(self.max_output_tokens_spin, "set_tooltip_text"):
            self.max_output_tokens_spin.set_tooltip_text(
                "Set to 0 to use the model default."
            )
        advanced_grid.attach(self.max_output_tokens_spin, 1, advanced_row, 1, 1)
        advanced_row += 1

        reasoning_label = Gtk.Label(label="Reasoning Effort:")
        reasoning_label.set_xalign(0.0)
        advanced_grid.attach(reasoning_label, 0, advanced_row, 1, 1)

        self.reasoning_effort_combo = Gtk.ComboBoxText()
        for option in self._reasoning_effort_values:
            self.reasoning_effort_combo.append_text(option)
        self.reasoning_effort_combo.set_hexpand(True)
        advanced_grid.attach(self.reasoning_effort_combo, 1, advanced_row, 1, 1)
        advanced_row += 1

        self.json_mode_toggle = Gtk.CheckButton(label="Force JSON responses")
        self.json_mode_toggle.set_halign(Gtk.Align.START)
        advanced_grid.attach(self.json_mode_toggle, 0, advanced_row, 2, 1)
        advanced_row += 1

        self.code_interpreter_toggle = Gtk.CheckButton(label="Enable OpenAI Code Interpreter")
        self.code_interpreter_toggle.set_halign(Gtk.Align.START)
        advanced_grid.attach(self.code_interpreter_toggle, 0, advanced_row, 2, 1)
        advanced_row += 1

        self.file_search_toggle = Gtk.CheckButton(label="Enable OpenAI File Search")
        self.file_search_toggle.set_halign(Gtk.Align.START)
        advanced_grid.attach(self.file_search_toggle, 0, advanced_row, 2, 1)
        advanced_row += 1

        schema_label = Gtk.Label(label="Response JSON Schema (optional):")
        schema_label.set_xalign(0.0)
        advanced_grid.attach(schema_label, 0, advanced_row, 2, 1)
        advanced_row += 1

        self._json_schema_buffer = None
        self._json_schema_widget = None

        text_view_cls = getattr(Gtk, "TextView", None)
        text_buffer_cls = getattr(Gtk, "TextBuffer", None)
        scrolled_cls = getattr(Gtk, "ScrolledWindow", None)

        if text_view_cls is not None and text_buffer_cls is not None:
            if scrolled_cls is not None:
                schema_scroller = scrolled_cls()
                if hasattr(schema_scroller, "set_policy"):
                    schema_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
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

            advanced_grid.attach(schema_scroller, 0, advanced_row, 2, 1)
        else:
            self._json_schema_widget = Gtk.Entry()
            if hasattr(self._json_schema_widget, "set_placeholder_text"):
                self._json_schema_widget.set_placeholder_text(
                    '{"name": "atlas_response", "schema": {"type": "object"}}'
                )
            if hasattr(self._json_schema_widget, "set_hexpand"):
                self._json_schema_widget.set_hexpand(True)
            advanced_grid.attach(self._json_schema_widget, 0, advanced_row, 2, 1)

        advanced_row += 1

        self.json_schema_feedback_label = Gtk.Label(
            label="Optional: provide a JSON schema to enforce structured responses."
        )
        self.json_schema_feedback_label.set_xalign(0.0)
        advanced_grid.attach(self.json_schema_feedback_label, 0, advanced_row, 2, 1)
        advanced_row += 1

        base_url_label = Gtk.Label(label="Custom Base URL:")
        base_url_label.set_xalign(0.0)
        advanced_grid.attach(base_url_label, 0, advanced_row, 1, 1)

        self.base_url_entry = Gtk.Entry()
        self.base_url_entry.set_hexpand(True)
        self.base_url_entry.set_placeholder_text("https://api.openai.com/v1")
        if hasattr(self.base_url_entry, "connect"):
            self.base_url_entry.connect("changed", self._on_base_url_changed)
        advanced_grid.attach(self.base_url_entry, 1, advanced_row, 1, 1)

        self._append_settings_page(advanced_box, "Advanced")

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

    def _append_settings_page(self, child: Gtk.Widget, title: str) -> None:
        if not hasattr(self, "settings_notebook") or self.settings_notebook is None:
            return

        label = Gtk.Label(label=title)
        if hasattr(label, "set_xalign"):
            label.set_xalign(0.0)

        notebook = self.settings_notebook
        append_page = getattr(notebook, "append_page", None)
        if callable(append_page):
            append_page(child, label)
            return

        if hasattr(notebook, "append"):
            wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
            wrapper.append(label)
            wrapper.append(child)
            notebook.append(wrapper)
            return

        if hasattr(notebook, "set_child"):
            notebook.set_child(child)

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
        max_output = settings.get("max_output_tokens")
        if max_output is None:
            self.max_output_tokens_spin.set_value(0)
        else:
            self.max_output_tokens_spin.set_value(float(max_output))
        effort_value = (settings.get("reasoning_effort") or "medium").lower()
        try:
            effort_index = self._reasoning_effort_values.index(effort_value)
        except ValueError:
            effort_index = self._reasoning_effort_values.index("medium")
        self.reasoning_effort_combo.set_active(effort_index)
        self.stream_toggle.set_active(bool(settings.get("stream", True)))
        self.function_call_toggle.set_active(bool(settings.get("function_calling", True)))
        self.parallel_tool_calls_toggle.set_active(bool(settings.get("parallel_tool_calls", True)))
        tool_choice_value = settings.get("tool_choice")
        self.require_tool_toggle.set_active(str(tool_choice_value).lower() == "required")
        audio_enabled = bool(settings.get("audio_enabled", False))
        self.audio_reply_toggle.set_active(audio_enabled)
        self._select_combo_value(
            self.audio_voice_combo,
            settings.get("audio_voice"),
            fallback_value=self._audio_voice_options[0],
        )
        if hasattr(self.audio_voice_combo, "get_active_text"):
            current_voice = self.audio_voice_combo.get_active_text()
            target_voice = settings.get("audio_voice")
            if (
                not current_voice
                and isinstance(target_voice, str)
                and target_voice in self._audio_voice_options
            ):
                self.audio_voice_combo.set_active(self._audio_voice_options.index(target_voice))
        self._select_combo_value(
            self.audio_format_combo,
            settings.get("audio_format"),
            fallback_value=self._audio_format_options[0],
        )
        if hasattr(self.audio_format_combo, "get_active_text"):
            current_format = self.audio_format_combo.get_active_text()
            target_format = settings.get("audio_format")
            if (
                not current_format
                and isinstance(target_format, str)
                and target_format in self._audio_format_options
            ):
                self.audio_format_combo.set_active(
                    self._audio_format_options.index(target_format)
                )
        self._update_audio_controls_state()
        if hasattr(self, "json_mode_toggle"):
            self.json_mode_toggle.set_active(bool(settings.get("json_mode", False)))
        if hasattr(self, "code_interpreter_toggle"):
            self.code_interpreter_toggle.set_active(
                bool(settings.get("enable_code_interpreter", False))
            )
        if hasattr(self, "file_search_toggle"):
            self.file_search_toggle.set_active(
                bool(settings.get("enable_file_search", False))
            )
        self._update_tool_controls_state()
        schema_payload = settings.get("json_schema") if isinstance(settings, dict) else None
        formatted_schema = self._format_json_schema(schema_payload)
        self._write_json_schema_text(formatted_schema)
        if formatted_schema:
            self._update_json_schema_feedback(True, "JSON schema loaded from saved settings.")
        else:
            self._update_json_schema_feedback(
                True, "Optional: provide a JSON schema to enforce structured responses."
            )
        self.organization_entry.set_text(settings.get("organization") or "")

        self._begin_model_refresh(settings)

    def _get_settings_snapshot(self) -> Dict[str, Any]:
        atlas = getattr(self, "ATLAS", None)
        if atlas is None or not hasattr(atlas, "get_openai_llm_settings"):
            logger.error("ATLAS facade for OpenAI settings is unavailable.")
            return {}

        try:
            settings = atlas.get_openai_llm_settings() or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read OpenAI settings from ATLAS: %s", exc, exc_info=True)
            return {}

        if isinstance(settings, dict):
            return dict(settings)

        logger.error("Unexpected OpenAI settings payload: %r", settings)
        return {}

    def _format_json_schema(self, payload: Any) -> str:
        if isinstance(payload, str):
            text = payload.strip()
            if text:
                return text
            return ""
        if isinstance(payload, dict) and payload:
            try:
                return json.dumps(payload, indent=2, sort_keys=True)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive logging
                logger.debug("Unable to format stored JSON schema: %s", exc)
        return ""

    def _write_json_schema_text(self, text: str) -> None:
        self._json_schema_text_cache = text or ""
        buffer = getattr(self, "_json_schema_buffer", None)
        if buffer is not None and hasattr(buffer, "set_text"):
            try:
                buffer.set_text(self._json_schema_text_cache)
                return
            except Exception:  # pragma: no cover - safety for stubbed environments
                pass

        widget = getattr(self, "_json_schema_widget", None)
        if widget is not None and hasattr(widget, "set_text"):
            widget.set_text(self._json_schema_text_cache)
            return

        self._json_schema_text_cache = text or ""

    def _read_json_schema_text(self) -> str:
        buffer = getattr(self, "_json_schema_buffer", None)
        if buffer is not None:
            get_text = getattr(buffer, "get_text", None)
            if callable(get_text):
                try:
                    start_iter = buffer.get_start_iter()
                    end_iter = buffer.get_end_iter()
                    value = get_text(start_iter, end_iter, True)
                    return value if isinstance(value, str) else str(value or "")
                except Exception:  # pragma: no cover - guard for stubbed GTK
                    pass

        widget = getattr(self, "_json_schema_widget", None)
        if widget is not None and hasattr(widget, "get_text"):
            try:
                value = widget.get_text()
                return value if isinstance(value, str) else str(value or "")
            except Exception:  # pragma: no cover - guard for stubbed GTK
                pass

        cache = getattr(self, "_json_schema_text_cache", "")
        return cache if isinstance(cache, str) else str(cache or "")

    def _update_json_schema_feedback(self, valid: bool, message: str) -> None:
        self._json_schema_is_valid = valid
        label = getattr(self, "json_schema_feedback_label", None)
        if label is None:
            return

        if hasattr(label, "set_label"):
            label.set_label(message)
        else:  # pragma: no cover - compatibility with GTK stubs
            label.label = message

        if hasattr(label, "add_css_class"):
            if valid and hasattr(label, "remove_css_class"):
                label.remove_css_class("error")
            elif not valid:
                label.add_css_class("error")

    def _refresh_api_key_status(self) -> None:
        atlas = getattr(self, "ATLAS", None)
        status_text = "Credential status is unavailable."

        if atlas is not None and hasattr(atlas, "get_provider_api_key_status"):
            try:
                payload = atlas.get_provider_api_key_status("OpenAI") or {}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Unable to determine OpenAI API key status: %s", exc, exc_info=True)
            else:
                has_key = bool(payload.get("has_key"))
                metadata = payload.get("metadata") if isinstance(payload, dict) else {}
                hint = ""
                if isinstance(metadata, dict):
                    hint = metadata.get("hint") or ""

                if has_key:
                    suffix = f" ({hint})" if hint else ""
                    status_text = f"An API key is saved for OpenAI.{suffix}"
                else:
                    status_text = "No API key saved for OpenAI."

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

    def _select_combo_value(
        self,
        combo: Gtk.ComboBoxText,
        value: Optional[str],
        *,
        fallback_value: Optional[str] = None,
    ) -> None:
        model = getattr(combo, "get_model", None)
        options: List[str] = []
        if callable(model):
            store = model()
            if store is not None:
                get_n_items = getattr(store, "get_n_items", None)
                get_item = getattr(store, "get_item", None)
                if callable(get_n_items) and callable(get_item):
                    for index in range(get_n_items()):
                        item = get_item(index)
                        getter = getattr(item, "get_string", None)
                        if callable(getter):
                            text_value = getter()
                            if isinstance(text_value, str):
                                options.append(text_value)
                else:
                    foreach = getattr(store, "foreach", None)
                    if callable(foreach):
                        def _collect(model_obj, _path, tree_iter, _user_data):
                            value_getter = getattr(model_obj, "get_value", None)
                            text_value = None
                            if callable(value_getter):
                                text_value = value_getter(tree_iter, 0)
                            elif isinstance(tree_iter, (list, tuple)) and tree_iter:
                                possible = tree_iter[0]
                                if isinstance(possible, str):
                                    text_value = possible
                            if isinstance(text_value, str):
                                options.append(text_value)
                            return False

                        foreach(_collect, None)
                    else:
                        iterator = None
                        if hasattr(store, "__iter__"):
                            try:
                                iterator = iter(store)
                            except TypeError:
                                iterator = None
                        if iterator is not None:
                            for row in iterator:
                                text_value = None
                                if isinstance(row, (list, tuple)) and row:
                                    candidate = row[0]
                                    if isinstance(candidate, str):
                                        text_value = candidate
                                elif isinstance(row, str):
                                    text_value = row
                                if isinstance(text_value, str):
                                    options.append(text_value)
        else:  # pragma: no cover - compatibility with GTK stubs
            stored_items = getattr(combo, "_items", None)
            if isinstance(stored_items, list) and stored_items:
                options = [str(item) for item in stored_items if isinstance(item, str)]
            else:
                active_text = getattr(combo, "get_active_text", None)
                if callable(active_text):
                    existing = active_text()
                    if isinstance(existing, str):
                        options = [existing]

        normalized_options = [entry.lower() for entry in options]
        target = (value or "").strip().lower()
        try:
            index = normalized_options.index(target) if target else -1
        except ValueError:
            index = -1

        if index >= 0:
            combo.set_active(index)
        elif fallback_value and fallback_value.lower() in normalized_options:
            combo.set_active(normalized_options.index(fallback_value.lower()))
        elif options:
            combo.set_active(0)

    def _update_audio_controls_state(self) -> None:
        is_enabled = self.audio_reply_toggle.get_active()
        self.audio_voice_combo.set_sensitive(is_enabled)
        self.audio_format_combo.set_sensitive(is_enabled)

    def _on_audio_toggle_toggled(self, _toggle: Gtk.CheckButton) -> None:
        self._update_audio_controls_state()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _begin_model_refresh(self, settings: Dict[str, Any]) -> None:
        placeholder = settings.get("model") or self._model_placeholder
        self.model_combo.remove_all()
        self.model_combo.append_text(placeholder)
        self.model_combo.set_active(0)

        atlas = getattr(self, "ATLAS", None)
        if atlas is None or not hasattr(atlas, "run_in_background") or not hasattr(atlas, "list_openai_models"):
            logger.error("ATLAS background runner is unavailable; cannot refresh models.")
            self._apply_model_results([], "Model discovery is unavailable.", settings.get("model"))
            return

        base_override = self._stored_base_url
        organization = settings.get("organization")

        def _handle_success(payload: Any) -> None:
            GLib.idle_add(self._apply_model_payload, payload, settings.get("model"))

        def _handle_error(exc: Exception) -> None:
            logger.error("Failed to refresh OpenAI models: %s", exc, exc_info=True)
            GLib.idle_add(
                self._apply_model_results,
                [],
                str(exc),
                settings.get("model"),
            )

        try:
            atlas.run_in_background(
                lambda: atlas.list_openai_models(
                    base_url=base_override if base_override is not None else settings.get("base_url"),
                    organization=organization,
                ),
                on_success=_handle_success,
                on_error=_handle_error,
                thread_name="openai-model-refresh",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to schedule OpenAI model refresh: %s", exc, exc_info=True)
            self._apply_model_results([], str(exc), settings.get("model"))

    def _load_cached_models(self) -> List[str]:
        atlas = getattr(self, "ATLAS", None)
        provider_manager = getattr(atlas, "provider_manager", None)
        model_manager = getattr(provider_manager, "model_manager", None)

        if model_manager is None:
            return []

        try:
            available = model_manager.get_available_models("OpenAI")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to read cached OpenAI models: %s", exc, exc_info=True)
            return []

        if isinstance(available, dict):
            models = available.get("OpenAI", [])
        else:
            models = available

        seen: set[str] = set()
        normalized: List[str] = []
        for entry in models:
            if isinstance(entry, str):
                name = entry.strip()
                if name and name not in seen:
                    normalized.append(name)
                    seen.add(name)

        return normalized

    def _apply_model_payload(self, payload: Any, preferred_model: Optional[str]) -> bool:
        models: List[str] = []
        error: Optional[str] = None

        if isinstance(payload, dict):
            models = payload.get("models") or []
            error = payload.get("error")
        elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
            models = payload[0] or []
            error = payload[1]
        elif payload:
            error = str(payload)

        if not isinstance(models, list):
            if isinstance(models, (tuple, set)):
                models = list(models)
            else:
                models = []

        if error is not None and not isinstance(error, str):
            error = str(error)

        return self._apply_model_results(models, error, preferred_model)

    def _apply_model_results(
        self, models: List[str], error: Optional[str], preferred_model: Optional[str]
    ) -> bool:
        cached_models = self._load_cached_models()

        def _normalize(names: List[str]) -> List[str]:
            seen: set[str] = set()
            ordered: List[str] = []
            for entry in names:
                if isinstance(entry, str):
                    name = entry.strip()
                    if name and name not in seen:
                        ordered.append(name)
                        seen.add(name)
            return ordered

        combined_models = _normalize(models)
        combined_models.extend([m for m in cached_models if m not in combined_models])

        preferred = preferred_model.strip() if isinstance(preferred_model, str) else ""

        if error:
            detail = error if isinstance(error, str) else str(error)
            logger.warning(
                "Falling back to cached OpenAI models because fetching failed: %s", detail
            )
            fallback_models = list(combined_models) if combined_models else list(cached_models)
            if preferred:
                if preferred in fallback_models:
                    fallback_models = [preferred] + [m for m in fallback_models if m != preferred]
                else:
                    fallback_models = [preferred] + fallback_models
            if not fallback_models:
                fallback_models = [preferred or "gpt-4o"]

            self.model_combo.remove_all()
            for name in fallback_models:
                self.model_combo.append_text(name)
            self.model_combo.set_active(0)
            self._show_message("Model Load Failed", detail, Gtk.MessageType.ERROR)
            self._available_models = list(fallback_models)
            return False

        if preferred and preferred not in combined_models:
            combined_models.insert(0, preferred)

        if not combined_models:
            combined_models = [preferred or "gpt-4o"]

        self.model_combo.remove_all()
        active_index = 0
        for idx, name in enumerate(combined_models):
            self.model_combo.append_text(name)
            if preferred and name == preferred:
                active_index = idx

        self.model_combo.set_active(active_index)
        self._available_models = list(combined_models)
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

        def handle_success(result):
            GLib.idle_add(self._handle_api_key_save_result, result)

        def handle_error(exc: Exception) -> None:
            logger.error("Error saving OpenAI API key: %s", exc, exc_info=True)
            GLib.idle_add(
                self._handle_api_key_save_result,
                {"success": False, "error": str(exc)},
            )

        helper = getattr(self.ATLAS, "update_provider_api_key_in_background", None)
        if callable(helper):
            try:
                helper(
                    "OpenAI",
                    api_key,
                    on_success=handle_success,
                    on_error=handle_error,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Unable to schedule OpenAI API key save: %s", exc, exc_info=True)
                self._show_message(
                    "Error",
                    f"Failed to save API key: {str(exc)}",
                    Gtk.MessageType.ERROR,
                )
            return

        updater = getattr(self.ATLAS, "update_provider_api_key", None)
        if not callable(updater):
            self._show_message(
                "Error",
                "Saving API keys is not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        try:
            future = run_async_in_thread(
                lambda: updater("OpenAI", api_key),
                on_success=handle_success,
                on_error=handle_error,
                logger=logger,
                thread_name="openai-api-key-fallback",
            )
            future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Unable to start fallback API key save task: %s", exc, exc_info=True)
            self._show_message(
                "Error",
                f"Failed to save API key: {str(exc)}",
                Gtk.MessageType.ERROR,
            )

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

        if model == self._model_placeholder:
            self._show_message(
                "Error",
                f"Please choose a default model instead of \"{self._model_placeholder}\".",
                Gtk.MessageType.ERROR,
            )
            return

        if self._available_models and model not in self._available_models:
            self._show_message(
                "Error",
                "The selected default model is not available. Please choose another option.",
                Gtk.MessageType.ERROR,
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

        max_output_value = self.max_output_tokens_spin.get_value_as_int()
        reasoning_effort = self.reasoning_effort_combo.get_active_text() or "medium"

        raw_schema_text = self._read_json_schema_text().strip()
        if raw_schema_text:
            try:
                parsed_schema = json.loads(raw_schema_text)
            except json.JSONDecodeError as exc:
                message = f"Invalid JSON schema: {exc.msg}"
                self._update_json_schema_feedback(False, message)
                self._show_message("Error", message, Gtk.MessageType.ERROR)
                return
            if not isinstance(parsed_schema, dict):
                message = "JSON schema must be a JSON object at the top level."
                self._update_json_schema_feedback(False, message)
                self._show_message("Error", message, Gtk.MessageType.ERROR)
                return
            self._update_json_schema_feedback(True, "JSON schema looks valid.")
        else:
            self._update_json_schema_feedback(
                True, "Optional: provide a JSON schema to enforce structured responses."
            )

        payload = {
            "model": model,
            "temperature": self.temperature_spin.get_value(),
            "top_p": self.top_p_spin.get_value(),
            "frequency_penalty": self.frequency_penalty_spin.get_value(),
            "presence_penalty": self.presence_penalty_spin.get_value(),
            "max_tokens": self.max_tokens_spin.get_value_as_int(),
            "max_output_tokens": max_output_value if max_output_value > 0 else None,
            "stream": self.stream_toggle.get_active(),
            "function_calling": self.function_call_toggle.get_active(),
            "parallel_tool_calls": self.parallel_tool_calls_toggle.get_active(),
            "base_url": base_url,
            "organization": self.organization_entry.get_text().strip() or None,
            "reasoning_effort": reasoning_effort.lower(),
            "json_mode": self.json_mode_toggle.get_active() if hasattr(self, "json_mode_toggle") else False,
            "json_schema": raw_schema_text if raw_schema_text else "",
            "enable_code_interpreter": (
                self.code_interpreter_toggle.get_active()
                if hasattr(self, "code_interpreter_toggle")
                else False
            ),
            "enable_file_search": (
                self.file_search_toggle.get_active()
                if hasattr(self, "file_search_toggle")
                else False
            ),
            "audio_enabled": self.audio_reply_toggle.get_active(),
            "audio_voice": self.audio_voice_combo.get_active_text(),
            "audio_format": self.audio_format_combo.get_active_text(),
        }

        function_calling_enabled = payload["function_calling"]
        if not function_calling_enabled:
            payload["parallel_tool_calls"] = False
            payload["tool_choice"] = "none"
            payload["enable_code_interpreter"] = False
            payload["enable_file_search"] = False
        elif self.require_tool_toggle.get_active():
            payload["tool_choice"] = "required"
        else:
            payload["tool_choice"] = None

        try:
            result = self.ATLAS.set_openai_llm_settings(**payload)
        except Exception as exc:
            logger.error("Error saving OpenAI settings: %s", exc, exc_info=True)
            self._show_message("Error", str(exc), Gtk.MessageType.ERROR)
            return

        if isinstance(result, dict) and result.get("success"):
            message = result.get("message", "OpenAI settings saved.")
            self._show_message("Success", message, Gtk.MessageType.INFO)
            if raw_schema_text:
                self._update_json_schema_feedback(True, "JSON schema saved successfully.")
            else:
                self._update_json_schema_feedback(
                    True, "Optional: provide a JSON schema to enforce structured responses."
                )
            self.close()
            return

        if isinstance(result, dict):
            detail = result.get("error") or result.get("message") or "Unable to save OpenAI settings."
        elif result is None:
            detail = "Unable to save OpenAI settings."
        else:
            detail = str(result)

        if raw_schema_text:
            self._update_json_schema_feedback(False, detail)
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

    def _on_function_call_toggle_toggled(self, _button: Gtk.CheckButton):
        self._update_tool_controls_state()

    def _update_tool_controls_state(self) -> None:
        enabled = self.function_call_toggle.get_active()
        widgets = [self.parallel_tool_calls_toggle, self.require_tool_toggle]
        for attr in ("code_interpreter_toggle", "file_search_toggle"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widgets.append(widget)

        for widget in widgets:
            if hasattr(widget, "set_sensitive"):
                widget.set_sensitive(enabled)
        if not enabled:
            if self.parallel_tool_calls_toggle.get_active():
                self.parallel_tool_calls_toggle.set_active(False)
            if self.require_tool_toggle.get_active():
                self.require_tool_toggle.set_active(False)
            for attr in ("code_interpreter_toggle", "file_search_toggle"):
                widget = getattr(self, attr, None)
                if widget is not None and widget.get_active():
                    widget.set_active(False)
