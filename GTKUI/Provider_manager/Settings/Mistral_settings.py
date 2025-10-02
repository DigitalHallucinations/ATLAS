from __future__ import annotations

"""GTK window for configuring default settings of the Mistral provider."""

import json
import logging
import os
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Utils.utils import apply_css, create_box

logger = logging.getLogger(__name__)


class MistralSettingsWindow(Gtk.Window):
    """Collect and persist preferences specific to the Mistral chat provider."""

    def __init__(self, ATLAS, config_manager, parent_window):
        super().__init__(title="Mistral Settings")
        apply_css()
        style_context = self.get_style_context()
        style_context.add_class("chat-page")
        style_context.add_class("sidebar")

        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window
        if parent_window is not None:
            self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_default_size(420, 360)

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

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        grid.set_margin_top(6)
        grid.set_margin_bottom(6)
        grid.set_margin_start(6)
        grid.set_margin_end(6)
        container.append(grid)

        row = 0
        model_label = Gtk.Label(label="Default model:")
        model_label.set_xalign(0.0)
        grid.attach(model_label, 0, row, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_hexpand(True)
        if hasattr(self.model_combo, "connect"):
            self.model_combo.connect("changed", self._on_model_combo_changed)
        grid.attach(self.model_combo, 1, row, 1, 1)

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
        stream_label = Gtk.Label(label="Streaming:")
        stream_label.set_xalign(0.0)
        grid.attach(stream_label, 0, row, 1, 1)

        self.stream_toggle = Gtk.CheckButton(label="Stream responses by default")
        self.stream_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.stream_toggle, 1, row, 1, 1)

        row += 1
        parallel_label = Gtk.Label(label="Parallel tool calls:")
        parallel_label.set_xalign(0.0)
        grid.attach(parallel_label, 0, row, 1, 1)

        self.parallel_tool_calls_toggle = Gtk.CheckButton(label="Allow parallel tool calls")
        self.parallel_tool_calls_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.parallel_tool_calls_toggle, 1, row, 1, 1)

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
        tool_choice_label = Gtk.Label(label="Tool choice preference:")
        tool_choice_label.set_xalign(0.0)
        grid.attach(tool_choice_label, 0, row, 1, 1)

        self.tool_choice_entry = Gtk.Entry()
        self.tool_choice_entry.set_hexpand(True)
        self.tool_choice_entry.set_placeholder_text("auto / none / JSON payload")
        grid.attach(self.tool_choice_entry, 1, row, 1, 1)

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

        save_button = Gtk.Button(label="Save Defaults")
        save_button.connect("clicked", self.on_save_clicked)
        button_box.append(save_button)

        close_button = Gtk.Button(label="Close")
        close_button.connect("clicked", lambda *_args: self.close())
        button_box.append(close_button)

    def refresh_settings(self, *, clear_message: bool = False) -> None:
        settings = self.config_manager.get_mistral_llm_settings()
        self._current_settings = dict(settings)

        self._refresh_model_options()
        self._select_model(settings.get("model", "") or "")
        self.temperature_spin.set_value(float(settings.get("temperature", 0.0)))
        self.top_p_spin.set_value(float(settings.get("top_p", 1.0)))
        self.frequency_penalty_spin.set_value(float(settings.get("frequency_penalty", 0.0)))
        self.presence_penalty_spin.set_value(float(settings.get("presence_penalty", 0.0)))

        max_tokens = settings.get("max_tokens")
        self.max_tokens_spin.set_value(float(max_tokens or 0))

        self.safe_prompt_toggle.set_active(bool(settings.get("safe_prompt", False)))
        self.stream_toggle.set_active(bool(settings.get("stream", True)))
        self.parallel_tool_calls_toggle.set_active(
            bool(settings.get("parallel_tool_calls", True))
        )

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
        if isinstance(tool_choice, Mapping):
            tool_choice_text = json.dumps(tool_choice, sort_keys=True)
        elif tool_choice is None:
            tool_choice_text = ""
        else:
            tool_choice_text = str(tool_choice)
        self.tool_choice_entry.set_text(tool_choice_text)

        tokens = settings.get("stop_sequences") or []
        self.stop_sequences_entry.set_text(", ".join(tokens))

        if clear_message:
            self._last_message = None
            self._update_label(self.message_label, "")

    def _set_message(self, title: str, body: str, message_type: Gtk.MessageType) -> None:
        self._last_message = (title, body, message_type)
        if body:
            display = f"{title}: {body}"
        else:
            display = title
        self._update_label(self.message_label, display)

    def _update_label(self, widget: Gtk.Widget, text: str) -> None:
        if hasattr(widget, "set_text"):
            widget.set_text(text)
        elif hasattr(widget, "set_label"):
            widget.set_label(text)
        else:  # pragma: no cover - fallback for stubs/tests
            setattr(widget, "label", text)

    def _parse_tool_choice(self, text: str) -> Any:
        cleaned = text.strip()
        if not cleaned:
            return None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return cleaned
        if isinstance(parsed, (dict, list)):
            return parsed
        return cleaned

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
                tool_choice=self._parse_tool_choice(self.tool_choice_entry.get_text()),
                parallel_tool_calls=self.parallel_tool_calls_toggle.get_active(),
                stop_sequences=self._parse_stop_sequences(),
                json_mode=self.json_mode_toggle.get_active(),
                json_schema=raw_schema_text if raw_schema_text else "",
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

    def _load_known_models(self) -> List[str]:
        models: List[str] = []

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

        if models:
            return self._deduplicate(models)

        search_paths: List[str] = []
        root_path = None
        if hasattr(self.config_manager, "get_app_root"):
            try:
                root_path = self.config_manager.get_app_root()
            except Exception:  # pragma: no cover - defensive fallback
                root_path = None
        if root_path:
            search_paths.append(
                os.path.join(
                    root_path, "modules", "Providers", "Mistral", "M_models.json"
                )
            )

        module_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        search_paths.append(
            os.path.join(module_root, "modules", "Providers", "Mistral", "M_models.json")
        )

        for path in search_paths:
            if not path:
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to read Mistral models from %s: %s", path, exc)
                continue

            entries = []
            if isinstance(payload, dict):
                entries = payload.get("models", [])
            elif isinstance(payload, list):
                entries = payload

            if isinstance(entries, list):
                models = [
                    entry.strip()
                    for entry in entries
                    if isinstance(entry, str) and entry.strip()
                ]
            else:
                models = []

            if models:
                return self._deduplicate(models)

        return []

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

    def _refresh_model_options(self) -> None:
        latest = self._load_known_models()
        if not self._model_options_initialized or latest != self._available_models:
            self._populate_model_combo(latest)

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
        return self.custom_model_entry.get_text().strip()

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
