from __future__ import annotations

"""GTK window for configuring default settings of the Mistral provider."""

import json
import logging
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

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

        self.model_entry = Gtk.Entry()
        self.model_entry.set_hexpand(True)
        self.model_entry.set_placeholder_text("e.g. mistral-large-latest")
        grid.attach(self.model_entry, 1, row, 1, 1)

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
        parallel_label = Gtk.Label(label="Parallel tool calls:")
        parallel_label.set_xalign(0.0)
        grid.attach(parallel_label, 0, row, 1, 1)

        self.parallel_tool_calls_toggle = Gtk.CheckButton(label="Allow parallel tool calls")
        self.parallel_tool_calls_toggle.set_halign(Gtk.Align.START)
        grid.attach(self.parallel_tool_calls_toggle, 1, row, 1, 1)

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

        self.model_entry.set_text(settings.get("model", "") or "")
        self.temperature_spin.set_value(float(settings.get("temperature", 0.0)))
        self.top_p_spin.set_value(float(settings.get("top_p", 1.0)))
        self.frequency_penalty_spin.set_value(float(settings.get("frequency_penalty", 0.0)))
        self.presence_penalty_spin.set_value(float(settings.get("presence_penalty", 0.0)))

        max_tokens = settings.get("max_tokens")
        self.max_tokens_spin.set_value(float(max_tokens or 0))

        self.safe_prompt_toggle.set_active(bool(settings.get("safe_prompt", False)))
        self.parallel_tool_calls_toggle.set_active(
            bool(settings.get("parallel_tool_calls", True))
        )

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

    def _parse_optional_int(self, value: str) -> Optional[int]:
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except ValueError as exc:
            raise ValueError("Random seed must be an integer.") from exc

    def on_save_clicked(self, _button) -> None:
        model = self.model_entry.get_text().strip()
        max_tokens_value = self.max_tokens_spin.get_value_as_int()
        max_tokens = max_tokens_value if max_tokens_value > 0 else None

        try:
            random_seed = self._parse_optional_int(self.random_seed_entry.get_text())
            saved = self.config_manager.set_mistral_llm_settings(
                model=model,
                temperature=self.temperature_spin.get_value(),
                top_p=self.top_p_spin.get_value(),
                max_tokens=max_tokens,
                safe_prompt=self.safe_prompt_toggle.get_active(),
                random_seed=random_seed,
                frequency_penalty=self.frequency_penalty_spin.get_value(),
                presence_penalty=self.presence_penalty_spin.get_value(),
                tool_choice=self._parse_tool_choice(self.tool_choice_entry.get_text()),
                parallel_tool_calls=self.parallel_tool_calls_toggle.get_active(),
            )
        except Exception as exc:
            logger.error("Failed to save Mistral settings: %s", exc, exc_info=True)
            message = str(exc) or "Unable to save Mistral defaults."
            self._set_message("Error", message, Gtk.MessageType.ERROR)
            return

        self._current_settings = dict(saved)
        self._set_message("Success", "Mistral defaults updated.", Gtk.MessageType.INFO)
        self.refresh_settings(clear_message=False)
