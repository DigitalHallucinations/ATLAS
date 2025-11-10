# UI/Chat/chat_page.py

from __future__ import annotations

"""
This module implements the ChatPage window, which displays the conversation
history, an input field, a microphone button for speech-to-text, and a send button.
It handles user input and displays both user messages and responses from the ATLAS language model.

It features robust error handling, nonblocking asynchronous processing via threads,
and schedules UI updates via GLib.idle_add.
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
from concurrent.futures import Future
from datetime import datetime
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import logging
from logging.handlers import RotatingFileHandler

from GTKUI.Utils.logging import GTKUILogHandler, read_recent_log_lines

# Configure logging for the chat page.
logger = logging.getLogger(__name__)


class _HandlerLevelFilter(logging.Filter):
    """Simple filter that enforces a minimum log level."""

    def __init__(self, level: int) -> None:
        super().__init__(name="chat-debug-level-filter")
        self._level = int(level)

    @property
    def level(self) -> int:
        return self._level

    def set_level(self, level: int) -> None:
        self._level = int(level)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - standard API
        """Return ``True`` when *record* meets the configured threshold."""

        return record.levelno >= self._level


class _DebugLogButtonUpdateFilter(logging.Filter):
    """Filter that schedules debug log button state refreshes."""

    def __init__(self, owner: "ChatPage") -> None:
        super().__init__(name="chat-debug-button-filter")
        self._owner = owner

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - standard API
        """Always allow the record and queue a button state update."""

        self._owner._queue_debug_log_button_update()
        return True


class ChatPage(Gtk.Box):
    """
    ChatPage window for user interaction.

    Displays the conversation history in a scrollable area, an input field,
    a microphone button to activate speech-to-text, and a send button.
    It updates the window title with the current persona's name and shows a status bar
    with LLM provider/model information as well as the active TTS provider and voice.
    """
    def __init__(self, atlas):
        """
        Initializes the ChatPage window.

        Args:
            atlas: The main ATLAS application instance.
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.ATLAS = atlas
        self._blackboard_subscription = None
        self._tool_activity_subscription = None
        self.connect("destroy", self._on_destroy)

        # Main vertical container.
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.vbox.set_hexpand(True)
        self.vbox.set_vexpand(True)
        self.append(self.vbox)

        # Notebook container for chat/terminal/debug tabs
        self.notebook = Gtk.Notebook()
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)

        # --- Chat tab ---
        self.chat_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.chat_tab_box.set_hexpand(True)
        self.chat_tab_box.set_vexpand(True)

        # Create a scrollable area for the conversation history.
        self.chat_history_scrolled = Gtk.ScrolledWindow()
        self.chat_history_scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.chat_history_scrolled.set_min_content_height(240)
        self.chat_history = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.chat_history.set_margin_start(10)
        self.chat_history.set_margin_end(10)
        self.chat_history.set_margin_top(6)
        self.chat_history.set_margin_bottom(6)
        self.chat_history_scrolled.set_child(self.chat_history)
        self.chat_history_scrolled.set_hexpand(True)
        self.chat_history_scrolled.set_vexpand(True)
        self.chat_tab_box.append(self.chat_history_scrolled)

        # --- Blackboard tab ---
        self.blackboard_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.blackboard_tab_box.set_margin_start(10)
        self.blackboard_tab_box.set_margin_end(10)
        self.blackboard_tab_box.set_margin_top(8)
        self.blackboard_tab_box.set_margin_bottom(8)
        self.blackboard_tab_box.set_hexpand(True)
        self.blackboard_tab_box.set_vexpand(True)

        self.blackboard_summary_label = Gtk.Label(xalign=0)
        self.blackboard_summary_label.set_wrap(True)
        self.blackboard_summary_label.add_css_class("body")
        self.blackboard_tab_box.append(self.blackboard_summary_label)

        self.blackboard_list = Gtk.ListBox()
        self.blackboard_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self.blackboard_list.add_css_class("boxed-list")
        self.blackboard_tab_box.append(self.blackboard_list)

        # Update window title with the current persona's name. This also
        # triggers a blackboard refresh, so ensure the widgets exist first.
        self.update_persona_label()

        # Create the input area with a multiline entry, a microphone button, and a send button.
        input_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        input_box.set_margin_top(8)
        input_box.set_margin_bottom(8)
        input_box.set_margin_start(10)
        input_box.set_margin_end(10)

        # Text input area.
        self.input_buffer = Gtk.TextBuffer()
        self.input_textview = Gtk.TextView.new_with_buffer(self.input_buffer)
        self.input_textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.input_textview.set_top_margin(6)
        self.input_textview.set_bottom_margin(6)
        self.input_textview.set_left_margin(6)
        self.input_textview.set_right_margin(6)
        self.input_textview.set_tooltip_text(
            "Enter to send • Shift+Enter for newline • Ctrl/⌘+L to refocus this field"
        )

        self.input_scroller = Gtk.ScrolledWindow()
        self.input_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.input_scroller.set_has_frame(False)
        self.input_scroller.set_hexpand(True)
        self.input_scroller.set_min_content_height(60)
        self.input_scroller.set_max_content_height(160)
        self.input_scroller.set_child(self.input_textview)
        input_box.append(self.input_scroller)

        # Keyboard controller for Enter/Shift+Enter handling within the text view.
        textview_keyctl = Gtk.EventControllerKey()
        textview_keyctl.connect("key-pressed", self.on_textview_key_pressed)
        self.input_textview.add_controller(textview_keyctl)

        # Keyboard controller for global shortcuts (e.g., Ctrl/⌘+L focus)
        keyctl = Gtk.EventControllerKey()
        keyctl.connect("key-pressed", self.on_key_pressed)
        self.add_controller(keyctl)

        # Microphone button for speech-to-text.
        self.mic_button = Gtk.Button()
        self._mic_icons = {
            "idle": self._make_icon("../../Icons/microphone.png", "audio-input-microphone"),
            "listening": self._make_icon("../../Icons/microphone-on.png", "media-record"),
        }
        self.mic_state_listening = False
        self.mic_button.set_child(self._mic_icons["idle"])
        self.mic_button.set_tooltip_text("Start listening (STT)")
        self.mic_button.get_style_context().add_class("mic-button")
        self.mic_button.connect("clicked", self.on_mic_button_click)
        input_box.append(self.mic_button)

        # Send button.
        self.send_button = Gtk.Button()
        self._set_button_icon(self.send_button, "../../Icons/send.png", "mail-send")
        self.send_button.set_tooltip_text("Send message")
        self.send_button.get_style_context().add_class("send-button")
        self.send_button.connect("clicked", self.on_send_message)
        input_box.append(self.send_button)

        self.chat_tab_box.append(input_box)

        chat_label = Gtk.Label(label="Chat")
        chat_label.add_css_class("caption")
        self.notebook.append_page(self.chat_tab_box, chat_label)

        blackboard_label = Gtk.Label(label="Blackboard")
        blackboard_label.add_css_class("caption")
        self.notebook.append_page(self.blackboard_tab_box, blackboard_label)

        # --- Terminal tab with persona context ---
        self.terminal_tab_container = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )
        self.terminal_tab_container.set_hexpand(True)
        self.terminal_tab_container.set_vexpand(True)
        self.terminal_tab_container.set_margin_top(6)
        self.terminal_tab_container.set_margin_bottom(6)
        self.terminal_tab_container.set_margin_start(6)
        self.terminal_tab_container.set_margin_end(6)

        terminal_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        terminal_header.set_hexpand(True)
        terminal_header.set_halign(Gtk.Align.FILL)

        terminal_header_spacer = Gtk.Box()
        terminal_header_spacer.set_hexpand(True)
        terminal_header.append(terminal_header_spacer)

        self._terminal_wrap_mode = Gtk.WrapMode.WORD_CHAR
        self._terminal_section_text_views: List[Gtk.TextView] = []

        self.terminal_wrap_toggle = Gtk.CheckButton(label="Wrap lines")
        self.terminal_wrap_toggle.set_tooltip_text(
            "Toggle line wrapping in terminal sections."
        )
        saved_wrap_state = self._load_terminal_wrap_preference()
        self.terminal_wrap_toggle.set_active(saved_wrap_state)
        self.terminal_wrap_toggle.connect("toggled", self._on_terminal_wrap_toggled)
        terminal_header.append(self.terminal_wrap_toggle)

        self.terminal_tab_container.append(terminal_header)

        self.terminal_scrolled = Gtk.ScrolledWindow()
        self.terminal_scrolled.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.terminal_scrolled.set_hexpand(True)
        self.terminal_scrolled.set_vexpand(True)
        self.terminal_tab_container.append(self.terminal_scrolled)

        self.terminal_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.terminal_tab_box.set_margin_top(6)
        self.terminal_tab_box.set_margin_bottom(6)
        self.terminal_tab_box.set_margin_start(6)
        self.terminal_tab_box.set_margin_end(6)
        self.terminal_tab_box.set_hexpand(True)
        self.terminal_tab_box.set_vexpand(True)
        self.terminal_scrolled.set_child(self.terminal_tab_box)
        self._terminal_sections: Dict[str, Gtk.TextBuffer] = {}
        self._terminal_section_buffers_by_widget: Dict[Gtk.Widget, Gtk.TextBuffer] = {}
        self._tool_log_entries: List[str] = []
        self._build_terminal_tab()
        terminal_label = Gtk.Label(label="Terminal")
        terminal_label.add_css_class("caption")
        self.notebook.append_page(self.terminal_tab_container, terminal_label)

        self.terminal_content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )
        self.terminal_content_box.set_margin_top(12)
        self.terminal_content_box.set_margin_bottom(12)
        self.terminal_content_box.set_margin_start(12)
        self.terminal_content_box.set_margin_end(12)
        self.terminal_content_box.set_hexpand(True)
        self.terminal_content_box.set_vexpand(False)
        self.terminal_tab_box.append(self.terminal_content_box)

        self.terminal_placeholder_label = Gtk.Label(
            label="No assistant thinking output available."
        )
        self.terminal_placeholder_label.set_halign(Gtk.Align.START)
        self.terminal_placeholder_label.set_valign(Gtk.Align.START)
        self.terminal_placeholder_label.add_css_class("dim-label")
        self.terminal_content_box.append(self.terminal_placeholder_label)

        self.thinking_expander = Gtk.Expander(label="Thinking")
        self.thinking_expander.set_hexpand(True)
        self.thinking_expander.set_visible(False)
        self.thinking_expander.set_expanded(False)

        self.thinking_buffer = Gtk.TextBuffer()
        self.thinking_text_view = Gtk.TextView.new_with_buffer(self.thinking_buffer)
        self.thinking_text_view.set_editable(False)
        self.thinking_text_view.set_cursor_visible(False)
        self.thinking_text_view.set_wrap_mode(self._get_terminal_wrap_mode())
        if hasattr(self.thinking_text_view, "set_monospace"):
            self.thinking_text_view.set_monospace(True)
        self.thinking_text_view.add_css_class("terminal-text")
        self.thinking_text_view.add_css_class("monospace")
        self._terminal_section_text_views.append(self.thinking_text_view)

        self.thinking_scrolled = Gtk.ScrolledWindow()
        self.thinking_scrolled.set_policy(
            Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC
        )
        self.thinking_scrolled.set_min_content_height(110)
        self.thinking_scrolled.set_child(self.thinking_text_view)

        self.thinking_expander.set_child(self.thinking_scrolled)
        self._install_terminal_copy_menu(
            self.thinking_expander, self.thinking_buffer, self.thinking_text_view
        )

        self.terminal_content_box.append(self.thinking_expander)

        # --- Debug tab with live logging ---
        self._debug_log_handler: Optional[GTKUILogHandler] = None
        self._debug_loggers_attached: List[logging.Logger] = []
        self._debug_handler_name = f"chat-debug-{id(self)}"
        self._debug_log_path: Optional[Path] = None
        self._debug_controls_updating = False
        self._debug_log_button_source_id: int = 0
        self._debug_log_button_filter: Optional[logging.Filter] = None
        self._debug_log_config = self._resolve_debug_log_config()
        self._debug_level_filter: Optional[_HandlerLevelFilter] = None
        self._build_debug_tab()

        self._blackboard_subscription = self.ATLAS.subscribe_event(
            "blackboard.events",
            self._handle_blackboard_event,
        )
        self._refresh_blackboard_tab()

        self.vbox.append(self.notebook)
        self._persona_change_handler = None
        self._refresh_terminal_tab()

        # Add a status area at the bottom of the window with a busy spinner and label.
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        status_box.set_margin_start(8)
        status_box.set_margin_end(8)
        status_box.set_margin_bottom(6)

        self.status_spinner = Gtk.Spinner()
        self.status_spinner.set_halign(Gtk.Align.START)
        self.status_spinner.set_valign(Gtk.Align.CENTER)
        self.status_spinner.set_visible(False)
        status_box.append(self.status_spinner)

        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        self._default_status_tooltip = "Active LLM provider/model and TTS status"
        self.status_label.set_tooltip_text(self._default_status_tooltip)
        status_box.append(self.status_label)

        self.vbox.append(status_box)
        self.update_status_bar()

        # Link provider changes to update the status bar and terminal tab.
        self._provider_change_handler = self._on_provider_changed
        self.ATLAS.add_provider_change_listener(self._provider_change_handler)
        self._active_user_listener = None
        self._current_user_display_name = self.ATLAS.get_user_display_name()
        self.update_persona_label()
        self._register_active_user_listener()
        self._register_persona_change_listener()
        self._register_tool_activity_listener()

        self.awaiting_response = False
        self._audio_output_dir = Path.home() / ".atlas" / "audio_responses"
        try:
            self._audio_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem issues are logged but non-fatal
            logger.warning("Unable to create audio output directory: %s", exc)
            self._audio_output_dir = Path.home()
        self._initialize_debug_logging()

    # --------------------------- Utilities ---------------------------

    def _make_icon(self, rel_path, fallback_icon_name):
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
            texture = Gdk.Texture.new_from_filename(icon_path)
            picture = Gtk.Picture.new_for_paintable(texture)
            picture.set_size_request(24, 24)
            picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            return picture
        except Exception as e:
            logger.debug(f"Icon load failed ({rel_path}): {e}")
            return Gtk.Image.new_from_icon_name(fallback_icon_name)

    def _set_button_icon(self, button: Gtk.Button, rel_path: str, fallback_icon_name: str):
        button.set_child(self._make_icon(rel_path, fallback_icon_name))

    # --------------------------- Terminal tab helpers ---------------------------

    def _build_terminal_tab(self) -> None:
        """Construct the collapsible sections within the terminal tab."""

        self._terminal_section_text_views.clear()

        sections = [
            ("System Prompt", "system_prompt", True),
            ("Persona Data", "persona_data", False),
            ("Conversation Messages", "conversation", True),
            ("Negotiation Trace", "negotiation", False),
            ("Metadata", "metadata", False),
            ("Declared Tools", "declared_tools", False),
            ("Tool Calls", "tool_calls", False),
            ("Tool Logs", "tool_logs", False),
        ]

        for title, key, expanded in sections:
            expander, buffer = self._create_terminal_section(title, expanded=expanded)
            self._terminal_sections[key] = buffer
            self.terminal_tab_box.append(expander)

    def _create_terminal_section(self, title: str, *, expanded: bool = False) -> Tuple['Gtk.Widget', Gtk.TextBuffer]:
        """Return an expander and backing text buffer for a terminal section."""

        buffer = Gtk.TextBuffer()
        text_view = Gtk.TextView.new_with_buffer(buffer)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        text_view.set_wrap_mode(self._get_terminal_wrap_mode())
        if hasattr(text_view, "set_monospace"):
            text_view.set_monospace(True)
        text_view.add_css_class("terminal-text")

        self._terminal_section_text_views.append(text_view)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_min_content_height(110)
        scrolled.set_child(text_view)

        expander_cls = getattr(Gtk, "Expander", None)
        if expander_cls is None:
            container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            header = Gtk.Label(label=title)
            header.add_css_class("caption")
            header.set_halign(Gtk.Align.START)
            container.append(header)
            container.append(scrolled)
            container.set_hexpand(True)
            container.set_margin_top(4)
            container.set_margin_bottom(4)
            container.set_margin_start(2)
            container.set_margin_end(2)
            self._install_terminal_copy_menu(container, buffer, text_view)
            return container, buffer

        expander = expander_cls(label=title)
        expander.set_child(scrolled)
        expander.set_hexpand(True)
        expander.set_margin_top(4)
        expander.set_margin_bottom(4)
        expander.set_margin_start(2)
        expander.set_margin_end(2)
        if hasattr(expander, "set_expanded"):
            expander.set_expanded(expanded)

        self._install_terminal_copy_menu(expander, buffer, text_view)
        return expander, buffer

    def _get_terminal_wrap_mode(self) -> Gtk.WrapMode:
        """Return the effective wrap mode for terminal text views."""

        if getattr(self, "terminal_wrap_toggle", None) and not self.terminal_wrap_toggle.get_active():
            return Gtk.WrapMode.NONE
        return self._terminal_wrap_mode

    def _load_terminal_wrap_preference(self) -> bool:
        """Read the persisted terminal wrap preference, defaulting to wrapping."""

        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        if ui_config is not None:
            getter = getattr(ui_config, "get_terminal_wrap_enabled", None)
            if callable(getter):
                try:
                    return bool(getter())
                except Exception:  # pragma: no cover - defensive fallback
                    logger.debug("Failed to load terminal wrap preference", exc_info=True)
                    return True

        if manager is None:
            return True

        try:
            value = manager.get_config("UI_TERMINAL_WRAP_ENABLED", True)
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("Error reading terminal wrap config", exc_info=True)
            return True

        if isinstance(value, bool):
            return value

        return True

    def _on_terminal_wrap_toggled(self, _button: Gtk.CheckButton) -> None:
        """Update terminal text views when the wrap toggle changes."""

        wrap_enabled = self.terminal_wrap_toggle.get_active()
        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        if ui_config is not None:
            setter = getattr(ui_config, "set_terminal_wrap_enabled", None)
            if callable(setter):
                try:
                    setter(wrap_enabled)
                except Exception:  # pragma: no cover - defensive fallback
                    logger.debug("Failed to persist terminal wrap preference", exc_info=True)
        elif manager is not None:
            setter = getattr(manager, "set_ui_terminal_wrap_enabled", None)
            if callable(setter):
                try:
                    setter(wrap_enabled)
                except Exception:  # pragma: no cover - defensive fallback
                    logger.debug("Failed to persist terminal wrap preference", exc_info=True)

        wrap_mode = self._get_terminal_wrap_mode()
        for text_view in self._terminal_section_text_views:
            text_view.set_wrap_mode(wrap_mode)

    def _install_terminal_copy_menu(
        self, section_widget: "Gtk.Widget", buffer: Gtk.TextBuffer, text_view: Gtk.TextView
    ) -> None:
        """Attach a context menu that copies the section buffer to the clipboard."""

        if not isinstance(getattr(self, "_terminal_section_buffers_by_widget", None), dict):
            self._terminal_section_buffers_by_widget = {}

        self._terminal_section_buffers_by_widget[section_widget] = buffer

        popover = Gtk.Popover()
        popover.set_has_arrow(True)
        popover.set_parent(text_view)

        menu_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        popover.set_child(menu_box)

        copy_button = Gtk.Button(label="Copy Text")
        copy_button.add_css_class("flat")
        copy_button.set_halign(Gtk.Align.FILL)
        copy_button.set_hexpand(True)
        copy_button.set_tooltip_text(
            "Copy selected text, or the entire section if nothing is selected."
        )
        menu_box.append(copy_button)

        def _copy_buffer_to_clipboard(*_args: object) -> None:
            target_buffer = self._terminal_section_buffers_by_widget.get(section_widget, buffer)
            selection_bounds = target_buffer.get_selection_bounds()
            if selection_bounds and selection_bounds[0]:
                _has_selection, start_iter, end_iter = selection_bounds
            else:
                start_iter = target_buffer.get_start_iter()
                end_iter = target_buffer.get_end_iter()
            text = target_buffer.get_text(start_iter, end_iter, True)
            display = text_view.get_display()
            if display is None:
                return
            clipboard = display.get_clipboard()
            if clipboard is None:
                return
            clipboard.set_text(text or "")
            popover.popdown()

        copy_button.connect("clicked", _copy_buffer_to_clipboard)

        click_controller = Gtk.GestureClick()
        click_controller.set_button(Gdk.BUTTON_SECONDARY)

        def _show_popover(gesture: Gtk.GestureClick, _n_press: int, x: float, y: float) -> None:
            if gesture.get_current_button() != Gdk.BUTTON_SECONDARY:
                return
            popover.set_pointing_to(Gdk.Rectangle(int(x), int(y), 1, 1))
            popover.popup()

        click_controller.connect("pressed", _show_popover)
        text_view.add_controller(click_controller)

    def _set_terminal_section_text(self, key: str, text: str) -> None:
        sections = getattr(self, "_terminal_sections", None)
        if not isinstance(sections, dict):
            return
        buffer = sections.get(key)
        if buffer is None:
            return
        buffer.set_text(text or "")

    def _refresh_terminal_tab(self, context: Optional[Dict[str, object]] = None) -> None:
        """Update terminal tab widgets with the latest persona/chat context."""

        context_data: Dict[str, object] = {}
        if isinstance(context, dict):
            context_data = context
        else:
            getter = getattr(self.ATLAS, "get_current_persona_context", None)
            if callable(getter):
                try:
                    fetched = getter() or {}
                except Exception as exc:
                    logger.error("Failed to fetch persona context: %s", exc, exc_info=True)
                else:
                    if isinstance(fetched, dict):
                        context_data = fetched

        prompt_text = str(context_data.get("system_prompt") or "No system prompt available.")
        self._set_terminal_section_text("system_prompt", prompt_text)

        substitutions = context_data.get("substitutions")
        if isinstance(substitutions, dict) and substitutions:
            persona_data_text = json.dumps(substitutions, ensure_ascii=False, indent=2)
        else:
            persona_data_text = "No persona-specific data available."
        self._set_terminal_section_text("persona_data", persona_data_text)

        history: List[Dict[str, object]] = []
        history_getter = getattr(self.ATLAS, "get_chat_history_snapshot", None)
        if callable(history_getter):
            try:
                candidate = history_getter()
            except Exception as exc:
                logger.error("Failed to obtain chat history snapshot: %s", exc, exc_info=True)
            else:
                if isinstance(candidate, list):
                    history = candidate

        conversation_text = self._format_conversation_history(history)
        self._set_terminal_section_text("conversation", conversation_text)

        negotiation_history: List[Dict[str, object]] = []
        negotiation_getter = getattr(self.ATLAS, "get_negotiation_log", None)
        if callable(negotiation_getter):
            try:
                candidate = negotiation_getter() or []
            except Exception as exc:
                logger.error("Failed to obtain negotiation history: %s", exc, exc_info=True)
            else:
                if isinstance(candidate, list):
                    negotiation_history = candidate

        negotiation_text = self._format_negotiation_history(negotiation_history)
        self._set_terminal_section_text("negotiation", negotiation_text)

        metadata_lines = []
        persona_name = context_data.get("persona_name") if isinstance(context_data, dict) else None
        if not persona_name:
            persona_name_getter = getattr(self.ATLAS, "get_active_persona_name", None)
            if callable(persona_name_getter):
                try:
                    persona_name = persona_name_getter()
                except Exception as exc:
                    logger.error("Failed to resolve active persona name: %s", exc, exc_info=True)

        metadata_lines.append(f"Persona: {persona_name or 'Unknown'}")
        metadata_lines.append(f"Messages recorded: {len(history)}")

        conversation_id = None
        session = getattr(self.ATLAS, "chat_session", None)
        if session is not None:
            get_conv_id = getattr(session, "get_conversation_id", None)
            if callable(get_conv_id):
                try:
                    conversation_id = get_conv_id()
                except Exception as exc:
                    logger.error("Failed to obtain conversation ID: %s", exc, exc_info=True)
            else:
                conversation_id = getattr(session, "conversation_id", None)
        metadata_lines.append(f"Conversation ID: {conversation_id or 'Unavailable'}")

        summary = {}
        status_getter = getattr(self.ATLAS, "get_chat_status_summary", None)
        if callable(status_getter):
            try:
                candidate = status_getter() or {}
            except Exception as exc:
                logger.error("Failed to fetch chat status summary: %s", exc, exc_info=True)
            else:
                if isinstance(candidate, dict):
                    summary = candidate

        if summary:
            metadata_lines.append("Status Summary:")
            for key, value in summary.items():
                metadata_lines.append(f"  {key}: {value}")

        metadata_lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        metadata_text = "\n".join(metadata_lines)
        self._set_terminal_section_text("metadata", metadata_text)

        self._refresh_tool_sections()

    def _on_provider_changed(self, status_summary=None) -> None:
        """Refresh UI elements when the active provider or model changes."""

        self.update_status_bar(status_summary)
        self._refresh_terminal_tab()

    def _refresh_tool_sections(self, history: Optional[List[Dict[str, object]]] = None) -> None:
        """Update the tool declaration and activity sections."""

        tools_snapshot: Dict[str, object] = {}
        tools_getter = getattr(self.ATLAS, "get_current_persona_tools", None)
        if callable(tools_getter):
            try:
                candidate = tools_getter() or {}
            except Exception as exc:
                logger.error("Failed to obtain persona tools: %s", exc, exc_info=True)
            else:
                if isinstance(candidate, dict):
                    tools_snapshot = candidate

        declared_tools_text = self._format_declared_tools(tools_snapshot)
        self._set_terminal_section_text("declared_tools", declared_tools_text)

        tool_history: List[Dict[str, object]] = []
        if isinstance(history, list):
            tool_history = history
        else:
            history_getter = getattr(self.ATLAS, "get_tool_activity_log", None)
            if callable(history_getter):
                try:
                    candidate = history_getter()
                except Exception as exc:
                    logger.error("Failed to obtain tool activity log: %s", exc, exc_info=True)
                else:
                    if isinstance(candidate, list):
                        tool_history = candidate

        tool_calls_text = self._format_tool_calls(tool_history)
        self._set_terminal_section_text("tool_calls", tool_calls_text)

        log_blocks = self._format_tool_log_entries(tool_history)
        self._tool_log_entries = log_blocks
        if log_blocks:
            tool_logs_text = "\n\n".join(log_blocks)
        else:
            tool_logs_text = "No tool logs available."
        self._set_terminal_section_text("tool_logs", tool_logs_text)

    # --------------------------- Blackboard helpers ---------------------------

    def _handle_blackboard_event(self, payload, _message=None) -> None:
        if not isinstance(payload, Mapping):
            return
        entry = payload.get("entry") if isinstance(payload, Mapping) else None
        if not isinstance(entry, Mapping):
            return
        active_scope = self._get_active_conversation_id()
        if active_scope and entry.get("scope_id") == active_scope:
            GLib.idle_add(self._refresh_blackboard_tab)

    def _refresh_blackboard_tab(self) -> None:
        summary_label = getattr(self, "blackboard_summary_label", None)
        list_widget = getattr(self, "blackboard_list", None)
        if summary_label is None or list_widget is None:
            return

        scope_id = self._get_active_conversation_id()
        if not scope_id:
            summary_label.set_text("No active conversation.")
            self._clear_blackboard_list(list_widget)
            placeholder = Gtk.Label(xalign=0)
            placeholder.set_wrap(True)
            placeholder.set_text(
                "Blackboard entries will appear once a conversation is active."
            )
            list_widget.append(placeholder)
            return

        try:
            summary = self.ATLAS.get_blackboard_summary(scope_id)
        except Exception as exc:
            logger.error("Failed to obtain blackboard summary: %s", exc, exc_info=True)
            summary = {}
        counts = summary.get("counts") if isinstance(summary, Mapping) else {}
        hypothesis_count = counts.get("hypothesis", 0) if isinstance(counts, Mapping) else 0
        claim_count = counts.get("claim", 0) if isinstance(counts, Mapping) else 0
        artifact_count = counts.get("artifact", 0) if isinstance(counts, Mapping) else 0
        summary_label.set_text(
            f"Hypotheses: {hypothesis_count} • Claims: {claim_count} • Artifacts: {artifact_count}"
        )

        entries = summary.get("entries") if isinstance(summary, Mapping) else []
        if not isinstance(entries, list):
            entries = []

        self._clear_blackboard_list(list_widget)
        if not entries:
            empty_label = Gtk.Label(xalign=0)
            empty_label.set_wrap(True)
            empty_label.set_text("No shared posts yet.")
            list_widget.append(empty_label)
            return

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            row.set_margin_bottom(6)
            heading = Gtk.Label(xalign=0)
            heading.set_wrap(True)
            heading.add_css_class("heading")
            heading.set_text(
                f"[{str(entry.get('category') or '').title()}] {entry.get('title') or ''}"
            )
            row.append(heading)

            content_label = Gtk.Label(xalign=0)
            content_label.set_wrap(True)
            content_label.set_text(str(entry.get("content") or ""))
            row.append(content_label)

            meta_bits: list[str] = []
            author = entry.get("author")
            if author:
                meta_bits.append(f"Author: {author}")
            created_at = entry.get("created_at")
            if isinstance(created_at, (int, float)):
                try:
                    created_dt = datetime.fromtimestamp(created_at)
                except Exception:
                    created_dt = None
                if created_dt is not None:
                    meta_bits.append(created_dt.strftime("%Y-%m-%d %H:%M"))
            tags = entry.get("tags")
            if isinstance(tags, list) and tags:
                meta_bits.append("Tags: " + ", ".join(str(tag) for tag in tags))

            meta_label = Gtk.Label(xalign=0)
            meta_label.add_css_class("caption")
            meta_label.set_wrap(True)
            if meta_bits:
                meta_label.set_text(" • ".join(meta_bits))
            else:
                meta_label.set_text("Shared item")
            row.append(meta_label)

            list_widget.append(row)

    def _clear_blackboard_list(self, list_widget: Optional[Gtk.ListBox] = None) -> None:
        if list_widget is None:
            list_widget = getattr(self, "blackboard_list", None)
        if list_widget is None:
            return

        child = list_widget.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            list_widget.remove(child)
            child = next_child

    def _get_active_conversation_id(self) -> Optional[str]:
        session = getattr(self.ATLAS, "chat_session", None)
        conversation_id: Optional[str] = None
        if session is not None:
            getter = getattr(session, "get_conversation_id", None)
            if callable(getter):
                try:
                    conversation_id = getter()
                except Exception as exc:
                    logger.error("Failed to obtain conversation ID: %s", exc, exc_info=True)
            elif hasattr(session, "conversation_id"):
                conversation_id = getattr(session, "conversation_id")

        if not conversation_id:
            return None
        text = str(conversation_id).strip()
        return text or None

    def _on_destroy(self, *_args) -> None:
        subscription = self._blackboard_subscription
        if subscription is not None:
            self.ATLAS.unsubscribe_event(subscription)
            self._blackboard_subscription = None

    def _format_declared_tools(self, snapshot: Dict[str, object]) -> str:
        if not snapshot:
            return "No tools declared for this persona."

        data = {}
        function_map = snapshot.get("function_map") if isinstance(snapshot, dict) else None
        if isinstance(function_map, dict) and function_map:
            data["python_functions"] = function_map

        payload = snapshot.get("function_payloads") if isinstance(snapshot, dict) else None
        if payload is not None:
            data["function_payloads"] = payload

        if not data:
            return "No tools declared for this persona."

        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(data)

    def _format_tool_calls(self, entries: List[Dict[str, object]]) -> str:
        if not entries:
            return "No tool activity recorded."

        lines: List[str] = []
        for entry in reversed(entries):
            metrics = entry.get("metrics") or {}
            payload = entry.get("payload") if entry.get("payload_included") else None
            payload_preview = entry.get("payload_preview") or {}

            def _payload_value(key: str):
                if isinstance(payload, dict) and key in payload:
                    return payload.get(key)
                return payload_preview.get(key, entry.get(key))

            name = str(entry.get("tool_name", "unknown"))
            timestamp = (
                metrics.get("completed_at")
                or metrics.get("started_at")
                or entry.get("completed_at")
                or entry.get("started_at")
                or "Unknown time"
            )
            status_value = metrics.get("status") or entry.get("status", "unknown")
            status = str(status_value).upper()
            duration = (
                metrics.get("latency_ms")
                if metrics.get("latency_ms") is not None
                else entry.get("duration_ms")
            )
            if isinstance(duration, (int, float)):
                duration_text = f"{duration:.0f} ms"
            else:
                duration_text = ""

            args_source = entry.get("arguments_text") or _payload_value("arguments")
            args_text = self._stringify_tool_section_value(args_source)
            if args_text:
                args_text = " ".join(args_text.split())
            if len(args_text) > 120:
                args_text = args_text[:117] + "..."

            line = f"[{timestamp}] {name}({args_text}) → {status}"
            if duration_text:
                line += f" • {duration_text}"
            lines.append(line)

        return "\n".join(lines)

    def _format_tool_log_entries(self, entries: List[Dict[str, object]]) -> List[str]:
        formatted: List[str] = []
        for entry in reversed(entries):
            metrics = entry.get("metrics") or {}
            payload = entry.get("payload") if entry.get("payload_included") else None
            payload_preview = entry.get("payload_preview") or {}

            def _payload_value(key: str):
                if isinstance(payload, dict) and key in payload:
                    return payload.get(key)
                return payload_preview.get(key, entry.get(key))

            name = str(entry.get("tool_name", "unknown"))
            status_value = metrics.get("status") or entry.get("status", "unknown")
            status = str(status_value).upper()
            started = (
                metrics.get("started_at")
                or entry.get("started_at")
                or "Unknown start"
            )
            completed = (
                metrics.get("completed_at")
                or entry.get("completed_at")
                or "Unknown end"
            )
            duration = (
                metrics.get("latency_ms")
                if metrics.get("latency_ms") is not None
                else entry.get("duration_ms")
            )
            duration_line = None
            if isinstance(duration, (int, float)):
                duration_line = f"Duration: {duration:.0f} ms"

            args_source = entry.get("arguments_text") or _payload_value("arguments")
            args_text = self._stringify_tool_section_value(args_source)
            result_source = entry.get("result_text") or _payload_value("result")
            result_text = self._stringify_tool_section_value(result_source)
            error_text = entry.get("error")
            stdout_text = self._stringify_tool_section_value(_payload_value("stdout")).strip()
            stderr_text = self._stringify_tool_section_value(_payload_value("stderr")).strip()

            block_lines = [f"{name} • {status}", f"Started: {started}", f"Completed: {completed}"]
            if duration_line:
                block_lines.append(duration_line)
            if args_text:
                block_lines.append(f"Args: {args_text}")
            if result_text:
                block_lines.append(f"Result: {result_text}")
            if stdout_text:
                block_lines.append("stdout:\n" + self._indent_multiline(stdout_text))
            if stderr_text:
                block_lines.append("stderr:\n" + self._indent_multiline(stderr_text))
            if error_text and status != "SUCCESS":
                block_lines.append(f"Error: {error_text}")

            formatted.append("\n".join(block_lines))

        return formatted

    def _stringify_tool_section_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def _indent_multiline(self, text: str, prefix: str = "    ") -> str:
        if not text:
            return ""
        return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())

    # --------------------------- Debug tab helpers ---------------------------

    def _resolve_debug_log_config(self) -> Dict[str, object]:
        defaults: Dict[str, object] = {
            "level": logging.INFO,
            "max_lines": 2000,
            "initial_lines": 400,
            "logger_names": [],
            "format": None,
        }

        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        if ui_config is None and manager is None:
            return defaults

        def _extract_int(value, fallback):
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value.strip())
                except ValueError:
                    return fallback
            return fallback

        if ui_config is not None:
            try:
                level_value = ui_config.get_debug_log_level()
            except Exception:
                level_value = None
        else:
            try:
                level_value = manager.get_config("UI_DEBUG_LOG_LEVEL")
            except Exception:
                level_value = None
        if isinstance(level_value, str):
            candidate = getattr(logging, level_value.upper(), None)
            if isinstance(candidate, int):
                defaults["level"] = candidate
        elif isinstance(level_value, int):
            defaults["level"] = level_value

        if ui_config is not None:
            try:
                configured_max = ui_config.get_debug_log_max_lines(defaults["max_lines"])
            except Exception:
                configured_max = None
            if isinstance(configured_max, int):
                defaults["max_lines"] = configured_max
            try:
                configured_initial = ui_config.get_debug_log_initial_lines(defaults["initial_lines"])
            except Exception:
                configured_initial = None
            if isinstance(configured_initial, int):
                defaults["initial_lines"] = configured_initial
            try:
                defaults["logger_names"] = ui_config.get_debug_logger_names()
            except Exception:
                defaults["logger_names"] = []
            try:
                format_value = ui_config.get_debug_log_format()
            except Exception:
                format_value = None
        else:
            try:
                max_lines_value = manager.get_config("UI_DEBUG_LOG_MAX_LINES")
            except Exception:
                max_lines_value = None
            defaults["max_lines"] = max(100, _extract_int(max_lines_value, defaults["max_lines"]))

            try:
                initial_lines_value = manager.get_config("UI_DEBUG_LOG_INITIAL_LINES")
            except Exception:
                initial_lines_value = None
            defaults["initial_lines"] = max(0, _extract_int(initial_lines_value, defaults["initial_lines"]))

            try:
                logger_names_value = manager.get_config("UI_DEBUG_LOGGERS")
            except Exception:
                logger_names_value = None
            logger_names: List[str] = []
            if isinstance(logger_names_value, str):
                for part in logger_names_value.split(","):
                    sanitized = part.strip()
                    if sanitized:
                        logger_names.append(sanitized)
            elif isinstance(logger_names_value, (list, tuple, set)):
                for entry in logger_names_value:
                    sanitized = str(entry).strip()
                    if sanitized:
                        logger_names.append(sanitized)
            defaults["logger_names"] = logger_names

            try:
                format_value = manager.get_config("UI_DEBUG_LOG_FORMAT")
            except Exception:
                format_value = None
        if isinstance(format_value, str) and format_value.strip():
            defaults["format"] = format_value.strip()

        return defaults

    def _build_debug_tab(self) -> None:
        self.debug_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.debug_tab_box.set_hexpand(True)
        self.debug_tab_box.set_vexpand(True)
        self.debug_tab_box.set_margin_top(6)
        self.debug_tab_box.set_margin_bottom(6)
        self.debug_tab_box.set_margin_start(6)
        self.debug_tab_box.set_margin_end(6)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.set_margin_bottom(4)
        self.debug_tab_box.append(controls)

        self.debug_clear_btn = Gtk.Button(label="Clear")
        self.debug_clear_btn.add_css_class("flat")
        self.debug_clear_btn.set_tooltip_text("Clear debug log output")
        self.debug_clear_btn.connect("clicked", self._on_debug_clear_clicked)
        controls.append(self.debug_clear_btn)

        self.debug_pause_btn = Gtk.ToggleButton(label="Pause")
        self.debug_pause_btn.add_css_class("flat")
        self.debug_pause_btn.set_tooltip_text("Temporarily pause live log updates")
        self.debug_pause_btn.connect("toggled", self._on_debug_pause_toggled)
        controls.append(self.debug_pause_btn)

        self.debug_open_log_btn = Gtk.Button(label="Open Log File…")
        self.debug_open_log_btn.add_css_class("flat")
        self.debug_open_log_btn.set_tooltip_text("Open the active debug log file in an external viewer")
        self.debug_open_log_btn.set_sensitive(False)
        self.debug_open_log_btn.connect("clicked", self._on_debug_open_log_clicked)
        controls.append(self.debug_open_log_btn)

        controls.append(self._spacer())

        level_label = Gtk.Label(label="Level:")
        level_label.add_css_class("caption")
        level_label.set_halign(Gtk.Align.END)
        controls.append(level_label)

        self._debug_level_options = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        self.debug_level_combo = Gtk.ComboBoxText()
        for option in self._debug_level_options:
            self.debug_level_combo.append_text(option)
        self.debug_level_combo.set_tooltip_text("Adjust the minimum log level shown")
        self.debug_level_combo.connect("changed", self._on_debug_level_changed)
        controls.append(self.debug_level_combo)

        logger_label = Gtk.Label(label="Loggers:")
        logger_label.add_css_class("caption")
        logger_label.set_halign(Gtk.Align.END)
        controls.append(logger_label)

        self.debug_logger_entry = Gtk.Entry()
        self.debug_logger_entry.set_hexpand(True)
        logger_names_text = ", ".join(self._debug_log_config.get("logger_names") or [])
        self.debug_logger_entry.set_text(logger_names_text)
        self.debug_logger_entry.set_tooltip_text("Comma-separated logger names to mirror in the UI debug console")
        self.debug_logger_entry.connect("activate", self._on_debug_logger_names_committed)
        focus_controller = Gtk.EventControllerFocus()
        focus_controller.connect("leave", self._on_debug_logger_entry_focus_leave)
        self.debug_logger_entry.add_controller(focus_controller)
        self._debug_logger_focus_controller = focus_controller
        controls.append(self.debug_logger_entry)

        retention_label = Gtk.Label(label="Max lines:")
        retention_label.add_css_class("caption")
        retention_label.set_halign(Gtk.Align.END)
        controls.append(retention_label)

        max_lines = int(self._debug_log_config.get("max_lines", 2000))
        adjustment = Gtk.Adjustment(
            value=float(max_lines),
            lower=100.0,
            upper=50000.0,
            step_increment=100.0,
            page_increment=500.0,
        )
        self.debug_retention_spin = Gtk.SpinButton()
        self.debug_retention_spin.set_adjustment(adjustment)
        self.debug_retention_spin.set_digits(0)
        self.debug_retention_spin.set_width_chars(5)
        self.debug_retention_spin.set_tooltip_text("Maximum number of log lines retained in the buffer")
        self.debug_retention_spin.connect("value-changed", self._on_debug_retention_changed)
        controls.append(self.debug_retention_spin)

        self.debug_log_buffer = Gtk.TextBuffer()
        self.debug_log_view = Gtk.TextView.new_with_buffer(self.debug_log_buffer)
        self.debug_log_view.set_editable(False)
        self.debug_log_view.set_cursor_visible(False)
        self.debug_log_view.set_wrap_mode(self._get_terminal_wrap_mode())
        if hasattr(self.debug_log_view, "set_monospace"):
            self.debug_log_view.set_monospace(True)
        self.debug_log_view.add_css_class("monospace")
        self.debug_log_view.set_hexpand(True)
        self.debug_log_view.set_vexpand(True)
        self._terminal_section_text_views.append(self.debug_log_view)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_child(self.debug_log_view)
        self._install_terminal_copy_menu(self.debug_log_view, self.debug_log_buffer, self.debug_log_view)
        self.debug_tab_box.append(scrolled)

        debug_label = Gtk.Label(label="Debug")
        debug_label.add_css_class("caption")
        self.notebook.append_page(self.debug_tab_box, debug_label)

    def _initialize_debug_logging(self) -> None:
        self._load_initial_debug_logs()
        self._attach_debug_log_handler()
        self._update_debug_controls()

    def _load_initial_debug_logs(self) -> None:
        buffer = getattr(self, "debug_log_buffer", None)
        if buffer is None:
            return

        path = self._resolve_active_log_path()
        if path is not None:
            self._debug_log_path = path
        limit = int(self._debug_log_config.get("initial_lines") or 0)
        if limit <= 0:
            limit = int(self._debug_log_config.get("max_lines", 2000))

        if path is None or not path.exists():
            buffer.set_text("")
        else:
            buffer.set_text(read_recent_log_lines(path, limit) or "")
        self._scroll_debug_log_to_end()
        self._ensure_debug_log_button_enabled()

    def _resolve_active_log_path(self) -> Optional[Path]:
        atlas_logger = getattr(self.ATLAS, "logger", None)
        if isinstance(atlas_logger, logging.Logger):
            for handler in getattr(atlas_logger, "handlers", []):
                if isinstance(handler, RotatingFileHandler):
                    base = getattr(handler, "baseFilename", None)
                    if base:
                        return Path(base)
            for handler in getattr(atlas_logger, "handlers", []):
                base = getattr(handler, "baseFilename", None)
                if base:
                    return Path(base)

        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        if ui_config is not None:
            try:
                configured = ui_config.get_debug_log_file_name()
            except Exception:
                configured = None
            try:
                app_root = ui_config.get_app_root()
            except Exception:
                app_root = None
        elif manager is not None:
            try:
                configured = manager.get_config("UI_DEBUG_LOG_FILE")
            except Exception:
                configured = None

            try:
                app_root = manager.get_config("APP_ROOT")
            except Exception:
                app_root = None
        else:
            configured = None
            app_root = None

        if configured:
            base_path = Path(app_root or Path.cwd()) / "logs" / str(configured)
            return base_path

        if app_root:
            candidate = Path(app_root) / "logs" / "CSSLM.log"
            return candidate

        return None

    def _resolve_log_formatter(self) -> logging.Formatter:
        atlas_logger = getattr(self.ATLAS, "logger", None)
        if isinstance(atlas_logger, logging.Logger):
            for handler in getattr(atlas_logger, "handlers", []):
                formatter = getattr(handler, "formatter", None)
                if formatter is not None:
                    return formatter

        fmt = self._debug_log_config.get("format") or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        return logging.Formatter(fmt)

    def _attach_debug_log_handler(self) -> None:
        buffer = getattr(self, "debug_log_buffer", None)
        if buffer is None:
            self._ensure_debug_log_button_enabled()
            return

        self._detach_debug_log_handler()

        handler = GTKUILogHandler(
            buffer,
            text_view=getattr(self, "debug_log_view", None),
            max_lines=int(self._debug_log_config.get("max_lines", 2000)),
        )
        handler.set_name(self._debug_handler_name)
        configured_level = self._debug_log_config.get("level", logging.INFO)
        try:
            level_value = int(configured_level)
        except (TypeError, ValueError):
            try:
                level_value = int(getattr(logging, str(configured_level), logging.INFO))
            except (TypeError, ValueError):
                level_value = logging.INFO
        handler.setLevel(level_value)
        self._debug_level_filter = _HandlerLevelFilter(level_value)
        handler.addFilter(self._debug_level_filter)
        handler.setFormatter(self._resolve_log_formatter())

        if self._debug_log_button_filter is None:
            self._debug_log_button_filter = _DebugLogButtonUpdateFilter(self)
        handler.addFilter(self._debug_log_button_filter)

        logger_candidates: List[logging.Logger] = []
        atlas_logger = getattr(self.ATLAS, "logger", None)
        if isinstance(atlas_logger, logging.Logger):
            logger_candidates.append(atlas_logger)

        extra_logger_names = self._debug_log_config.get("logger_names") or []
        for name in extra_logger_names:
            try:
                candidate = logging.getLogger(name)
            except Exception:
                continue
            if isinstance(candidate, logging.Logger):
                logger_candidates.append(candidate)

        attached: List[logging.Logger] = []
        seen_ids = set()
        for logger_obj in logger_candidates:
            if id(logger_obj) in seen_ids:
                continue
            seen_ids.add(id(logger_obj))
            duplicate = False
            for existing in getattr(logger_obj, "handlers", []):
                if existing.get_name() == handler.get_name():
                    duplicate = True
                    break
            if not duplicate:
                logger_obj.addHandler(handler)
            attached.append(logger_obj)

        if attached:
            self._debug_log_handler = handler
            self._debug_loggers_attached = attached
        else:
            handler.close()
            self._debug_level_filter = None

        self._ensure_debug_log_button_enabled()

    def _detach_debug_log_handler(self) -> None:
        handler = getattr(self, "_debug_log_handler", None)
        if handler is None:
            self._cancel_debug_log_button_update()
            return

        attached = getattr(self, "_debug_loggers_attached", [])
        if not isinstance(attached, list):
            attached = []
        self._debug_loggers_attached = attached

        for logger_obj in list(self._debug_loggers_attached):
            try:
                logger_obj.removeHandler(handler)
            except ValueError:
                pass
        button_filter = getattr(self, "_debug_log_button_filter", None)
        if button_filter is not None and hasattr(handler, "removeFilter"):
            try:
                handler.removeFilter(button_filter)
            except ValueError:
                pass
        level_filter = getattr(self, "_debug_level_filter", None)
        if level_filter is not None and hasattr(handler, "removeFilter"):
            try:
                handler.removeFilter(level_filter)
            except ValueError:
                pass
        if hasattr(handler, "close"):
            handler.close()
        self._debug_loggers_attached = []
        self._debug_log_handler = None
        self._debug_level_filter = None
        self._cancel_debug_log_button_update()

    def _update_debug_controls(self) -> None:
        if not hasattr(self, "debug_level_combo"):
            return

        self._debug_controls_updating = True
        try:
            handler = self._debug_log_handler
            current_level = int(self._debug_log_config.get("level", logging.INFO))
            if handler is not None:
                current_level = handler.level
            level_name = logging.getLevelName(current_level)
            if isinstance(level_name, int):
                level_name = logging.getLevelName(logging.INFO)
            level_name = str(level_name)
            try:
                index = self._debug_level_options.index(level_name)
            except ValueError:
                index = self._debug_level_options.index("INFO")
            self.debug_level_combo.set_active(index)

            paused = handler.paused if handler is not None else False
            self.debug_pause_btn.set_active(paused)
            self.debug_pause_btn.set_label("Resume" if paused else "Pause")

            max_lines = int(self._debug_log_config.get("max_lines", 2000))
            if handler is not None:
                max_lines = handler.max_lines
            self.debug_retention_spin.set_value(float(max_lines))

            entry = getattr(self, "debug_logger_entry", None)
            if isinstance(entry, Gtk.Entry):
                names_value = self._debug_log_config.get("logger_names") or []
                entry.set_text(", ".join(str(name) for name in names_value))
        finally:
            self._debug_controls_updating = False

    def _queue_debug_log_button_update(self) -> None:
        source_id = getattr(self, "_debug_log_button_source_id", 0)
        if source_id:
            return

        def _on_idle() -> bool:
            self._debug_log_button_source_id = 0
            self._ensure_debug_log_button_enabled()
            return False

        try:
            source_handle = GLib.idle_add(_on_idle)
        except Exception:
            source_handle = 0
        if source_handle:
            self._debug_log_button_source_id = int(source_handle)
        else:
            _on_idle()

    def _cancel_debug_log_button_update(self) -> None:
        source_id = getattr(self, "_debug_log_button_source_id", 0)
        if not source_id:
            return
        try:
            GLib.source_remove(source_id)
        except Exception:
            pass
        self._debug_log_button_source_id = 0

    def _ensure_debug_log_button_enabled(self) -> None:
        path_value = getattr(self, "_debug_log_path", None)
        try:
            path = Path(path_value) if path_value is not None else None
        except (TypeError, ValueError):
            path = None

        if path is None or not path.exists():
            candidate = self._resolve_active_log_path()
            if candidate is not None and candidate.exists():
                self._debug_log_path = candidate
                path = candidate

        self._update_debug_log_button()

    def _update_debug_log_button(self) -> None:
        button = getattr(self, "debug_open_log_btn", None)
        if not isinstance(button, Gtk.Button):
            return

        path_value = getattr(self, "_debug_log_path", None)
        try:
            path = Path(path_value) if path_value is not None else None
        except (TypeError, ValueError):
            path = None

        button.set_sensitive(bool(path and path.exists()))

    def _scroll_debug_log_to_end(self) -> None:
        view = getattr(self, "debug_log_view", None)
        buffer = getattr(self, "debug_log_buffer", None)
        if view is None or buffer is None:
            return
        end_iter = buffer.get_end_iter()
        view.scroll_to_iter(end_iter, 0.0, False, 0.0, 1.0)

    def _on_debug_clear_clicked(self, *_args):
        if self._debug_log_handler is not None:
            self._debug_log_handler.clear()
        elif hasattr(self, "debug_log_buffer"):
            self.debug_log_buffer.set_text("")

    def _on_debug_pause_toggled(self, button: Gtk.ToggleButton):
        handler = getattr(self, "_debug_log_handler", None)
        target_btn = getattr(self, "debug_pause_btn", None)

        if handler is None:
            if button.get_active():
                button.set_active(False)
            if isinstance(target_btn, Gtk.ToggleButton):
                target_btn.set_label("Pause")
            return

        paused = button.get_active()
        handler.set_paused(paused)
        if isinstance(target_btn, Gtk.ToggleButton):
            target_btn.set_label("Resume" if paused else "Pause")

    def _on_debug_open_log_clicked(self, *_args):
        path_value = getattr(self, "_debug_log_path", None)
        if not path_value:
            self._update_debug_log_button()
            return

        try:
            path = Path(path_value)
        except (TypeError, ValueError):
            logger.warning("Invalid debug log path: %s", path_value)
            self._update_debug_log_button()
            return

        if not path.exists():
            logger.warning("Debug log path does not exist: %s", path)
            self._update_debug_log_button()
            return

        uri_launcher_cls = getattr(Gtk, "UriLauncher", None)
        if uri_launcher_cls is not None:
            launcher = None
            try:
                if hasattr(uri_launcher_cls, "new"):
                    launcher = uri_launcher_cls.new(path.as_uri())
            except Exception:
                launcher = None

            if launcher is None:
                try:
                    launcher = uri_launcher_cls(uri=path.as_uri())
                except TypeError:
                    try:
                        launcher = uri_launcher_cls()
                        if hasattr(launcher, "set_uri"):
                            launcher.set_uri(path.as_uri())
                        else:
                            setattr(launcher, "uri", path.as_uri())
                    except Exception:
                        launcher = None
                except Exception:
                    launcher = None

            if launcher is not None:
                try:
                    launcher.launch(self, None, None)
                    return
                except TypeError:
                    try:
                        launcher.launch(self, None)
                        return
                    except Exception:
                        logger.warning("Failed to launch debug log via Gtk.UriLauncher", exc_info=True)
                except Exception:
                    logger.warning("Failed to launch debug log via Gtk.UriLauncher", exc_info=True)

        platform_value = sys.platform
        fallback_order = "Windows os.startfile -> macOS open -> Linux xdg-open"

        if platform_value.startswith("win"):
            attempted = "Windows os.startfile"
        elif platform_value == "darwin":
            attempted = "macOS open"
        else:
            attempted = "Linux xdg-open"

        logger.debug(
            "Gtk.UriLauncher unavailable or failed; attempting %s (fallback order: %s)",
            attempted,
            fallback_order,
        )

        if platform_value.startswith("win"):
            try:
                os.startfile(str(path))
                return
            except Exception:
                logger.warning(
                    "Failed to open debug log via Windows os.startfile", exc_info=True
                )
                self._update_debug_log_button()
                return
        elif platform_value == "darwin":
            try:
                subprocess.Popen(["open", str(path)])
                return
            except Exception:
                logger.warning(
                    "Failed to open debug log via macOS 'open' command", exc_info=True
                )
                self._update_debug_log_button()
                return
        else:
            try:
                command = f"xdg-open {shlex.quote(str(path))}"
                GLib.spawn_command_line_async(command)
                return
            except Exception:
                logger.warning("Failed to open debug log via xdg-open", exc_info=True)
                self._update_debug_log_button()

    def _on_debug_level_changed(self, combo: Gtk.ComboBoxText):
        if self._debug_controls_updating:
            return
        index = combo.get_active()
        if index < 0:
            text = combo.get_active_text()
            level_name = text or "INFO"
        else:
            level_name = self._debug_level_options[min(index, len(self._debug_level_options) - 1)]
        level_value = getattr(logging, level_name, None)
        if not isinstance(level_value, int):
            try:
                level_value = int(level_name)
            except (TypeError, ValueError):
                level_value = getattr(logging, str(level_name).upper(), logging.INFO)
        if not isinstance(level_value, int):
            level_value = logging.INFO
        self._debug_log_config["level"] = level_value
        if self._debug_log_handler is not None:
            self._debug_log_handler.setLevel(level_value)
        if isinstance(self._debug_level_filter, _HandlerLevelFilter):
            self._debug_level_filter.set_level(level_value)

        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        if ui_config is not None:
            try:
                ui_config.set_debug_log_level(level_value)
            except Exception:
                logger.warning(
                    "Failed to persist UI debug log level", exc_info=True
                )
        elif manager is not None:
            setter = getattr(manager, "set_ui_debug_log_level", None)
            if callable(setter):
                try:
                    setter(level_value)
                except Exception:
                    logger.warning(
                        "Failed to persist UI debug log level", exc_info=True
                    )

    def _on_debug_logger_entry_focus_leave(
        self, controller: Gtk.EventControllerFocus, *_args
    ) -> bool:
        entry: Optional[Gtk.Entry]
        widget = None
        if controller is not None:
            getter = getattr(controller, "get_widget", None)
            if callable(getter):
                widget = getter()
        if isinstance(widget, Gtk.Entry):
            entry = widget
        else:
            entry = getattr(self, "debug_logger_entry", None)
        if isinstance(entry, Gtk.Entry):
            self._on_debug_logger_names_committed(entry)
        return False

    def _on_debug_logger_names_committed(self, entry: Gtk.Entry, *_args):
        if self._debug_controls_updating:
            return False

        text = entry.get_text() if isinstance(entry, Gtk.Entry) else ""
        updated_names: List[str] = []
        for part in text.split(","):
            sanitized = part.strip()
            if sanitized:
                updated_names.append(sanitized)

        entry.set_text(", ".join(updated_names))

        current_names = list(self._debug_log_config.get("logger_names") or [])
        if updated_names == current_names:
            return False

        self._debug_log_config["logger_names"] = updated_names
        self._attach_debug_log_handler()

        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        target = None
        if ui_config is not None:
            target = getattr(ui_config, "set_debug_logger_names", None)
        elif manager is not None:
            target = getattr(manager, "set_ui_debug_logger_names", None)

        if callable(target):
            try:
                target(updated_names)
            except Exception as exc:
                logger.warning("Failed to persist debug logger names: %s", exc)

        return False

    def _on_debug_retention_changed(self, spin: Gtk.SpinButton):
        if self._debug_controls_updating:
            return
        value = spin.get_value_as_int()

        self._debug_log_config["max_lines"] = value
        if self._debug_log_handler is not None:
            self._debug_log_handler.set_max_lines(value)

        manager = getattr(self.ATLAS, "config_manager", None)
        ui_config = getattr(self.ATLAS, "ui_config", None) or getattr(manager, "ui_config", None)
        setter = None
        if ui_config is not None:
            setter = getattr(ui_config, "set_debug_log_max_lines", None)
        elif manager is not None:
            setter = getattr(manager, "set_ui_debug_log_max_lines", None)

        persisted_value: Optional[int] = None
        if callable(setter):
            try:
                persisted_value = setter(value)
            except Exception as exc:
                logger.warning("Failed to persist debug retention limit: %s", exc)

        if isinstance(persisted_value, int) and persisted_value != value:
            self._debug_controls_updating = True
            try:
                spin.set_value(float(persisted_value))
                self._debug_log_config["max_lines"] = persisted_value
                if self._debug_log_handler is not None:
                    self._debug_log_handler.set_max_lines(persisted_value)
            finally:
                self._debug_controls_updating = False

    def _format_conversation_history(self, history: List[Dict[str, object]]) -> str:
        if not history:
            return "No conversation messages recorded yet."

        lines: list[str] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue

            timestamp = entry.get("timestamp") or ""
            role = entry.get("role") or "unknown"
            lines.append(f"[{timestamp}] {role}")

            content = self._stringify_terminal_value(entry.get("content"))
            if content:
                for segment in content.splitlines() or [""]:
                    lines.append(f"  {segment}")

            metadata = entry.get("metadata")
            if metadata:
                metadata_text = self._stringify_terminal_value(metadata)
                if metadata_text:
                    lines.append(f"  metadata: {metadata_text}")

            extra_keys = [
                key for key in entry.keys() if key not in {"timestamp", "role", "content", "metadata"}
            ]
            for key in extra_keys:
                value_text = self._stringify_terminal_value(entry.get(key))
                if value_text:
                    lines.append(f"  {key}: {value_text}")

            lines.append("")

        formatted = "\n".join(lines).strip()
        return formatted or "No conversation messages recorded yet."

    def _format_negotiation_history(self, history: List[Dict[str, object]]) -> str:
        if not history:
            return "No negotiation activity recorded."

        lines: list[str] = []
        for entry in history:
            if not isinstance(entry, Mapping):
                continue

            protocol = str(entry.get("protocol") or "").upper() or "UNKNOWN"
            status = str(entry.get("status") or "").upper() or "PENDING"
            started = entry.get("started_at")
            timestamp: Optional[str] = None
            if isinstance(started, (int, float)):
                try:
                    timestamp = datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:  # pragma: no cover - defensive formatting
                    timestamp = None

            header_bits = [protocol, status]
            if timestamp:
                header_bits.append(timestamp)
            lines.append(" • ".join(bit for bit in header_bits if bit))

            selected = entry.get("selected")
            if isinstance(selected, Mapping) and selected:
                participant = selected.get("participant") or "unknown"
                score = selected.get("score")
                selection_summary = f"Selected: {participant}"
                if score is not None:
                    selection_summary += f" (score={score})"
                lines.append(f"  {selection_summary}")
                selected_text = selected.get("content")
                if selected_text:
                    excerpt = str(selected_text).strip()
                    if len(excerpt) > 160:
                        excerpt = excerpt[:157] + "..."
                    if excerpt:
                        lines.append(f"    {excerpt}")

            participants = entry.get("participants")
            if isinstance(participants, list):
                for participant_entry in participants:
                    if not isinstance(participant_entry, Mapping):
                        continue
                    participant_name = participant_entry.get("participant") or "agent"
                    participant_status = participant_entry.get("status") or ""
                    participant_score = participant_entry.get("score")
                    summary = f"  - {participant_name}: {participant_status}"
                    if participant_score is not None:
                        summary += f" (score={participant_score})"
                    lines.append(summary)
                    rationale = participant_entry.get("rationale")
                    if rationale:
                        lines.append(f"      rationale: {rationale}")

            notes = entry.get("notes")
            if notes:
                lines.append(f"  Notes: {notes}")

            lines.append("")

        return "\n".join(lines).strip() or "No negotiation activity recorded."

    def _stringify_terminal_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)
        except Exception:  # pragma: no cover - defensive fallback
            logger.debug("Unable to stringify value for terminal view", exc_info=True)
            return str(value)

    # --------------------------- Persona helpers ---------------------------

    def update_persona_label(self):
        """
        Update the window title with the current persona's name.
        """
        persona_name = self.ATLAS.get_active_persona_name()

        user_display = getattr(self, "_current_user_display_name", None)

        self._refresh_blackboard_tab()

        # Update the title on the containing window when available. ``ChatPage``
        # is a ``Gtk.Box`` embedded inside the main notebook, so it no longer
        # inherits a ``set_title`` method directly. Query the current root
        # widget and update its title when supported to avoid attribute errors
        # on GTK4.
        target_title: Optional[str]
        if user_display:
            target_title = f"{persona_name} • {user_display}"
        else:
            target_title = persona_name

        if not target_title:
            return

        # Prefer ``self.set_title`` for backwards compatibility (e.g. unit
        # tests that monkeypatch the method). Fallback to the root window.
        set_title = getattr(self, "set_title", None)
        if callable(set_title):
            set_title(target_title)
            return

        root = getattr(self, "get_root", None)
        if callable(root):
            root_widget = root()
            if root_widget is not None:
                window_set_title = getattr(root_widget, "set_title", None)
                if callable(window_set_title):
                    window_set_title(target_title)

    def _register_active_user_listener(self) -> None:
        def _listener(username: str, display_name: str) -> None:
            GLib.idle_add(self._apply_active_user_identity, username, display_name)

        try:
            self.ATLAS.add_active_user_change_listener(_listener)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Unable to subscribe to active user changes: %s", exc, exc_info=True)
            self._active_user_listener = None
        else:
            self._active_user_listener = _listener

    def _register_persona_change_listener(self) -> None:
        """Subscribe to persona context updates when the backend supports it."""

        registrar = getattr(self.ATLAS, "add_persona_change_listener", None)
        if not callable(registrar):
            self._persona_change_handler = None
            return

        def _listener(context: Optional[Dict[str, object]]) -> None:
            GLib.idle_add(self._on_persona_context_changed, context)

        try:
            registrar(_listener)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Unable to subscribe to persona changes: %s", exc, exc_info=True)
            self._persona_change_handler = None
        else:
            self._persona_change_handler = _listener

    def _register_tool_activity_listener(self) -> None:
        """Listen for tool activity updates to refresh terminal diagnostics."""

        def _listener(entry: Optional[Dict[str, object]]) -> None:
            GLib.idle_add(self._on_tool_activity_event, entry)

        try:
            handle = self.ATLAS.subscribe_event(
                "tool_activity",
                _listener,
                legacy_only=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Unable to subscribe to tool activity events: %s", exc, exc_info=True)
            self._tool_activity_subscription = None
        else:
            self._tool_activity_subscription = handle

    def _on_persona_context_changed(self, context: Optional[Dict[str, object]]) -> bool:
        self.update_persona_label()
        self._refresh_terminal_tab(context)
        return False

    def _on_tool_activity_event(self, _entry: Optional[Dict[str, object]]) -> bool:
        history: Optional[List[Dict[str, object]]] = None
        history_getter = getattr(self.ATLAS, "get_tool_activity_log", None)
        if callable(history_getter):
            try:
                candidate = history_getter()
            except Exception as exc:
                logger.error("Failed to refresh tool activity log: %s", exc, exc_info=True)
            else:
                if isinstance(candidate, list):
                    history = candidate

        self._refresh_tool_sections(history=history)
        return False

    def _apply_active_user_identity(self, username: str, display_name: str) -> bool:
        self._current_user_display_name = display_name or username
        label = getattr(self, "user_title_label", None)
        if label is not None:
            text = f"Active user: {self._current_user_display_name or ''}".rstrip()
            setter = getattr(label, "set_text", None)
            if callable(setter):
                setter(text)
        self.update_persona_label()
        self.update_status_bar()
        return False

    # --------------------------- Actions ---------------------------

    def on_send_message(self, widget):
        """
        Handler for when the user sends a message.
        """
        if self.awaiting_response:
            return
        buffer = self.input_buffer
        start_iter = buffer.get_start_iter()
        end_iter = buffer.get_end_iter()
        message = buffer.get_text(start_iter, end_iter, True).strip()
        if not message:
            return
        user_name = self.ATLAS.get_user_display_name()
        self.add_message_bubble(user_name, message, is_user=True, audio=None)
        buffer.set_text("")
        self.input_textview.grab_focus()
        self._set_busy_state(True)

        def handle_success(persona_name: str, response_payload):
            display_name = persona_name or "Assistant"

            def update():
                normalized = self._normalize_response_payload(response_payload)
                bubble_kwargs = {
                    "audio": normalized.get("audio"),
                    "thinking": normalized.get("thinking"),
                }
                timestamp_value = normalized.get("timestamp")
                if timestamp_value is not None:
                    bubble_kwargs["timestamp"] = timestamp_value
                self.add_message_bubble(
                    display_name,
                    normalized["text"],
                    **bubble_kwargs,
                )
                self._on_response_complete()
                self._refresh_terminal_tab()
                return False

            GLib.idle_add(update)

        def handle_error(persona_name: str, exc: Exception):
            display_name = persona_name or "Assistant"
            logger.error(f"Error retrieving model response: {exc}")

            def update():
                self.add_message_bubble(
                    display_name,
                    f"Error: {exc}",
                    audio=None,
                    thinking=None,
                )
                self._on_response_complete()
                self._refresh_terminal_tab()
                return False

            GLib.idle_add(update)

        self.ATLAS.send_chat_message_async(
            message,
            on_success=handle_success,
            on_error=handle_error,
            thread_name="ChatResponseWorker",
        )

    def on_key_pressed(self, _controller, keyval, keycode, state):
        """
        Global key handling for convenient editing:
         - Ctrl/⌘ + L focuses the text input area.
        """
        ctrl = (state & Gdk.ModifierType.CONTROL_MASK) or (state & Gdk.ModifierType.META_MASK)
        if ctrl and keyval in (Gdk.KEY_l, Gdk.KEY_L):
            self.input_textview.grab_focus()
            return True
        return False

    def on_textview_key_pressed(self, _controller, keyval, keycode, state):
        """Handle Enter/Shift+Enter behavior within the text view."""
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            shift = bool(state & Gdk.ModifierType.SHIFT_MASK)
            if shift:
                return False  # allow newline insertion
            self.on_send_message(self.input_textview)
            return True  # prevent default newline
        return False

    def on_mic_button_click(self, _widget):
        """Toggle speech-to-text recording/transcription using backend helpers."""

        if not self.mic_state_listening:
            payload = self.ATLAS.start_stt_listening()
            self._apply_stt_state(payload)
            if not payload.get("ok", False):
                GLib.timeout_add_seconds(
                    2, lambda: (self.update_status_bar(payload.get("status_summary")) or False)
                )
            return

        payload = self.ATLAS.stop_stt_and_transcribe()
        self._apply_stt_state(payload)

        future = payload.get("transcription_future")
        if isinstance(future, Future):
            future.add_done_callback(self._on_transcription_future_ready)
        elif payload.get("error"):
            GLib.timeout_add_seconds(
                2, lambda: (self.update_status_bar(payload.get("status_summary")) or False)
            )

    def _on_transcription_future_ready(self, future: Future):
        try:
            result_payload = future.result()
        except Exception as exc:  # noqa: BLE001 - ensure UI reflects failure gracefully.
            logger.error("Transcription future raised an exception: %s", exc, exc_info=True)
            result_payload = {
                "ok": False,
                "status_text": f"Transcription failed: {exc}",
                "provider": None,
                "listening": False,
                "spinner": False,
                "transcript": "",
                "error": f"Transcription failed: {exc}",
                "status_tooltip": self._default_status_tooltip,
                "status_summary": self.ATLAS.get_chat_status_summary(),
            }

        def dispatch():
            self._handle_transcription_payload(result_payload)
            return False

        GLib.idle_add(dispatch)

    def _handle_transcription_payload(self, payload: dict):
        transcript = (payload.get("transcript") or "").strip()
        error_message = payload.get("error")

        if transcript:
            buf = self.input_buffer
            insertion = transcript
            if buf.get_char_count() > 0:
                insertion = "\n" + insertion
            buf.insert(buf.get_end_iter(), insertion)
            buf.place_cursor(buf.get_end_iter())
        elif error_message:
            logger.error(error_message)

        self._apply_stt_state(payload, final=True)

    def _apply_stt_state(self, payload: dict, *, final: bool = False):
        if not isinstance(payload, dict):
            return

        if "listening" in payload:
            self._set_mic_visual(bool(payload.get("listening")))

        if "spinner" in payload:
            self._set_spinner_active(bool(payload.get("spinner")))

        tooltip_text = payload.get("status_tooltip") or self._default_status_tooltip
        self.status_label.set_tooltip_text(tooltip_text)

        status_text = payload.get("status_text")
        if status_text:
            self.status_label.set_text(status_text)

        if final:
            GLib.timeout_add_seconds(
                2, lambda: (self.update_status_bar(payload.get("status_summary")) or False)
            )

    def _set_mic_visual(self, listening: bool):
        self.mic_state_listening = listening
        self.mic_button.set_child(self._mic_icons["listening" if listening else "idle"])
        self.mic_button.set_tooltip_text("Stop listening (STT)" if listening else "Start listening (STT)")

    def _set_spinner_active(self, active: bool):
        self.status_spinner.set_visible(active)
        if active:
            self.status_spinner.start()
        else:
            self.status_spinner.stop()

    def add_message_bubble(
        self,
        sender,
        message,
        is_user=False,
        audio=None,
        thinking=None,
        timestamp=None,
    ):
        """
        Adds a message bubble to the conversation history area.

        Args:
            sender (str): The name of the message sender.
            message (str): The message content.
            is_user (bool): True if the message is from the user, False if from the assistant.
        """
        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        bubble.set_margin_top(4)
        bubble.set_margin_bottom(4)

        header_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        sender_label = Gtk.Label(label=sender)
        sender_label.set_halign(Gtk.Align.START)
        sender_label.add_css_class("caption")
        header_row.append(sender_label)
        header_row.append(self._spacer())
        # timestamp-ish (lightweight)
        display_time, tooltip_text = self._format_message_timestamp(timestamp)
        ts = Gtk.Label(label=display_time)
        ts.set_tooltip_text(tooltip_text)
        ts.add_css_class("dim-label")
        ts.set_halign(Gtk.Align.END)
        header_row.append(ts)
        bubble.append(header_row)

        message_label = Gtk.Label(label=message)
        message_label.set_wrap(True)
        message_label.set_max_width_chars(46)
        message_label.set_halign(Gtk.Align.START)
        message_label.set_selectable(True)
        message_label.set_tooltip_text("Right-click for options")

        bubble_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        bubble_box.append(message_label)
        bubble_box.get_style_context().add_class("message-bubble")

        if audio:
            controls = self._build_audio_controls(audio)
            if controls is not None:
                bubble_box.append(controls)

        # Apply different styling based on whether the message is from the user or assistant.
        if is_user:
            bubble_box.get_style_context().add_class("user-message")
            bubble.set_halign(Gtk.Align.END)
        else:
            bubble_box.get_style_context().add_class("assistant-message")
            bubble.set_halign(Gtk.Align.START)

        # Context menu: Copy / Delete
        click = Gtk.GestureClick(button=3)  # right-click
        click.connect("pressed", self._on_bubble_right_click, message_label, bubble)
        bubble_box.add_controller(click)

        bubble.append(bubble_box)
        self.chat_history.append(bubble)

        if not is_user:
            self._update_terminal_thinking(thinking)

        # Schedule scrolling to the bottom after a short delay.
        GLib.timeout_add(100, self.scroll_to_bottom)

    def _update_terminal_thinking(self, thinking: Optional[str]) -> None:
        expander = getattr(self, "thinking_expander", None)
        buffer = getattr(self, "thinking_buffer", None)
        placeholder = getattr(self, "terminal_placeholder_label", None)

        if expander is None or buffer is None:
            return

        has_thinking = thinking is not None and thinking != ""
        if has_thinking:
            buffer.set_text(str(thinking))
            expander.set_visible(True)
            if hasattr(expander, "set_expanded"):
                expander.set_expanded(False)
            if placeholder is not None:
                placeholder.set_visible(False)
        else:
            buffer.set_text("")
            expander.set_visible(False)
            if placeholder is not None:
                placeholder.set_visible(True)

    def _set_busy_state(self, busy: bool):
        self.awaiting_response = busy
        self.send_button.set_sensitive(not busy)
        if busy:
            self.status_spinner.set_visible(True)
            self.status_spinner.start()
        else:
            self.status_spinner.stop()
            self.status_spinner.set_visible(False)

    def _on_response_complete(self):
        self._set_busy_state(False)
        return False

    def _on_bubble_right_click(self, gesture, n_press, x, y, message_label: Gtk.Label, bubble: Gtk.Box):
        menu = Gtk.PopoverMenu()
        model = Gio.Menu()
        model.append("Copy", "bubble.copy")
        model.append("Delete", "bubble.delete")
        menu.set_menu_model(model)
        menu.set_has_arrow(False)

        # Actions
        copy_action = Gio.SimpleAction.new("copy", None)
        copy_action.connect("activate", lambda *_: self._copy_label_text(message_label))
        delete_action = Gio.SimpleAction.new("delete", None)
        delete_action.connect("activate", lambda *_: self._delete_bubble(bubble))

        action_group = Gio.SimpleActionGroup()
        action_group.add_action(copy_action)
        action_group.add_action(delete_action)
        bubble.insert_action_group("bubble", action_group)

        menu.set_parent(bubble)
        menu.set_pointing_to(Gdk.Rectangle(int(x), int(y), 1, 1))
        menu.popup()

    def _copy_label_text(self, label: Gtk.Label):
        text = label.get_text() or ""
        display = self.get_display()
        display.get_clipboard().set(text)

    def _delete_bubble(self, bubble: Gtk.Box):
        parent = bubble.get_parent()
        if parent:
            parent.remove(bubble)

    def _spacer(self):
        s = Gtk.Box()
        s.set_hexpand(True)
        return s

    def _format_message_timestamp(self, timestamp):
        if timestamp is None:
            dt_value = datetime.now()
            formatted = dt_value.strftime("%Y-%m-%d %H:%M:%S")
            return dt_value.strftime("%H:%M"), formatted

        if isinstance(timestamp, datetime):
            formatted = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            return timestamp.strftime("%H:%M"), formatted

        if isinstance(timestamp, (int, float)):
            try:
                dt_value = datetime.fromtimestamp(timestamp)
            except (TypeError, ValueError, OSError):
                text = str(timestamp)
                return text, text
            formatted = dt_value.strftime("%Y-%m-%d %H:%M:%S")
            return dt_value.strftime("%H:%M"), formatted

        if isinstance(timestamp, str):
            stripped = timestamp.strip()
            if not stripped:
                return "", ""
            parsers = [
                lambda value: datetime.fromisoformat(value),
                lambda value: datetime.strptime(value, "%Y-%m-%d %H:%M:%S"),
            ]
            for parser in parsers:
                try:
                    dt_value = parser(stripped)
                except ValueError:
                    continue
                else:
                    return dt_value.strftime("%H:%M"), stripped
            return stripped, stripped

        text = str(timestamp)
        return text, text

    def _normalize_response_payload(self, payload) -> dict:
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            audio_data = payload.get("audio")
            audio_payload = None
            if audio_data:
                audio_payload = {
                    "data": audio_data,
                    "format": payload.get("audio_format"),
                    "voice": payload.get("audio_voice"),
                    "id": payload.get("audio_id"),
                }
            normalized = {"text": text, "audio": audio_payload}
            if payload.get("thinking") is not None:
                normalized["thinking"] = str(payload.get("thinking") or "")
            timestamp_value = payload.get("timestamp")
            if timestamp_value is None:
                timestamp_value = payload.get("created_at")
            if timestamp_value is not None:
                normalized["timestamp"] = timestamp_value
            return normalized

        return {"text": str(payload or ""), "audio": None}

    def _build_audio_controls(self, audio_payload: dict):
        data = audio_payload.get("data")
        if not isinstance(data, (bytes, bytearray)):
            return None

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.add_css_class("audio-response-box")

        description = audio_payload.get("voice") or "Audio response"
        label = Gtk.Label(label=str(description))
        label.add_css_class("dim-label")
        label.set_halign(Gtk.Align.START)
        controls.append(label)

        play_button = Gtk.Button(label="Play audio")
        play_button.add_css_class("flat")
        play_button.connect(
            "clicked",
            lambda *_: self._play_audio_bytes(bytes(data), audio_payload.get("format")),
        )
        controls.append(play_button)

        save_button = Gtk.Button(label="Save audio")
        save_button.add_css_class("flat")
        save_button.connect(
            "clicked",
            lambda *_: self._save_audio_bytes(bytes(data), audio_payload.get("format")),
        )
        controls.append(save_button)

        return controls

    def _guess_audio_extension(self, fmt: Optional[str]) -> str:
        if not fmt:
            return ".wav"
        normalized = fmt.lower().strip()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        return normalized

    def _play_audio_bytes(self, data: bytes, fmt: Optional[str]) -> None:
        suffix = self._guess_audio_extension(fmt)
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                temp.write(data)
                temp_path = Path(temp.name)
        except Exception as exc:  # pragma: no cover - filesystem fallback
            logger.error("Failed to write temporary audio file: %s", exc)
            self.status_label.set_text("Unable to play audio (write error).")
            return

        uri = temp_path.as_uri()
        try:
            Gio.AppInfo.launch_default_for_uri(uri)
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.error("Failed to launch audio player: %s", exc)
            self.status_label.set_text("Unable to open audio player.")

    def _save_audio_bytes(self, data: bytes, fmt: Optional[str]) -> None:
        suffix = self._guess_audio_extension(fmt)
        target_name = datetime.now().strftime("response-%Y%m%d-%H%M%S") + suffix
        target_path = self._audio_output_dir / target_name

        try:
            with target_path.open("wb") as handle:
                handle.write(data)
        except Exception as exc:
            logger.error("Failed to save audio response: %s", exc)
            self.status_label.set_text("Unable to save audio response.")
            return

        self.status_label.set_text(f"Audio saved to {target_path}")

    def scroll_to_bottom(self):
        """
        Scrolls the chat history scrolled window to the bottom.

        Returns:
            bool: False to stop further timeout events.
        """
        vadj = self.chat_history_scrolled.get_vadjustment()
        # Use upper - page_size so we truly land at the bottom without overshoot
        bottom = max(0.0, vadj.get_upper() - vadj.get_page_size())
        vadj.set_value(bottom)
        return False  # Stop the timeout

    def _on_close_request(self, *_args):
        """Unregister provider change listeners before the window closes."""

        self._detach_debug_log_handler()
        if self._provider_change_handler is not None:
            self.ATLAS.remove_provider_change_listener(self._provider_change_handler)
            self._provider_change_handler = None
        if self._active_user_listener is not None:
            self.ATLAS.remove_active_user_change_listener(self._active_user_listener)
            self._active_user_listener = None
        if self._persona_change_handler is not None:
            remover = getattr(self.ATLAS, "remove_persona_change_listener", None)
            if callable(remover):
                try:
                    remover(self._persona_change_handler)
                except Exception as exc:  # pragma: no cover - defensive logging only
                    logger.error("Failed to remove persona change listener: %s", exc, exc_info=True)
            self._persona_change_handler = None
        if self._tool_activity_subscription is not None:
            self.ATLAS.unsubscribe_event(self._tool_activity_subscription)
            self._tool_activity_subscription = None
        return False

    def update_status_bar(self, status_summary=None):
        """
        Updates the status label with current LLM provider/model information as well as
        the active TTS provider and its selected voice.
        """
        if not status_summary:
            try:
                status_summary = self.ATLAS.get_chat_status_summary()
            except Exception as exc:
                logger.error("Failed to fetch chat status summary: %s", exc, exc_info=True)
                status_summary = {}
        try:
            status_message = self.ATLAS.format_chat_status(status_summary)
        except Exception as exc:
            logger.error("Failed to format chat status: %s", exc, exc_info=True)
            status_message = "LLM: Unknown • Model: No model selected • TTS: None (Voice: Not Set)"

        self.status_label.set_text(status_message)


# Imports required by popover menu action wiring
from gi.repository import Gio  # noqa: E402  (keep at bottom for clarity)
