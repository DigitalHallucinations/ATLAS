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
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import Future
from datetime import datetime
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import logging
from logging.handlers import RotatingFileHandler

from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.logging import GTKUILogHandler, read_recent_log_lines
from modules.Tools.tool_event_system import event_system

# Configure logging for the chat page.
logger = logging.getLogger(__name__)


class ChatPage(AtlasWindow):
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
        super().__init__(default_size=(700, 520))
        self.ATLAS = atlas

        # --- Header bar with persona title & quick actions ---
        self.header_bar = Gtk.HeaderBar()
        self.set_titlebar(self.header_bar)

        # Persona title and active user labels inside header
        self.header_title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.header_title_box.set_hexpand(True)

        self.persona_title_label = Gtk.Label(xalign=0)
        self.persona_title_label.add_css_class("title-1")
        self.persona_title_label.set_tooltip_text("Current persona")
        self.header_title_box.append(self.persona_title_label)

        self.user_title_label = Gtk.Label(xalign=0)
        self.user_title_label.add_css_class("caption")
        self.user_title_label.set_tooltip_text("Signed-in account")
        self.header_title_box.append(self.user_title_label)

        self.header_bar.set_title_widget(self.header_title_box)

        # Clear chat button
        self.clear_btn = Gtk.Button()
        self.clear_btn.set_tooltip_text("Clear chat history")
        self._set_button_icon(self.clear_btn, "../../Icons/clear.png", "user-trash")
        self.clear_btn.add_css_class("flat")
        self.clear_btn.connect("clicked", self.on_clear_chat)
        self.header_bar.pack_end(self.clear_btn)

        # Export chat button
        self.export_btn = Gtk.Button()
        self.export_btn.set_tooltip_text("Export chat to a text file")
        self._set_button_icon(self.export_btn, "../../Icons/export.png", "document-save")
        self.export_btn.add_css_class("flat")
        self.export_btn.connect("clicked", self.on_export_chat)
        self.header_bar.pack_end(self.export_btn)

        # Main vertical container.
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_child(self.vbox)

        # Update window title with the current persona's name.
        self.update_persona_label()

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

        # --- Terminal tab with persona context ---
        self.terminal_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.terminal_tab_box.set_hexpand(True)
        self.terminal_tab_box.set_vexpand(True)
        self.terminal_tab_box.set_margin_top(6)
        self.terminal_tab_box.set_margin_bottom(6)
        self.terminal_tab_box.set_margin_start(6)
        self.terminal_tab_box.set_margin_end(6)

        self.terminal_scrolled = Gtk.ScrolledWindow()
        self.terminal_scrolled.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.terminal_scrolled.set_hexpand(True)
        self.terminal_scrolled.set_vexpand(True)
        self.terminal_scrolled.set_child(self.terminal_tab_box)
        self._terminal_sections: Dict[str, Gtk.TextBuffer] = {}
        self._terminal_section_buffers_by_widget: Dict[Gtk.Widget, Gtk.TextBuffer] = {}
        self._tool_log_entries: List[str] = []
        self._tool_activity_listener = None
        self._build_terminal_tab()
        terminal_label = Gtk.Label(label="Terminal")
        terminal_label.add_css_class("caption")
        self.notebook.append_page(self.terminal_scrolled, terminal_label)

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

        self.thinking_label = Gtk.Label()
        self.thinking_label.set_wrap(True)
        self.thinking_label.set_xalign(0.0)
        self.thinking_label.set_halign(Gtk.Align.START)
        self.thinking_label.set_valign(Gtk.Align.START)
        self.thinking_label.add_css_class("monospace")
        self.thinking_expander.set_child(self.thinking_label)

        self.terminal_content_box.append(self.thinking_expander)

        # --- Debug tab with live logging ---
        self._debug_log_handler: Optional[GTKUILogHandler] = None
        self._debug_loggers_attached: List[logging.Logger] = []
        self._debug_handler_name = f"chat-debug-{id(self)}"
        self._debug_log_path: Optional[Path] = None
        self._debug_controls_updating = False
        self._debug_log_config = self._resolve_debug_log_config()
        self._build_debug_tab()

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
        self.user_title_label.set_text(f"Active user: {self._current_user_display_name}")
        self.update_persona_label()
        self._register_active_user_listener()
        self._register_persona_change_listener()
        self._register_tool_activity_listener()
        self.connect("close-request", self._on_close_request)

        self.awaiting_response = False
        self._audio_output_dir = Path.home() / ".atlas" / "audio_responses"
        try:
            self._audio_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem issues are logged but non-fatal
            logger.warning("Unable to create audio output directory: %s", exc)
            self._audio_output_dir = Path.home()
        self._export_dialog = None
        self._initialize_debug_logging()

        self.present()

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

        sections = [
            ("System Prompt", "system_prompt", True),
            ("Persona Data", "persona_data", False),
            ("Conversation Messages", "conversation", True),
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
        text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        if hasattr(text_view, "set_monospace"):
            text_view.set_monospace(True)
        text_view.add_css_class("terminal-text")

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

        copy_button = Gtk.Button(label="Copy")
        copy_button.add_css_class("flat")
        copy_button.set_halign(Gtk.Align.FILL)
        copy_button.set_hexpand(True)
        menu_box.append(copy_button)

        def _copy_buffer_to_clipboard(*_args: object) -> None:
            target_buffer = self._terminal_section_buffers_by_widget.get(section_widget, buffer)
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
            name = str(entry.get("tool_name", "unknown"))
            timestamp = (
                entry.get("completed_at")
                or entry.get("started_at")
                or "Unknown time"
            )
            status = str(entry.get("status", "unknown")).upper()
            duration = entry.get("duration_ms")
            if isinstance(duration, (int, float)):
                duration_text = f"{duration:.0f} ms"
            else:
                duration_text = ""

            args_text = entry.get("arguments_text") or self._stringify_tool_section_value(
                entry.get("arguments")
            )
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
            name = str(entry.get("tool_name", "unknown"))
            status = str(entry.get("status", "unknown")).upper()
            started = entry.get("started_at") or "Unknown start"
            completed = entry.get("completed_at") or "Unknown end"
            duration = entry.get("duration_ms")
            duration_line = None
            if isinstance(duration, (int, float)):
                duration_line = f"Duration: {duration:.0f} ms"

            args_text = entry.get("arguments_text") or self._stringify_tool_section_value(
                entry.get("arguments")
            )
            result_text = entry.get("result_text") or self._stringify_tool_section_value(
                entry.get("result")
            )
            error_text = entry.get("error")
            stdout_text = (entry.get("stdout") or "").strip()
            stderr_text = (entry.get("stderr") or "").strip()

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
        if manager is None:
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
        self.debug_logger_entry.connect("focus-out-event", self._on_debug_logger_names_committed)
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
        self.debug_log_view.set_wrap_mode(Gtk.WrapMode.NONE)
        if hasattr(self.debug_log_view, "set_monospace"):
            self.debug_log_view.set_monospace(True)
        self.debug_log_view.add_css_class("monospace")
        self.debug_log_view.set_hexpand(True)
        self.debug_log_view.set_vexpand(True)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_child(self.debug_log_view)
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
        self._update_debug_log_button()

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
        if manager is None:
            return None

        try:
            configured = manager.get_config("UI_DEBUG_LOG_FILE")
        except Exception:
            configured = None

        try:
            app_root = manager.get_config("APP_ROOT")
        except Exception:
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
            self._update_debug_log_button()
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
        handler.setFormatter(self._resolve_log_formatter())

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
            for logger_obj in attached:
                try:
                    logger_obj.setLevel(level_value)
                except Exception:
                    continue
        else:
            handler.close()

        self._update_debug_log_button()

    def _detach_debug_log_handler(self) -> None:
        handler = self._debug_log_handler
        if handler is None:
            return

        for logger_obj in list(self._debug_loggers_attached):
            try:
                logger_obj.removeHandler(handler)
            except ValueError:
                pass
        handler.close()
        self._debug_loggers_attached = []
        self._debug_log_handler = None

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
        if self._debug_log_handler is None:
            button.set_active(False)
            return
        paused = button.get_active()
        self._debug_log_handler.set_paused(paused)
        self._debug_pause_btn.set_label("Resume" if paused else "Pause")

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

        try:
            command = f"xdg-open {shlex.quote(str(path))}"
            GLib.spawn_command_line_async(command)
        except Exception:
            logger.warning("Failed to open debug log via xdg-open", exc_info=True)

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
        loggers_to_update: List[logging.Logger] = []
        loggers_to_update.extend(list(self._debug_loggers_attached))
        atlas_logger = getattr(self.ATLAS, "logger", None)
        if isinstance(atlas_logger, logging.Logger):
            loggers_to_update.append(atlas_logger)
        seen_ids = set()
        for logger_obj in loggers_to_update:
            if id(logger_obj) in seen_ids:
                continue
            seen_ids.add(id(logger_obj))
            try:
                logger_obj.setLevel(level_value)
            except Exception:
                continue

        manager = getattr(self.ATLAS, "config_manager", None)
        if manager is not None:
            persisted_value: object = logging.getLevelName(level_value)
            if isinstance(persisted_value, int):
                persisted_value = level_value
            if isinstance(getattr(manager, "config", None), dict):
                manager.config["UI_DEBUG_LOG_LEVEL"] = persisted_value
            if isinstance(getattr(manager, "yaml_config", None), dict):
                manager.yaml_config["UI_DEBUG_LOG_LEVEL"] = persisted_value
            writer = getattr(manager, "_write_yaml_config", None)
            if callable(writer):
                try:
                    writer()
                except Exception:
                    logger.warning(
                        "Failed to persist UI debug log level", exc_info=True
                    )

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
        setter = getattr(manager, "set_ui_debug_logger_names", None)
        if callable(setter):
            try:
                setter(updated_names)
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

    # --------------------------- Header helpers ---------------------------

    def update_persona_label(self):
        """
        Updates the window title and header label with the current persona's name.
        """
        persona_name = self.ATLAS.get_active_persona_name()
        self.persona_title_label.set_text(persona_name)

        user_display = getattr(self, "_current_user_display_name", None)
        if hasattr(self, "user_title_label") and user_display:
            self.user_title_label.set_text(f"Active user: {user_display}")

        if user_display:
            self.set_title(f"{persona_name} • {user_display}")
        else:
            self.set_title(persona_name)

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
            event_system.subscribe("tool_activity", _listener)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error("Unable to subscribe to tool activity events: %s", exc, exc_info=True)
            self._tool_activity_listener = None
        else:
            self._tool_activity_listener = _listener

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
        if hasattr(self, "user_title_label"):
            self.user_title_label.set_text(f"Active user: {self._current_user_display_name}")
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
                self.add_message_bubble(
                    display_name,
                    normalized["text"],
                    audio=normalized.get("audio"),
                    thinking=normalized.get("thinking"),
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

    def add_message_bubble(self, sender, message, is_user=False, audio=None, thinking=None):
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
        now = datetime.now()
        ts = Gtk.Label(label=now.strftime("%H:%M"))
        ts.set_tooltip_text(now.strftime("%Y-%m-%d %H:%M:%S"))
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
        label = getattr(self, "thinking_label", None)
        placeholder = getattr(self, "terminal_placeholder_label", None)

        if expander is None or label is None:
            return

        has_thinking = thinking is not None and thinking != ""
        if has_thinking:
            label.set_text(str(thinking))
            expander.set_visible(True)
            if hasattr(expander, "set_expanded"):
                expander.set_expanded(False)
            if placeholder is not None:
                placeholder.set_visible(False)
        else:
            label.set_text("")
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

    def on_clear_chat(self, _btn):
        """
        Clears the current chat history from the UI (does not modify persisted history unless your app does so elsewhere).
        """
        for child in list(self.chat_history.get_children()):
            self.chat_history.remove(child)

        result: Dict[str, Optional[str]] = {}
        try:
            result = self.ATLAS.reset_chat_history() or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to reset chat history: %s", exc)
            result = {"success": False, "error": "Failed to reset chat history."}
        finally:
            # Always clear the terminal/thinking panes so stale content is removed.
            self._refresh_terminal_tab()
            self._update_terminal_thinking(None)

        if result.get("success"):
            status_message = result.get("message", "Chat cleared.")
        else:
            status_message = result.get("error", "Failed to reset chat history.")
        self.status_label.set_text(status_message)
        if result.get("success"):
            status_summary = result.get("status_summary")
            GLib.timeout_add_seconds(
                2, lambda: (self.update_status_bar(status_summary) or False)
            )

    def on_export_chat(self, _btn):
        """Launch a modal dialog that lets the user export the chat history."""
        if self._export_dialog is not None:
            # An export dialog is already active.
            return

        dialog = Gtk.FileChooserNative(
            title="Export chat",
            action=Gtk.FileChooserAction.SAVE,
            transient_for=self,
        )
        dialog.set_modal(True)
        dialog.set_current_name("chat.txt")

        self.export_btn.set_sensitive(False)
        self._export_dialog = dialog
        dialog.connect("response", self._on_export_dialog_response)
        dialog.show()

    def _on_export_dialog_response(self, dialog, response):
        """Handle export dialog responses and perform the chat history export."""
        try:
            if response == Gtk.ResponseType.ACCEPT:
                gio_file = dialog.get_file()
                path = gio_file.get_path() if gio_file is not None else None
                if not path:
                    self.status_label.set_text("No file selected for export.")
                    return

                result = self.ATLAS.export_chat_history(path)
                if result.get("success"):
                    self.status_label.set_text(
                        result.get("message")
                        or "Chat history exported successfully."
                    )
                else:
                    self.status_label.set_text(
                        result.get("error") or "Export failed."
                    )
            else:
                self.status_label.set_text("Export cancelled.")
        finally:
            self.export_btn.set_sensitive(True)
            dialog.destroy()
            if self._export_dialog is dialog:
                self._export_dialog = None

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
        if self._tool_activity_listener is not None:
            try:
                event_system.unsubscribe("tool_activity", self._tool_activity_listener)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.error("Failed to remove tool activity listener: %s", exc, exc_info=True)
            self._tool_activity_listener = None
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
