# UI/Chat/chat_page.py

"""
This module implements the ChatPage window, which displays the conversation
history, an input field, a microphone button for speech-to-text, and a send button.
It handles user input and displays both user messages and responses from the ATLAS language model.

It features robust error handling, nonblocking asynchronous processing via threads,
and schedules UI updates via GLib.idle_add.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from concurrent.futures import Future
from datetime import datetime
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import logging

from GTKUI.Utils.utils import apply_css

# Configure logging for the chat page.
logger = logging.getLogger(__name__)


class ChatPage(Gtk.Window):
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
        super().__init__()
        self.ATLAS = atlas
        self.set_default_size(700, 520)

        # Apply style classes to match the dark theme of the sidebar.
        self.get_style_context().add_class("chat-page")
        self.get_style_context().add_class("sidebar")

        # Apply centralized CSS styling.
        apply_css()

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

        # --- Terminal tab placeholder ---
        self.terminal_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.terminal_tab_box.set_hexpand(True)
        self.terminal_tab_box.set_vexpand(True)
        terminal_label = Gtk.Label(label="Terminal")
        terminal_label.add_css_class("caption")
        self.notebook.append_page(self.terminal_tab_box, terminal_label)

        # --- Debug tab placeholder ---
        self.debug_tab_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.debug_tab_box.set_hexpand(True)
        self.debug_tab_box.set_vexpand(True)
        debug_label = Gtk.Label(label="Debug")
        debug_label.add_css_class("caption")
        self.notebook.append_page(self.debug_tab_box, debug_label)

        self.vbox.append(self.notebook)

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

        # Link provider changes to update the status bar.
        self._provider_change_handler = self.update_status_bar
        self.ATLAS.add_provider_change_listener(self._provider_change_handler)
        self._active_user_listener = None
        self._current_user_display_name = self.ATLAS.get_user_display_name()
        self.user_title_label.set_text(f"Active user: {self._current_user_display_name}")
        self.update_persona_label()
        self._register_active_user_listener()
        self.connect("close-request", self._on_close_request)

        self.awaiting_response = False
        self._audio_output_dir = Path.home() / ".atlas" / "audio_responses"
        try:
            self._audio_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem issues are logged but non-fatal
            logger.warning("Unable to create audio output directory: %s", exc)
            self._audio_output_dir = Path.home()
        self._export_dialog = None

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
                )
                self._on_response_complete()
                return False

            GLib.idle_add(update)

        def handle_error(persona_name: str, exc: Exception):
            display_name = persona_name or "Assistant"
            logger.error(f"Error retrieving model response: {exc}")

            def update():
                self.add_message_bubble(display_name, f"Error: {exc}", audio=None)
                self._on_response_complete()
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

    def add_message_bubble(self, sender, message, is_user=False, audio=None):
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

        # Schedule scrolling to the bottom after a short delay.
        GLib.timeout_add(100, self.scroll_to_bottom)

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
            return {"text": text, "audio": audio_payload}

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
        result = self.ATLAS.reset_chat_history()
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

        self.ATLAS.remove_provider_change_listener(self._provider_change_handler)
        if self._active_user_listener is not None:
            self.ATLAS.remove_active_user_change_listener(self._active_user_listener)
            self._active_user_listener = None
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
