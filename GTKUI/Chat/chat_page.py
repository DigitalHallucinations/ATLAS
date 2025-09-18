# UI/Chat/chat_page.py

"""
This module implements the ChatPage window, which displays the conversation
history, an input field, a microphone button for speech-to-text, and a send button.
It handles user input and displays both user messages and responses from the ATLAS language model.

It features robust error handling, nonblocking asynchronous processing via threads,
and schedules UI updates via GLib.idle_add.
"""

import os
import asyncio
import threading
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib, GObject
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
        self.chat_session = atlas.chat_session
        self.set_default_size(700, 520)

        # Apply style classes to match the dark theme of the sidebar.
        self.get_style_context().add_class("chat-page")
        self.get_style_context().add_class("sidebar")

        # Apply centralized CSS styling.
        apply_css()

        # --- Header bar with persona title & quick actions ---
        self.header_bar = Gtk.HeaderBar()
        self.set_titlebar(self.header_bar)

        # Persona title label inside header
        self.persona_title_label = Gtk.Label(xalign=0)
        self.persona_title_label.add_css_class("title-1")
        self.persona_title_label.set_tooltip_text("Current persona")
        self.header_bar.set_title_widget(self.persona_title_label)

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
        self.vbox.append(self.chat_history_scrolled)

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

        self.vbox.append(input_box)

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
        self.status_label.set_tooltip_text("Active LLM provider/model and TTS status")
        status_box.append(self.status_label)

        self.vbox.append(status_box)
        self.update_status_bar()

        # Link provider changes to update the status bar.
        self.ATLAS.notify_provider_changed = self.update_status_bar

        self.awaiting_response = False

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
        persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Chat')
        self.set_title(persona_name)
        self.persona_title_label.set_text(persona_name)

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
        user_name = self.ATLAS.user
        self.add_message_bubble(user_name, message, is_user=True)
        buffer.set_text("")
        self.input_textview.grab_focus()
        self._set_busy_state(True)
        # Start a new thread to process the model's response asynchronously.
        threading.Thread(
            target=self.handle_model_response_thread,
            args=(message,),
            daemon=True
        ).start()

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

    def on_mic_button_click(self, widget):
        """
        Toggle speech-to-text recording/transcription.
        """
        stt = self.ATLAS.speech_manager.active_stt
        if not stt:
            logger.error("No active STT service configured.")
            self.status_label.set_text("No STT service configured.")
            return

        # Toggle recording based on the STT provider's 'recording' attribute.
        currently_recording = bool(getattr(stt, 'recording', False))
        if not currently_recording:
            logger.info("Starting speech recognition...")
            self._set_mic_visual(listening=True)
            self.status_label.set_text("Listening…")
            try:
                self.ATLAS.speech_manager.listen()
            except Exception as e:
                logger.error(f"Error starting STT: {e}")
                self._set_mic_visual(listening=False)
                self.status_label.set_text("Failed to start listening.")
        else:
            logger.info("Stopping speech recognition and transcribing…")
            try:
                self.ATLAS.speech_manager.stop_listening()
            except Exception as e:
                logger.error(f"Error stopping STT: {e}")
            # Run transcription in a separate thread to avoid blocking the UI.
            def transcribe_and_update():
                try:
                    audio_file = getattr(stt, 'audio_file', "output.wav")
                    transcript = stt.transcribe(audio_file) or ""
                    def update_buffer():
                        text = transcript.strip()
                        if not text:
                            return
                        buf = self.input_buffer
                        if buf.get_char_count() > 0:
                            buf.insert(buf.get_end_iter(), "\n" + text)
                        else:
                            buf.insert(buf.get_end_iter(), text)
                        buf.place_cursor(buf.get_end_iter())
                    GLib.idle_add(update_buffer)
                    GLib.idle_add(lambda: self.status_label.set_text("Transcription complete."))
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    GLib.idle_add(lambda: self.status_label.set_text(f"Transcription failed: {e}"))
                finally:
                    GLib.timeout_add_seconds(2, lambda: (self.update_status_bar() or False))
                    GLib.idle_add(lambda: self._set_mic_visual(listening=False))
            threading.Thread(target=transcribe_and_update, daemon=True).start()

    def _set_mic_visual(self, listening: bool):
        self.mic_state_listening = listening
        self.mic_button.set_child(self._mic_icons["listening" if listening else "idle"])
        self.mic_button.set_tooltip_text("Stop listening (STT)" if listening else "Start listening (STT)")

    def handle_model_response_thread(self, message):
        """
        Processes the model's response asynchronously in a separate thread,
        then schedules the UI update to add the assistant's message bubble.
        Also triggers TTS if enabled.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.chat_session.send_message(message))

            # Trigger TTS for the response if enabled.
            try:
                loop.run_until_complete(self.ATLAS.speech_manager.text_to_speech(response))
            except Exception as tts_err:
                logger.warning(f"TTS error (continuing): {tts_err}")

            persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Assistant')
            GLib.idle_add(self.add_message_bubble, persona_name, response)
        except Exception as e:
            logger.error(f"Error in handle_model_response: {e}")
            GLib.idle_add(self.add_message_bubble, "Assistant", f"Error: {e}")
        finally:
            try:
                loop.close()
            except Exception:
                pass
            GLib.idle_add(self._on_response_complete)

    def add_message_bubble(self, sender, message, is_user=False):
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
        ts = Gtk.Label(label="")
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

        bubble_box = Gtk.Box()
        bubble_box.append(message_label)
        bubble_box.get_style_context().add_class("message-bubble")

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
        self.status_label.set_text("Chat cleared.")
        GLib.timeout_add_seconds(2, lambda: (self.update_status_bar() or False))

    def on_export_chat(self, _btn):
        """
        Exports the chat history to a simple UTF-8 text file chosen by the user.
        """
        dialog = Gtk.FileChooserNative(
            title="Export chat",
            action=Gtk.FileChooserAction.SAVE,
            transient_for=self
        )
        dialog.set_current_name("chat.txt")
        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:
            path = dialog.get_filename()
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for row in self.chat_history.get_children():
                        # row -> bubble (Box)
                        kids = row.get_children()
                        if len(kids) >= 2:
                            # header_row, bubble_box
                            header_row = kids[0]
                            sender = header_row.get_first_child().get_text() if isinstance(header_row.get_first_child(), Gtk.Label) else "Sender"
                            message_label = kids[1].get_first_child()
                            text = message_label.get_text() if isinstance(message_label, Gtk.Label) else ""
                            f.write(f"{sender}: {text}\n\n")
                self.status_label.set_text(f"Exported chat to: {path}")
            except Exception as e:
                logger.error(f"Export error: {e}")
                self.status_label.set_text(f"Export failed: {e}")
        dialog.destroy()

    def update_status_bar(self, provider=None, model=None):
        """
        Updates the status label with current LLM provider/model information as well as
        the active TTS provider and its selected voice.
        """
        # Retrieve LLM provider and model information.
        try:
            llm_provider = self.ATLAS.provider_manager.get_current_provider() or "Unknown"
        except Exception:
            llm_provider = "Unknown"
        try:
            llm_model = self.ATLAS.provider_manager.get_current_model() or "No model selected"
        except Exception:
            llm_model = "No model selected"

        # Retrieve TTS provider and voice information.
        try:
            tts_provider = self.ATLAS.speech_manager.get_default_tts_provider() or "None"
        except Exception:
            tts_provider = "None"

        tts = getattr(self.ATLAS.speech_manager, "active_tts", None)
        tts_voice = "Not Set"
        if tts:
            try:
                if hasattr(tts, "get_current_voice") and callable(getattr(tts, "get_current_voice")):
                    tts_voice = tts.get_current_voice() or tts_voice
                elif hasattr(tts, "voice_ids") and tts.voice_ids:
                    tts_voice = tts.voice_ids[0].get('name', tts_voice)
                elif hasattr(tts, "voice") and tts.voice is not None:
                    # e.g., Google TTS voice object
                    tts_voice = getattr(tts.voice, "name", tts_voice)
            except Exception as e:
                logger.debug(f"TTS voice fetch issue: {e}")

        status_message = (
            f"LLM: {llm_provider} • Model: {llm_model} • "
            f"TTS: {tts_provider} (Voice: {tts_voice})"
        )
        self.status_label.set_text(status_message)


# Imports required by popover menu action wiring
from gi.repository import Gio  # noqa: E402  (keep at bottom for clarity)
