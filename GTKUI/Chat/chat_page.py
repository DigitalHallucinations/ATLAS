# UI/Chat/chat_page.py

"""
UI/Chat/chat_page.py

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
    It updates the window title with the current persona's name and shows a status bar with provider/model info.
    """
    def __init__(self, atlas):
        """
        Initialize the ChatPage window.

        Args:
            atlas: The main ATLAS application instance.
        """
        super().__init__()
        self.ATLAS = atlas
        self.chat_session = atlas.chat_session
        self.set_default_size(600, 400)
        
        # Add both "chat-page" and "sidebar" style classes to match the dark theme of the sidebar.
        self.get_style_context().add_class("chat-page")
        self.get_style_context().add_class("sidebar")
        
        # Apply centralized CSS.
        apply_css()

        # Main vertical container.
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_child(self.vbox)

        # Update window title with current persona name.
        self.update_persona_label()

        # Add a horizontal separator.
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(5)
        self.vbox.append(separator)

        # Create a scrollable area for chat history.
        self.chat_history_scrolled = Gtk.ScrolledWindow()
        self.chat_history_scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.chat_history_scrolled.set_min_content_height(200)
        self.chat_history = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.chat_history.set_margin_start(10)
        self.chat_history.set_margin_end(10)
        self.chat_history_scrolled.set_child(self.chat_history)
        self.chat_history_scrolled.set_hexpand(True)
        self.chat_history_scrolled.set_vexpand(True)
        self.vbox.append(self.chat_history_scrolled)

        # Create the input area with an Entry, Microphone, and Send button.
        input_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        input_box.set_margin_top(10)
        input_box.set_margin_bottom(10)
        input_box.set_margin_start(10)
        input_box.set_margin_end(10)

        # Text input entry.
        self.input_entry = Gtk.Entry()
        self.input_entry.set_placeholder_text("Type a message...")
        self.input_entry.connect("activate", self.on_send_message)
        self.input_entry.set_hexpand(True)
        input_box.append(self.input_entry)

        # Microphone button for speech-to-text.
        mic_button = Gtk.Button()
        try:
            mic_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Icons/microphone.png")
            mic_texture = Gdk.Texture.new_from_filename(mic_icon_path)
            mic_icon = Gtk.Picture.new_for_paintable(mic_texture)
            mic_icon.set_size_request(24, 24)
            mic_icon.set_content_fit(Gtk.ContentFit.CONTAIN)
        except Exception as e:
            logger.error(f"Error loading microphone icon: {e}")
            mic_icon = Gtk.Image.new_from_icon_name("audio-input-microphone")
        mic_button.set_child(mic_icon)
        mic_button.get_style_context().add_class("mic-button")
        mic_button.connect("clicked", self.on_mic_button_click)
        input_box.append(mic_button)

        # Send button.
        send_button = Gtk.Button()
        try:
            # Compute absolute path for the send icon.
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Icons/send.png")
            texture = Gdk.Texture.new_from_filename(icon_path)
            icon = Gtk.Picture.new_for_paintable(texture)
            icon.set_size_request(24, 24)
            icon.set_content_fit(Gtk.ContentFit.CONTAIN)
        except Exception as e:
            logger.error(f"Error loading send icon: {e}")
            icon = Gtk.Image.new_from_icon_name("image-missing")
        send_button.set_child(icon)
        send_button.get_style_context().add_class("send-button")
        send_button.connect("clicked", self.on_send_message)
        input_box.append(send_button)

        self.vbox.append(input_box)

        # Add a status label at the bottom.
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_margin_start(5)
        self.status_label.set_margin_end(5)
        self.vbox.append(self.status_label)
        self.update_status_bar()

        # Link provider changes to the status bar update.
        self.ATLAS.notify_provider_changed = self.update_status_bar

        self.present()

    def update_persona_label(self):
        """
        Updates the window title with the current persona's name.
        """
        persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Chat')
        self.set_title(persona_name)

    def on_send_message(self, widget):
        """
        Handler for when the user sends a message.

        Reads the input, adds the user's message to the conversation history, clears the input,
        and spawns a separate thread to handle the model's response.
        """
        message = self.input_entry.get_text().strip()
        if message:
            user_name = self.ATLAS.user
            self.add_message_bubble(user_name, message, is_user=True)
            self.input_entry.set_text("")
            # Start a new thread to call the model asynchronously.
            threading.Thread(
                target=self.handle_model_response_thread,
                args=(message,),
                daemon=True
            ).start()

    def on_mic_button_click(self, widget):
        """
        Handler for the microphone button click to toggle speech-to-text.

        If speech recognition is not active, it starts listening and updates the status label.
        If already listening, it stops recording, transcribes the audio, and inserts the transcript into the input field.
        """
        # Get the active STT service from the Speech Manager.
        stt = self.ATLAS.speech_manager.active_stt
        if not stt:
            logger.error("No active STT service configured.")
            return

        # Toggle recording based on the STT's 'recording' attribute.
        if not getattr(stt, 'recording', False):
            logger.info("Starting speech recognition...")
            self.status_label.set_text("Listening...")
            self.ATLAS.speech_manager.listen()
        else:
            logger.info("Stopping speech recognition and transcribing...")
            self.ATLAS.speech_manager.stop_listening()
            
            # Run the transcription in a separate thread to avoid blocking the UI.
            def transcribe_and_update():
                transcript = stt.transcribe("output.wav")
                # Safely update the UI from this thread.
                GLib.idle_add(self.input_entry.set_text, transcript)
                GLib.idle_add(self.status_label.set_text, "Transcription complete.")
            threading.Thread(target=transcribe_and_update, daemon=True).start()

    def handle_model_response_thread(self, message):
        """
        Runs the model response in a separate thread to avoid blocking the UI.

        Creates a dedicated event loop for the async call and schedules the UI update using GLib.idle_add.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.chat_session.send_message(message))
            loop.close()
            persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Assistant')
            GLib.idle_add(self.add_message_bubble, persona_name, response)
        except Exception as e:
            logger.error(f"Error in handle_model_response: {e}")
            GLib.idle_add(self.add_message_bubble, "Assistant", f"Error: {e}")

    def add_message_bubble(self, sender, message, is_user=False):
        """
        Adds a message bubble to the chat history area.

        Args:
            sender (str): The name of the message sender.
            message (str): The message content.
            is_user (bool): True if the message is from the user, False if from the assistant.
        """
        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        bubble.set_margin_top(5)
        bubble.set_margin_bottom(5)

        sender_label = Gtk.Label(label=sender)
        sender_label.set_halign(Gtk.Align.START)
        bubble.append(sender_label)

        message_label = Gtk.Label(label=message)
        message_label.set_wrap(True)
        message_label.set_max_width_chars(32)
        message_label.set_halign(Gtk.Align.START)

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

        bubble.append(bubble_box)
        self.chat_history.append(bubble)
        # Schedule scrolling to the bottom after a short delay.
        GLib.timeout_add(100, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        """
        Scrolls the chat history scrolled window to the bottom.
        This is scheduled via GLib.timeout_add to allow the layout to update.
        """
        vadjustment = self.chat_history_scrolled.get_vadjustment()
        vadjustment.set_value(vadjustment.get_upper())
        return False  # Stop the timeout

    def update_status_bar(self, provider=None, model=None):
        """
        Updates the status label with the current provider and model information.

        Args:
            provider (str, optional): Provider name; if None, retrieved from ATLAS.
            model (str, optional): Model name; if None, retrieved from ATLAS.
        """
        provider = provider or self.ATLAS.provider_manager.get_current_provider()
        model = model or self.ATLAS.provider_manager.get_current_model() or "No model selected"
        status_message = f"Provider: {provider} | Model: {model}"
        self.status_label.set_text(status_message)
