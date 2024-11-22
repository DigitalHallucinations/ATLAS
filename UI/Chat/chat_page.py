# UI/Chat/chat_page.py

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib

import os
import asyncio
import threading

from UI.Utils.utils import apply_css


class ChatPage(Gtk.Window):
    def __init__(self, atlas):
        super().__init__()
        self.ATLAS = atlas
        self.chat_session = atlas.chat_session
        self.set_default_size(600, 400)

        # Assign a CSS class to the window for targeted styling
        self.get_style_context().add_class("chat-page")

        # Apply centralized CSS
        apply_css()

        # Main vertical box
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_child(self.vbox)  # Use set_child() in GTK 4

        # Update the window title with the current persona's name
        self.update_persona_label()

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(5)
        self.vbox.append(separator)

        # Chat history scrolled window
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

        # Input area
        input_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        input_box.set_margin_top(10)
        input_box.set_margin_bottom(10)
        input_box.set_margin_start(10)
        input_box.set_margin_end(10)

        self.input_entry = Gtk.Entry()
        self.input_entry.set_placeholder_text("Type a message...")
        self.input_entry.connect("activate", self.on_send_message)
        self.input_entry.set_hexpand(True)
        input_box.append(self.input_entry)

        send_button = Gtk.Button()
        try:
            # Construct the icon path relative to this file
            icon_path = os.path.join(os.path.dirname(__file__), "../../Icons/send.png")
            icon_path = os.path.abspath(icon_path)
            # Load the send icon using Gdk.Texture
            texture = Gdk.Texture.new_from_filename(icon_path)
            # Create a Gtk.Picture for the icon
            icon = Gtk.Picture.new_for_paintable(texture)
            icon.set_size_request(24, 24)
            icon.set_content_fit(Gtk.ContentFit.CONTAIN)
        except GLib.Error as e:
            print(f"Error loading icon: {e}")
            icon = Gtk.Image.new_from_icon_name("image-missing")  # Fallback icon

        send_button.set_child(icon)
        send_button.get_style_context().add_class("send-button")
        send_button.connect("clicked", self.on_send_message)
        input_box.append(send_button)

        self.vbox.append(input_box)

        # Status label (replacing status bar)
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_margin_start(5)
        self.status_label.set_margin_end(5)
        self.vbox.append(self.status_label)
        self.update_status_bar()

        # Link status bar updates to provider changes
        self.ATLAS.notify_provider_changed = self.update_status_bar

        self.present()  # Use present() instead of show_all()

    def update_persona_label(self):
        """
        Updates the window title with the current persona's name.
        """
        persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Chat')
        self.set_title(persona_name)  # Set the window title

    def on_send_message(self, widget):
        """
        Handles the event when a message is sent.
        """
        message = self.input_entry.get_text().strip()
        if message:
            user_name = self.ATLAS.user
            self.add_message_bubble(user_name, message, is_user=True)
            self.input_entry.set_text("")
            threading.Thread(
                target=self.handle_model_response_thread,
                args=(message,),
                daemon=True
            ).start()

    def handle_model_response_thread(self, message):
        """
        Handles the model response in a separate thread to avoid blocking the UI.
        """
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.chat_session.send_message(message))
            loop.close()

            persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Assistant')
            GLib.idle_add(self.add_message_bubble, persona_name, response)
        except Exception as e:
            self.ATLAS.logger.error(f"Error in handle_model_response: {e}")
            GLib.idle_add(self.add_message_bubble, "Assistant", f"Error: {e}")

    def add_message_bubble(self, sender, message, is_user=False):
        """
        Adds a message bubble to the chat history.

        Args:
            sender (str): The sender of the message.
            message (str): The message content.
            is_user (bool): Whether the message is from the user.
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

        if is_user:
            bubble_box.get_style_context().add_class("user-message")
            bubble.set_halign(Gtk.Align.END)
        else:
            bubble_box.get_style_context().add_class("assistant-message")
            bubble.set_halign(Gtk.Align.START)

        bubble.append(bubble_box)
        self.chat_history.append(bubble)

        # Scroll to the bottom of the chat history
        vadjustment = self.chat_history_scrolled.get_vadjustment()
        vadjustment.set_value(vadjustment.get_upper())

    def update_status_bar(self, provider=None, model=None):
        """
        Updates the status label with the current provider and model.

        Args:
            provider (str, optional): The provider to display.
            model (str, optional): The model to display.
        """
        provider = provider or self.ATLAS.provider_manager.get_current_provider()
        model = model or self.ATLAS.provider_manager.get_current_model() or "No model selected"
        status_message = f"Provider: {provider} | Model: {model}"
        self.status_label.set_text(status_message)
