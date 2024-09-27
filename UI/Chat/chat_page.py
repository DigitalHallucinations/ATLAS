# UI/Chat/chat_page.py

import asyncio
import threading
from gi.repository import Gtk, GLib, Gdk, Pango

class ChatPage(Gtk.Window):
    def __init__(self, atlas):
        super().__init__(title="Chat Page")
        self.ATLAS = atlas
        self.chat_session = atlas.chat_session
        self.set_default_size(600, 400)
        self.set_keep_above(False)
        self.stick()

        self.apply_css_styling()

        # Position window next to sidebar
        self.position_window_next_to_sidebar(self, 600)  # Assuming window width is 600

        # Main vertical box
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(self.vbox)

        # Top label with current persona name
        self.persona_label = Gtk.Label()
        self.persona_label.set_name("persona-label")
        self.update_persona_label()
        self.persona_label.set_xalign(0.0)
        self.persona_label.set_margin_top(10)
        self.persona_label.set_margin_start(10)
        self.persona_label.set_margin_bottom(5)
        self.vbox.pack_start(self.persona_label, False, False, 0)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self.vbox.pack_start(separator, False, False, 5)

        # Chat history scrolled window
        self.chat_history_scrolled = Gtk.ScrolledWindow()
        self.chat_history_scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.chat_history = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.chat_history.set_margin_start(10)
        self.chat_history.set_margin_end(10)
        self.chat_history_scrolled.add(self.chat_history)
        self.vbox.pack_start(self.chat_history_scrolled, True, True, 0)

        # Input area
        input_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        input_box.set_margin_top(10)
        input_box.set_margin_bottom(10)
        input_box.set_margin_start(10)
        input_box.set_margin_end(10)

        self.input_entry = Gtk.Entry()
        self.input_entry.set_placeholder_text("Type a message...")
        self.input_entry.connect("activate", self.on_send_message)
        input_box.pack_start(self.input_entry, True, True, 0)

        send_button = Gtk.Button.new_from_icon_name("send", Gtk.IconSize.BUTTON)
        send_button.connect("clicked", self.on_send_message)
        input_box.pack_start(send_button, False, False, 0)

        self.vbox.pack_start(input_box, False, False, 0)

        # Status bar
        self.status_bar = Gtk.Statusbar()
        context_id = self.status_bar.get_context_id("Status")
        provider = self.ATLAS.provider_manager.get_current_provider()
        model = self.ATLAS.provider_manager.get_current_model() or "No model selected"
        status_message = f"Provider: {provider} | Model: {model}"
        self.status_bar.push(context_id, status_message)
        self.vbox.pack_end(self.status_bar, False, False, 0)

        self.show_all()

    def update_persona_label(self):
        persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Chat')
        self.persona_label.set_text(persona_name)

    def on_send_message(self, widget):
        message = self.input_entry.get_text().strip()
        if message:
            user_name = self.ATLAS.user
            self.add_message_bubble(user_name, message, is_user=True)
            self.input_entry.set_text("")
            threading.Thread(target=self.handle_model_response_thread, args=(message,), daemon=True).start()

    def handle_model_response_thread(self, message):
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
        bubble = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        bubble.set_margin_top(5)
        bubble.set_margin_bottom(5)

        sender_label = Gtk.Label(label=sender)
        sender_label.set_halign(Gtk.Align.START)
        bubble.pack_start(sender_label, False, False, 0)

        message_label = Gtk.Label(label=message)
        message_label.set_line_wrap(True)
        message_label.set_max_width_chars(32)
        message_label.set_justify(Gtk.Justification.LEFT)
        message_label.set_halign(Gtk.Align.START)

        bubble_box = Gtk.EventBox()
        bubble_box.add(message_label)
        bubble_box.get_style_context().add_class("message-bubble")
        
        if is_user:
            bubble_box.get_style_context().add_class("user-message")
            bubble.set_halign(Gtk.Align.END)
        else:
            bubble_box.get_style_context().add_class("assistant-message")
            bubble.set_halign(Gtk.Align.START)

        bubble.pack_start(bubble_box, False, False, 0)
        self.chat_history.pack_start(bubble, False, False, 0)
        self.chat_history.show_all()

        # Scroll to the bottom
        adj = self.chat_history_scrolled.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    def position_window_next_to_sidebar(self, window: Gtk.Window, window_width: int):
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor()
        monitor_geometry = monitor.get_geometry()
        screen_width = monitor_geometry.width

        window_height = monitor_geometry.height

        window_x = screen_width - 50 - 10 - window_width
        window.set_default_size(
            window_width,
            window_height if window_height < monitor_geometry.height else monitor_geometry.height - 50
        )

        window.move(window_x, 0)

    def apply_css_styling(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            window {
                background-color: #2b2b2b;
            }
            label {
                color: white;
                font-size: 14px;
            }
            #persona-label {
                font-size: 18px;
                font-weight: bold;
            }
            .message-bubble { 
                border-radius: 18px; 
                padding: 8px 12px;
            }
            .user-message { 
                background-color: #0084ff; 
                color: white;
            }
            .assistant-message { 
                background-color: #3c3c3c; 
                color: white;
            }
            entry {
                background-color: #3c3c3c;
                color: white;
                border-radius: 20px;
                padding: 8px 12px;
                caret-color: white;
            }
            button {
                background-color: #555555;
                color: white;
            }
            statusbar {
                background-color: #2b2b2b;
                color: white;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )