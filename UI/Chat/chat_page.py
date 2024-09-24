# UI/Chat/chat_page.py

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, Gdk, GLib

import asyncio

class ChatPage(Gtk.Window):
    def __init__(self, atlas):
        super().__init__(title="Chat Page")
        self.ATLAS = atlas
        self.chat_session = atlas.chat_session  # Access the ChatSession
        self.set_default_size(600, 400)
        self.set_keep_above(False)
        self.stick()

        self.apply_css_styling()

        # Position window next to sidebar
        self.position_window_next_to_sidebar(self, 600)  # Assuming window width is 600

        # Main vertical box
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        self.add(self.vbox)

        # Top label
        self.chat_label = Gtk.Label(label="Chat")
        self.chat_label.set_xalign(0.0)
        self.chat_label.set_margin_top(10)
        self.chat_label.set_margin_start(10)
        self.chat_label.set_margin_bottom(5)
        self.vbox.pack_start(self.chat_label, False, False, 0)

        # Separator
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        self.vbox.pack_start(separator, False, False, 5)

        # Chat history scrolled window
        self.chat_history_scrolled = Gtk.ScrolledWindow()
        self.chat_history_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.chat_history = Gtk.TextView()
        self.chat_history.set_editable(False)
        self.chat_history.set_wrap_mode(Gtk.WrapMode.WORD)
        self.chat_history_buffer = self.chat_history.get_buffer()
        self.chat_history_scrolled.add(self.chat_history)
        self.vbox.pack_start(self.chat_history_scrolled, True, True, 0)

        # Input entry and send button in an hbox
        input_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        self.input_entry = Gtk.Entry()
        self.input_entry.set_placeholder_text("Type your message here...")
        self.input_entry.connect("activate", self.on_enter_pressed)
        input_hbox.pack_start(self.input_entry, True, True, 0)

        self.send_button = Gtk.Button(label="Send")
        self.send_button.connect("clicked", self.on_send_button_clicked)
        input_hbox.pack_start(self.send_button, False, False, 0)

        self.vbox.pack_start(input_hbox, False, False, 5)

        # Status bar
        self.status_bar = Gtk.Statusbar()
        context_id = self.status_bar.get_context_id("Status")
        provider = self.ATLAS.provider_manager.get_current_provider()
        model = self.ATLAS.provider_manager.get_current_model() or "No model selected"
        status_message = f"Provider: {provider} | Model: {model}"
        self.status_bar.push(context_id, status_message)
        self.vbox.pack_end(self.status_bar, False, True, 0)

        self.show_all()

    def on_enter_pressed(self, widget):
        self.send_message()

    def on_send_button_clicked(self, widget):
        self.send_message()

    def send_message(self):
        message = self.input_entry.get_text().strip()
        if message:
            # Append user's message to chat history
            self.append_to_chat_history(f"You: {message}\n")

            # Clear input entry
            self.input_entry.set_text("")

            # Schedule the asynchronous response handling
            asyncio.ensure_future(self.handle_model_response(message))

    async def handle_model_response(self, message):
        try:
            response = await self.chat_session.send_message(message)
            # Schedule UI update
            GLib.idle_add(self.append_to_chat_history, f"Assistant: {response}\n")
        except Exception as e:
            self.ATLAS.logger.error(f"Error in handle_model_response: {e}")
            GLib.idle_add(self.append_to_chat_history, f"Assistant: Error: {e}\n")


    def append_to_chat_history(self, text):
        end_iter = self.chat_history_buffer.get_end_iter()
        self.chat_history_buffer.insert(end_iter, text)
        self.chat_history.scroll_to_mark(self.chat_history_buffer.get_insert(), 0.0, True, 0.0, 1.0)

    def position_window_next_to_sidebar(self, window: Gtk.Window, window_width: int):
        """
        Positions the given window next to the sidebar.

        Args:
            window (Gtk.Window): The window to position.
            window_width (int): The width of the window.
        """
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
                font-size: 18px;
                font-weight: bold;
            }
            textview {
                background-color: #3c3c3c;
                color: white;
                padding: 10px;
            }
            entry {
                background-color: #3c3c3c;
                color: white;
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
