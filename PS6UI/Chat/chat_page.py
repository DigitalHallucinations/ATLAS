# PS6UI/Chat/chat_page.py

import os
import asyncio
import threading

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QIcon

class ChatPage(QMainWindow):
    # Signal to update chat from background thread
    message_received = Signal(str, str, bool)  # sender, message, is_user

    def __init__(self, atlas):
        super().__init__()
        self.ATLAS = atlas
        self.chat_session = atlas.chat_session

        self.resize(600, 400)
        self.set_chat_page_style()

        # Main central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Update the window title with the persona's name
        self.update_persona_label()

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setContentsMargins(5, 5, 5, 5)
        self.main_layout.addWidget(separator)

        # Chat history (QScrollArea)
        self.chat_history_container = QWidget()
        self.chat_history_layout = QVBoxLayout(self.chat_history_container)
        self.chat_history_layout.setSpacing(10)
        self.chat_history_layout.setContentsMargins(10, 0, 10, 0)

        self.chat_history_scrolled = QScrollArea()
        self.chat_history_scrolled.setWidgetResizable(True)
        self.chat_history_scrolled.setWidget(self.chat_history_container)
        self.main_layout.addWidget(self.chat_history_scrolled)

        # Input area
        input_hbox = QHBoxLayout()
        input_hbox.setSpacing(5)
        input_hbox.setContentsMargins(10, 10, 10, 10)

        self.input_entry = QLineEdit()
        self.input_entry.setPlaceholderText("Type a message...")
        self.input_entry.returnPressed.connect(self.on_send_message)
        input_hbox.addWidget(self.input_entry)

        send_button = QPushButton()
        send_button.setFixedSize(32, 32)

        # Load send icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "../../Icons/send.png")
            icon_path = os.path.abspath(icon_path)
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                icon = QIcon(pixmap)
                send_button.setIcon(icon)
                send_button.setIconSize(pixmap.size().scaled(24, 24, Qt.KeepAspectRatio))
            else:
                send_button.setText("Send")
        except Exception as e:
            print(f"Error loading icon: {e}")
            send_button.setText("Send")

        send_button.clicked.connect(self.on_send_message)
        input_hbox.addWidget(send_button)
        self.main_layout.addLayout(input_hbox)

        # Status label
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignLeft)
        self.status_label.setContentsMargins(5, 0, 5, 5)
        self.main_layout.addWidget(self.status_label)
        self.update_status_bar()

        # Connect the signal for provider changes
        self.ATLAS.notify_provider_changed = self.update_status_bar

        # Connect the message_received signal to the add_message_bubble method
        self.message_received.connect(self.add_message_bubble)

        # Show the window
        self.show()

    def set_chat_page_style(self):
        # set a styleSheet for the chat page
        # Example:
        # self.setStyleSheet("background-color: #2b2b2b; color: white;")
        pass

    def update_persona_label(self):
        """
        Updates the window title with the current persona's name.
        """
        persona_name = self.ATLAS.persona_manager.current_persona.get('name', 'Chat')
        self.setWindowTitle(persona_name)

    def on_send_message(self):
        """
        Handles the event when a message is sent.
        """
        message = self.input_entry.text().strip()
        if message:
            user_name = self.ATLAS.user
            self.add_message_bubble(user_name, message, True)
            self.input_entry.clear()
            # Run model response in a background thread
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
            # Emit signal to update UI in main thread
            self.message_received.emit(persona_name, response, False)
        except Exception as e:
            self.ATLAS.logger.error(f"Error in handle_model_response: {e}")
            self.message_received.emit("Assistant", f"Error: {e}", False)

    @Slot(str, str, bool)
    def add_message_bubble(self, sender, message, is_user):
        """
        Adds a message bubble to the chat history.

        Args:
            sender (str): The sender of the message.
            message (str): The message content.
            is_user (bool): Whether the message is from the user.
        """
        bubble_widget = QWidget()
        bubble_layout = QVBoxLayout(bubble_widget)
        bubble_layout.setSpacing(5)
        bubble_layout.setContentsMargins(0, 5, 0, 5)

        sender_label = QLabel(sender)
        sender_label.setAlignment(Qt.AlignLeft)
        bubble_layout.addWidget(sender_label)

        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setFixedWidth(250)  
        bubble_box = QWidget()
        bubble_box_layout = QHBoxLayout(bubble_box)
        bubble_box_layout.setContentsMargins(0, 0, 0, 0)
        bubble_box_layout.addWidget(message_label)

        # Apply styles depending on user or assistant
        if is_user:
            bubble_box_layout.setAlignment(Qt.AlignRight)
            # Add style for user message here
            # bubble_box.setStyleSheet("background-color: #...;")
            bubble_layout.setAlignment(Qt.AlignRight)
        else:
            bubble_box_layout.setAlignment(Qt.AlignLeft)
            # Add style for assistant message here
            bubble_layout.setAlignment(Qt.AlignLeft)

        bubble_layout.addWidget(bubble_box)
        self.chat_history_layout.addWidget(bubble_widget)

        # Scroll to bottom
        self.chat_history_scrolled.verticalScrollBar().setValue(
            self.chat_history_scrolled.verticalScrollBar().maximum()
        )

    def update_status_bar(self, provider=None, model=None):
        """
        Updates the status label with the current provider and model.
        """
        provider = provider or self.ATLAS.provider_manager.get_current_provider()
        model = model or self.ATLAS.provider_manager.get_current_model() or "No model selected"
        status_message = f"Provider: {provider} | Model: {model}"
        self.status_label.setText(status_message)
