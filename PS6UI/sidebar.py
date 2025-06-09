# PS6UI/sidebar.py

import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QFrame, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt

from PS6UI.Chat.chat_page import ChatPage
from PS6UI.Persona_manager.persona_management import PersonaManagement
from PS6UI.Provider_manager.provider_management import ProviderManagement
from PS6UI.Settings.Speech.speech_settings import SpeechSettingsWindow

class Sidebar(QMainWindow):
    def __init__(self, atlas):
        super().__init__()
        self.setWindowTitle("Sidebar")
        self.ATLAS = atlas
        self.persona_management = PersonaManagement(self.ATLAS, self)
        self.provider_management = ProviderManagement(self.ATLAS, self)

        # Set default size and remove window decorations
        self.resize(50, 600)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Apply basic styling (equivalent to apply_css())
        # You can load from a .qss file if desired.
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #4a90d9;
            }
            QPushButton:pressed {
                background-color: #357ABD;
            }
        """)

        # Create the central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.box = QVBoxLayout()
        self.box.setSpacing(5)
        self.box.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(self.box)

        # Add icons (buttons)
        self.create_icon("Icons/providers.png", self.show_provider_menu, 42)
        self.create_icon("Icons/history.png", self.handle_history_button, 42)
        self.create_icon("Icons/chat.png", self.show_chat_page, 42)
        self.create_icon("Icons/speech.png", self.show_speech_settings, 42)
        self.create_icon("Icons/agent.png", self.show_persona_menu, 42)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #2b2b2b;")
        separator.setLineWidth(1)
        self.box.addWidget(separator)

        # Bottom icons
        self.create_icon("Icons/settings.png", self.show_settings_page, 42)
        self.create_icon("Icons/power_button.png", self.close_application, 42)

    def create_icon(self, icon_path, callback, icon_size):
        full_icon_path = os.path.join(os.path.dirname(__file__), "..", icon_path)
        full_icon_path = os.path.abspath(full_icon_path)

        button = QPushButton()
        button.setFixedSize(icon_size, icon_size)

        if os.path.exists(full_icon_path):
            pixmap = QPixmap(full_icon_path)
            icon = QIcon(pixmap)
            button.setIcon(icon)
            # Scale the icon to fit the button
            button.setIconSize(pixmap.rect().size().scaled(icon_size, icon_size, Qt.KeepAspectRatio))
        else:
            # Fallback if icon not found
            button.setText("X")

        button.clicked.connect(callback)
        self.box.addWidget(button)

    def show_provider_menu(self):
        if self.ATLAS.is_initialized():
            self.provider_management.show_provider_menu()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def handle_history_button(self):
        if self.ATLAS.is_initialized():
            self.ATLAS.log_history()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_chat_page(self):
        if self.ATLAS.is_initialized():
            chat_page = ChatPage(self.ATLAS)
            self.ATLAS.chat_page = chat_page  # Store reference to chat_page
            chat_page.show()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_persona_menu(self):
        if self.ATLAS.is_initialized():
            self.persona_management.show_persona_menu()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_speech_settings(self):
        if self.ATLAS.is_initialized():
            dlg = SpeechSettingsWindow(self.ATLAS, self)
            dlg.exec()
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def show_settings_page(self):
        if self.ATLAS.is_initialized():
            # Replace with your settings window as needed
            QMessageBox.information(self, "Settings", "Settings Page Content Here")
        else:
            self.show_error_dialog("ATLAS is not fully initialized. Please try again later.")

    def close_application(self):
        print("Closing application")
        self.close()

    def show_error_dialog(self, message):
        QMessageBox.critical(self, "Initialization Error", message)

    def set_application(self, app):
        # Not typically needed in Qt, but you can store a reference if required.
        self.app = app
