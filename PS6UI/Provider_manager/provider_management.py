# PS6UI/Provider_manager/provider_management.py

import os
import logging
import asyncio
import threading
from functools import partial

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QIcon

from modules.Providers.HuggingFace.HF_gen_response import HuggingFaceGenerator
from PS6UI.Provider_manager.Settings.HF_settings import HuggingFaceSettingsWindow

class ProviderManagement:
    """
    Manages provider-related functionalities, including displaying available
    providers, handling provider selection, and managing provider settings.
    """

    def __init__(self, ATLAS, parent_window):
        """
        Initializes the ProviderManagement with the given ATLAS instance and parent window.

        Args:
            ATLAS (ATLAS): The main ATLAS instance.
            parent_window (QWidget): The parent window.
        """
        self.ATLAS = ATLAS
        self.parent_window = parent_window
        self.provider_window = None
        self.config_manager = self.ATLAS.config_manager
        self.logger = logging.getLogger(__name__)

    def show_provider_menu(self):
        """
        Displays the provider selection window, listing all available providers.
        Each provider has a label and a settings icon.
        """
        self.provider_window = QDialog(self.parent_window)
        self.provider_window.setWindowTitle("Select Provider")
        self.provider_window.resize(300, 400)
        self.provider_window.setModal(True)

        layout = QVBoxLayout(self.provider_window)

        provider_names = self.ATLAS.get_available_providers()
        for provider_name in provider_names:
            hbox = QHBoxLayout()

            label = QLabel(provider_name)
            label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            # Define a closure for label click event so each label selects the correct provider
            def label_mouse_release(event, p=provider_name):
                self.select_provider(p)
            label.mouseReleaseEvent = label_mouse_release

            settings_icon_path = os.path.join(os.path.dirname(__file__), "../../Icons/settings.png")
            settings_icon_path = os.path.abspath(settings_icon_path)

            settings_button = QPushButton()
            settings_button.setFixedSize(16,16)
            if os.path.exists(settings_icon_path):
                pixmap = QPixmap(settings_icon_path).scaled(16,16, Qt.KeepAspectRatio)
                settings_button.setIcon(QIcon(pixmap))
            else:
                settings_button.setText("S")

            # Use partial to ensure the correct provider_name is used
            settings_button.clicked.connect(partial(self.open_provider_settings, provider_name))

            hbox.addWidget(label)
            hbox.addWidget(settings_button, alignment=Qt.AlignRight)
            layout.addLayout(hbox)

        self.provider_window.exec()

    def select_provider(self, provider: str):
        """
        Handles the selection of a provider. Sets the current provider in ATLAS.
        If the provider's API key is not set, prompts the user to enter it.
        """
        api_key = self.config_manager.get_config(f"{provider.upper()}_API_KEY")
        if not api_key:
            self.logger.info(f"No API key set for provider {provider}. Prompting user to enter it.")
            self.open_provider_settings(provider)
        else:
            threading.Thread(target=self.set_current_provider_thread, args=(provider,), daemon=True).start()
            self.logger.info(f"Provider {provider} selected.")
            if self.provider_window:
                self.provider_window.close()

    def set_current_provider_thread(self, provider):
        """
        Thread target to set the current provider.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.ATLAS.set_current_provider(provider))
            self.logger.info(f"Provider {provider} set successfully.")
        except Exception as e:
            self.logger.error(f"Failed to set provider {provider}: {e}")
            QTimer.singleShot(0, lambda: self.show_error_dialog(f"Failed to set provider {provider}: {e}"))
        finally:
            loop.close()

    def open_provider_settings(self, provider_name: str):
        """
        Opens the appropriate settings dialog for the given provider.
        If HuggingFace, opens the HuggingFaceSettingsWindow.
        Otherwise, opens a generic provider settings dialog for API key entry.
        """
        # Do not close the provider_window here. Let the provider menu remain open or be closed by user if needed.
        # This ensures the dialog does not immediately close when we open the settings.

        if provider_name == "HuggingFace":
            if self.ATLAS.provider_manager.huggingface_generator is None:
                self.logger.info("Initializing huggingface_generator in open_provider_settings")
                self.ATLAS.provider_manager.huggingface_generator = HuggingFaceGenerator(self.config_manager)
                self.logger.info("huggingface_generator initialized successfully")
            else:
                self.logger.info("huggingface_generator already initialized")
            self.show_huggingface_settings()
        else:
            self.show_provider_settings(provider_name)

    def show_huggingface_settings(self):
        """
        Displays the HuggingFace settings window.
        """
        settings_window = HuggingFaceSettingsWindow(self.ATLAS, self.config_manager, self.parent_window)
        # Display the dialog modally
        settings_window.exec()

    def show_provider_settings(self, provider_name: str):
        """
        Displays the settings window for a specific provider, including the API key entry.
        """
        dialog = QDialog(self.parent_window)
        dialog.setWindowTitle(f"Settings for {provider_name}")
        dialog.resize(400, 300)
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)

        provider_label = QLabel("Provider:")
        provider_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(provider_label)

        provider_value = QLabel(provider_name)
        provider_value.setAlignment(Qt.AlignLeft)
        layout.addWidget(provider_value)

        api_key_label = QLabel("API Key:")
        api_key_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(api_key_label)

        self.api_key_entry = QLineEdit()
        self.api_key_entry.setPlaceholderText("Enter your API key here")
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.api_key_entry)

        existing_api_key = self.get_existing_api_key(provider_name)
        if existing_api_key:
            self.api_key_entry.setText(existing_api_key)

        # Use partial to ensure the callback has correct provider_name and dialog
        save_button = QPushButton("Save")
        save_button.clicked.connect(partial(self.on_save_button_clicked, provider_name, dialog))
        layout.addWidget(save_button)

        dialog.exec()

    def get_existing_api_key(self, provider_name: str) -> str:
        """
        Retrieves the existing API key for the given provider from ConfigManager.
        """
        api_key_methods = {
            "OpenAI": self.config_manager.get_openai_api_key,
            "Mistral": self.config_manager.get_mistral_api_key,
            "Google": self.config_manager.get_google_api_key,
            "HuggingFace": self.config_manager.get_huggingface_api_key,
            "Anthropic": self.config_manager.get_anthropic_api_key,
            "Grok": self.config_manager.get_grok_api_key,
        }

        get_key_func = api_key_methods.get(provider_name)
        if get_key_func:
            return get_key_func() or ""
        return ""

    def on_save_button_clicked(self, provider_name: str, dialog: QDialog):
        """
        Handles the Save button click in the provider settings dialog.
        Validates and updates the API key for the provider.
        """
        new_api_key = self.api_key_entry.text().strip()
        if not new_api_key:
            self.show_error_dialog("API Key cannot be empty.")
            return

        # Update API key in a thread
        threading.Thread(target=self.update_api_key_thread, args=(provider_name, new_api_key, dialog), daemon=True).start()

    def update_api_key_thread(self, provider_name: str, new_api_key: str, dialog: QDialog):
        """
        Thread target to update the API key and refresh the provider asynchronously.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            self.config_manager.update_api_key(provider_name, new_api_key)
            self.logger.info(f"API Key for {provider_name} updated.")
            loop.run_until_complete(self.refresh_provider_async(provider_name))
            self.logger.info(f"Provider {provider_name} refreshed.")
            QTimer.singleShot(0, lambda: self.show_info_dialog(f"API Key for {provider_name} saved successfully."))
            QTimer.singleShot(0, dialog.close)
        except Exception as e:
            self.logger.error(f"Failed to save API Key: {str(e)}")
            QTimer.singleShot(0, lambda: self.show_error_dialog(f"Failed to save API Key: {str(e)}"))
        finally:
            loop.close()

    async def refresh_provider_async(self, provider_name: str):
        """
        Refreshes the current provider if it matches the given provider_name.
        """
        if provider_name == self.ATLAS.provider_manager.current_llm_provider:
            try:
                await self.ATLAS.provider_manager.set_current_provider(provider_name)
                self.logger.info(f"Provider {provider_name} refreshed with new API key.")
            except Exception as e:
                self.logger.error(f"Error refreshing provider {provider_name}: {e}")
                QTimer.singleShot(0, lambda: self.show_error_dialog(f"Error refreshing provider {provider_name}: {e}"))

    def show_error_dialog(self, message: str):
        """
        Displays an error dialog with the given message.
        """
        QMessageBox.critical(self.parent_window, "Error", message)

    def show_info_dialog(self, message: str):
        """
        Displays an info dialog with the given message.
        """
        QMessageBox.information(self.parent_window, "Information", message)
