# GTKUI/Settings/Speech/speech_settings.py

"""
Module: speech_settings.py
Description:
    Enterprise production–ready Speech Settings window.
    Allows configuration of Text-to-Speech (TTS) and Speech-to-Text (STT) settings.
    In addition, provides input fields to configure API keys for speech providers as well as
    Google Cloud credentials. If an API key is not provided, the corresponding provider is disabled.
    
Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib
import os
import logging
from GTKUI.Utils.utils import apply_css

logger = logging.getLogger(__name__)

class SpeechSettings(Gtk.Window):
    def __init__(self, atlas):
        """
        Initializes the Speech Settings window.
        
        Args:
            atlas: The main ATLAS application instance.
        """
        super().__init__(title="Speech Settings")
        self.ATLAS = atlas
        self.set_default_size(400, 450)
        apply_css()

        # Main container.
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        self.set_child(vbox)

        # --- TTS Settings Section ---
        tts_frame = Gtk.Frame(label="Text-to-Speech (TTS) Settings")
        tts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        tts_frame.set_child(tts_box)

        # Voice selection combo box.
        self.voice_combo = Gtk.ComboBoxText()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        voices = self.ATLAS.speech_manager.get_tts_voices(current_tts_provider)
        for voice in voices:
            self.voice_combo.append_text(voice.get('name', 'Unknown'))
        if voices:
            self.voice_combo.set_active(0)
        tts_box.append(Gtk.Label(label="Select TTS Voice:"))
        tts_box.append(self.voice_combo)

        # TTS enable/disable switch.
        self.tts_switch = Gtk.Switch()
        self.tts_switch.set_active(self.ATLAS.speech_manager.get_tts_status(current_tts_provider))
        tts_box.append(Gtk.Label(label="Enable TTS:"))
        tts_box.append(self.tts_switch)

        # API key input for Eleven Labs TTS.
        self.eleven_api_entry = Gtk.Entry()
        existing_xi_api = self.ATLAS.config_manager.get_config("XI_API_KEY") or ""
        self.eleven_api_entry.set_text(existing_xi_api)
        tts_box.append(Gtk.Label(label="Eleven Labs API Key (optional):"))
        tts_box.append(self.eleven_api_entry)

        # --- STT Settings Section ---
        stt_frame = Gtk.Frame(label="Speech-to-Text (STT) Settings")
        stt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        stt_frame.set_child(stt_box)

        # STT provider selection combo box.
        self.stt_combo = Gtk.ComboBoxText()
        for key in self.ATLAS.speech_manager.stt_services.keys():
            self.stt_combo.append_text(key)
        self.stt_combo.set_active(0)
        stt_box.append(Gtk.Label(label="Select STT Provider:"))
        stt_box.append(self.stt_combo)

        # Whisper mode selection combo box (only applicable if a Whisper provider is selected).
        self.whisper_mode_combo = Gtk.ComboBoxText()
        self.whisper_mode_combo.append_text("Local")
        self.whisper_mode_combo.append_text("Online")
        self.whisper_mode_combo.set_active(0)
        stt_box.append(Gtk.Label(label="Whisper Mode (if applicable):"))
        stt_box.append(self.whisper_mode_combo)

        # API key input for OpenAI (used by Whisper online mode).
        self.openai_api_entry = Gtk.Entry()
        existing_openai_api = self.ATLAS.config_manager.get_config("OPENAI_API_KEY") or ""
        self.openai_api_entry.set_text(existing_openai_api)
        stt_box.append(Gtk.Label(label="OpenAI API Key for Whisper Online (optional):"))
        stt_box.append(self.openai_api_entry)
        
        # --- Google Cloud Credentials Section ---
        # This field allows setting the path to the Google Cloud credentials JSON file.
        google_frame = Gtk.Frame(label="Google Cloud Credentials")
        google_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        google_frame.set_child(google_box)
        
        self.google_credentials_entry = Gtk.Entry()
        existing_google_creds = self.ATLAS.config_manager.get_config("GOOGLE_APPLICATION_CREDENTIALS") or ""
        self.google_credentials_entry.set_text(existing_google_creds)
        google_box.append(Gtk.Label(label="Google Credentials JSON Path (optional):"))
        google_box.append(self.google_credentials_entry)
        
        # Save button to apply the settings.
        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)

        # Pack sections into the main container.
        vbox.append(tts_frame)
        vbox.append(stt_frame)
        vbox.append(google_frame)
        vbox.append(save_button)
        self.present()

    def on_save_clicked(self, widget):
        """
        Handler for the save button.
        Applies TTS and STT settings via the speech manager.
        Also saves API key configuration and Google credentials to the ConfigManager
        and updates the environment.
        """
        # Update API keys and credentials in the ConfigManager and process environment.
        eleven_api_key = self.eleven_api_entry.get_text().strip()
        openai_api_key = self.openai_api_entry.get_text().strip()
        google_creds = self.google_credentials_entry.get_text().strip()

        # Save values to config (assumes set method persists configuration)
        self.ATLAS.config_manager.set("XI_API_KEY", eleven_api_key)
        self.ATLAS.config_manager.set("OPENAI_API_KEY", openai_api_key)
        self.ATLAS.config_manager.set("GOOGLE_APPLICATION_CREDENTIALS", google_creds)

        # Update the environment so provider modules using os.getenv pick up the changes.
        os.environ["XI_API_KEY"] = eleven_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds

        # Apply TTS settings.
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        tts_enabled = self.tts_switch.get_active()
        self.ATLAS.speech_manager.set_tts_status(tts_enabled, current_tts_provider)
        selected_voice_name = self.voice_combo.get_active_text()
        voices = self.ATLAS.speech_manager.get_tts_voices(current_tts_provider)
        selected_voice = next((v for v in voices if v.get('name') == selected_voice_name), None)
        if selected_voice:
            self.ATLAS.speech_manager.set_tts_voice(selected_voice, current_tts_provider)

        # Apply STT settings.
        selected_stt_provider = self.stt_combo.get_active_text()
        if selected_stt_provider.startswith("whisper"):
            mode = self.whisper_mode_combo.get_active_text().lower()
            try:
                from modules.Speech_Services.whisper_stt import WhisperSTT
                whisper_stt = WhisperSTT(mode=mode)
                provider_key = f"whisper_{mode}"
                self.ATLAS.speech_manager.add_stt_provider(provider_key, whisper_stt)
                self.ATLAS.speech_manager.set_default_stt_provider(provider_key)
                logger.info(f"STT provider set to {provider_key}")
            except Exception as e:
                logger.error(f"Error setting Whisper mode: {e}")
        else:
            self.ATLAS.speech_manager.set_default_stt_provider(selected_stt_provider)
        self.destroy()

