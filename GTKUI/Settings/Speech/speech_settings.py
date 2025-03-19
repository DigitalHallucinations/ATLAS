# GTKUI/Settings/Speech/speech_settings.py

"""
Module: speech_settings.py
Description:
    Enterprise production–ready Speech Settings window with a tabbed layout.
    The General tab now lets the user turn TTS and STT on or off independently and select
    the default providers for each.
    All Google settings (credentials used for both Google TTS and STT) are on a single Google tab.
    Separate tabs exist for Eleven Labs TTS and Whisper STT.
    
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
        Initializes the Speech Settings window with a tabbed interface.

        Args:
            atlas: The main ATLAS application instance.
        """
        super().__init__(title="Speech Settings")
        self.ATLAS = atlas

        # Use the same dark style classes as the rest of the UI
        self.get_style_context().add_class("chat-page")
        self.get_style_context().add_class("sidebar")
        
        # Apply global CSS (style.css)
        apply_css()

        self.set_default_size(500, 500)

        # Main container
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        self.set_child(vbox)

        # Create a Notebook for tabbed settings
        self.notebook = Gtk.Notebook()
        vbox.append(self.notebook)

        # Create each tab/page
        self.create_general_tab()
        self.create_eleven_labs_tts_tab()
        self.create_google_tab()
        self.create_whisper_stt_tab()

        # "Save Settings" button at the bottom of the window
        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)
        vbox.append(save_button)

        self.present()

    # -------------------------------------------------------------------------
    #   TAB 1: GENERAL
    # -------------------------------------------------------------------------
    def create_general_tab(self):
        """
        Creates the "General" tab for overall speech-related settings.
        Contains separate switches to enable/disable TTS and STT,
        and selectors for the default TTS and STT providers.
        """
        general_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        general_box.set_margin_top(6)
        general_box.set_margin_bottom(6)
        general_box.set_margin_start(6)
        general_box.set_margin_end(6)

        # TTS Enable Switch
        tts_label = Gtk.Label(label="Enable TTS:")
        self.general_tts_switch = Gtk.Switch()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        self.general_tts_switch.set_active(self.ATLAS.speech_manager.get_tts_status(current_tts_provider))
        general_box.append(tts_label)
        general_box.append(self.general_tts_switch)

        # Default TTS Provider Selector
        tts_provider_label = Gtk.Label(label="Default TTS Provider:")
        self.default_tts_combo = Gtk.ComboBoxText()
        for key in self.ATLAS.speech_manager.tts_services.keys():
            self.default_tts_combo.append_text(key)
        # Set current active default provider if available.
        if current_tts_provider:
            index = list(self.ATLAS.speech_manager.tts_services.keys()).index(current_tts_provider)
            self.default_tts_combo.set_active(index)
        general_box.append(tts_provider_label)
        general_box.append(self.default_tts_combo)

        # STT Enable Switch
        stt_label = Gtk.Label(label="Enable STT:")
        self.general_stt_switch = Gtk.Switch()
        default_stt = self.ATLAS.speech_manager.get_default_stt_provider()
        stt_enabled = True if default_stt else False
        self.general_stt_switch.set_active(stt_enabled)
        general_box.append(stt_label)
        general_box.append(self.general_stt_switch)

        # Default STT Provider Selector
        stt_provider_label = Gtk.Label(label="Default STT Provider:")
        self.default_stt_combo = Gtk.ComboBoxText()
        for key in self.ATLAS.speech_manager.stt_services.keys():
            self.default_stt_combo.append_text(key)
        if default_stt:
            index = list(self.ATLAS.speech_manager.stt_services.keys()).index(default_stt)
            self.default_stt_combo.set_active(index)
        general_box.append(stt_provider_label)
        general_box.append(self.default_stt_combo)

        # Add the "General" page to the notebook
        self.notebook.append_page(general_box, Gtk.Label(label="General"))

    # -------------------------------------------------------------------------
    #   TAB 2: ELEVEN LABS TTS
    # -------------------------------------------------------------------------
    def create_eleven_labs_tts_tab(self):
        """
        Creates a tab specifically for Eleven Labs TTS settings:
        - Voice selection
        - Eleven Labs API key
        """
        eleven_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        eleven_box.set_margin_top(6)
        eleven_box.set_margin_bottom(6)
        eleven_box.set_margin_start(6)
        eleven_box.set_margin_end(6)

        # Voice selection combo box
        self.voice_combo = Gtk.ComboBoxText()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        voices = self.ATLAS.speech_manager.get_tts_voices(current_tts_provider)
        for voice in voices:
            voice_name = voice.get('name', 'Unknown')
            self.voice_combo.append_text(voice_name)
        if voices:
            self.voice_combo.set_active(0)
        eleven_box.append(Gtk.Label(label="Select TTS Voice:"))
        eleven_box.append(self.voice_combo)

        # Eleven Labs API key
        self.eleven_api_entry = Gtk.Entry()
        existing_xi_api = self.ATLAS.config_manager.get_config("XI_API_KEY") or ""
        self.eleven_api_entry.set_text(existing_xi_api)
        eleven_box.append(Gtk.Label(label="Eleven Labs API Key:"))
        eleven_box.append(self.eleven_api_entry)

        # Add the "Eleven Labs TTS" page to the notebook
        self.notebook.append_page(eleven_box, Gtk.Label(label="Eleven Labs TTS"))

    # -------------------------------------------------------------------------
    #   TAB 3: GOOGLE (TTS & STT)
    # -------------------------------------------------------------------------
    def create_google_tab(self):
        """
        Creates a tab for all Google settings used by both TTS and STT.
        Contains Google Cloud credentials.
        """
        google_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        google_box.set_margin_top(6)
        google_box.set_margin_bottom(6)
        google_box.set_margin_start(6)
        google_box.set_margin_end(6)

        # Google Cloud Credentials entry
        self.google_credentials_entry = Gtk.Entry()
        existing_google_creds = self.ATLAS.config_manager.get_config("GOOGLE_APPLICATION_CREDENTIALS") or ""
        self.google_credentials_entry.set_text(existing_google_creds)
        google_box.append(Gtk.Label(label="Google Cloud Credentials JSON Path:"))
        google_box.append(self.google_credentials_entry)

        google_box.append(Gtk.Label(label="These credentials are used for both Google TTS and STT."))

        self.notebook.append_page(google_box, Gtk.Label(label="Google"))

    # -------------------------------------------------------------------------
    #   TAB 4: WHISPER STT
    # -------------------------------------------------------------------------
    def create_whisper_stt_tab(self):
        """
        Creates a tab for Whisper STT settings:
        - Whisper mode (Local/Online)
        - OpenAI API key (for Whisper Online)
        """
        whisper_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        whisper_box.set_margin_top(6)
        whisper_box.set_margin_bottom(6)
        whisper_box.set_margin_start(6)
        whisper_box.set_margin_end(6)

        # Whisper mode selection combo box
        whisper_mode_label = Gtk.Label(label="Whisper Mode:")
        self.whisper_mode_combo = Gtk.ComboBoxText()
        self.whisper_mode_combo.append_text("Local")
        self.whisper_mode_combo.append_text("Online")
        self.whisper_mode_combo.set_active(0)
        whisper_box.append(whisper_mode_label)
        whisper_box.append(self.whisper_mode_combo)

        # OpenAI API key for Whisper Online
        self.openai_api_entry = Gtk.Entry()
        existing_openai_api = self.ATLAS.config_manager.get_config("OPENAI_API_KEY") or ""
        self.openai_api_entry.set_text(existing_openai_api)
        whisper_box.append(Gtk.Label(label="OpenAI API Key (for Whisper Online):"))
        whisper_box.append(self.openai_api_entry)

        self.notebook.append_page(whisper_box, Gtk.Label(label="Whisper STT"))

    # -------------------------------------------------------------------------
    #   SAVE SETTINGS
    # -------------------------------------------------------------------------
    def on_save_clicked(self, widget):
        """
        Handler for the "Save Settings" button.
        Gathers data from all tabs and updates the configuration manager and environment.
        Also updates the active TTS/STT providers in the Speech Manager.
        """
        # ---------------------------
        # General Tab: TTS and STT toggles & default provider selectors
        # ---------------------------
        tts_enabled = self.general_tts_switch.get_active()
        stt_enabled = self.general_stt_switch.get_active()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        self.ATLAS.speech_manager.set_tts_status(tts_enabled, current_tts_provider)
        logger.info(f"General settings - TTS enabled: {tts_enabled}, STT enabled: {stt_enabled}")

        # Update default providers based on selectors.
        selected_tts_provider = self.default_tts_combo.get_active_text()
        if selected_tts_provider:
            self.ATLAS.speech_manager.set_default_tts_provider(selected_tts_provider)
            logger.info(f"Default TTS provider set to: {selected_tts_provider}")

        if not stt_enabled:
            # Disable STT by setting active_stt to None.
            self.ATLAS.speech_manager.active_stt = None
            logger.info("STT has been disabled.")
        else:
            selected_stt_provider = self.default_stt_combo.get_active_text()
            if selected_stt_provider:
                self.ATLAS.speech_manager.set_default_stt_provider(selected_stt_provider)
                logger.info(f"Default STT provider set to: {selected_stt_provider}")
            else:
                logger.warning("STT enabled but no default provider was selected.")

        # ---------------------------
        # Google Tab: Update credentials
        # ---------------------------
        google_creds = self.google_credentials_entry.get_text().strip()
        self.ATLAS.config_manager.config["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds

        # ---------------------------
        # Eleven Labs TTS Tab: Update API key and voice
        # ---------------------------
        eleven_api_key = self.eleven_api_entry.get_text().strip()
        self.ATLAS.config_manager.config["XI_API_KEY"] = eleven_api_key
        os.environ["XI_API_KEY"] = eleven_api_key

        selected_voice_name = self.voice_combo.get_active_text()
        voices = self.ATLAS.speech_manager.get_tts_voices(self.ATLAS.speech_manager.get_default_tts_provider())
        selected_voice = next((v for v in voices if v.get('name') == selected_voice_name), None)
        if selected_voice:
            self.ATLAS.speech_manager.set_tts_voice(selected_voice, self.ATLAS.speech_manager.get_default_tts_provider())

                # ---------------------------
        # Whisper STT Tab: Update Whisper mode and OpenAI API key
        # ---------------------------
        openai_api_key = self.openai_api_entry.get_text().strip()
        self.ATLAS.config_manager.config["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        whisper_mode = self.whisper_mode_combo.get_active_text().lower()  # Define whisper_mode here
        from modules.Speech_Services.whisper_stt import WhisperSTT
        try:
            whisper_stt = WhisperSTT(mode=whisper_mode)
            provider_key = f"whisper_{whisper_mode}"
            self.ATLAS.speech_manager.add_stt_provider(provider_key, whisper_stt)
            self.ATLAS.speech_manager.set_default_stt_provider(provider_key)
        except Exception as e:
            logger.error(f"Error setting Whisper mode: {e}")

        logger.info("Speech settings saved.")
        self.destroy()
