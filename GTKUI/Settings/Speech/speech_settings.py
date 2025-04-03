# GTKUI/Settings/Speech/speech_settings.py

"""
Module: speech_settings.py
Description:
    Enterprise production–ready Speech Settings window with a tabbed layout.
    Each tab now saves only its own settings. When a user switches tabs,
    if unsaved changes exist, they’re prompted to save or discard.
    In the Open AI tab:
      - All Open AI functions (both STT and TTS) are consolidated into one tab.
      - The STT settings include a provider drop–down (with options like Whisper Online,
        GPT‑4o STT, and GPT‑4o Mini STT), language selection, task (transcribe/translate),
        and an initial prompt (with its label above the entry).
      - The TTS settings section includes a provider drop–down (currently only GPT‑4o Mini TTS)
        and uses a password field for the API key that hides its content after saving.
    
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

        # Apply global CSS styling.
        self.get_style_context().add_class("chat-page")
        self.get_style_context().add_class("sidebar")
        apply_css()

        self.set_default_size(500, 800)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        vbox.set_margin_top(10)
        vbox.set_margin_bottom(10)
        vbox.set_margin_start(10)
        vbox.set_margin_end(10)
        self.set_child(vbox)

        # Notebook for tabs.
        self.notebook = Gtk.Notebook()
        vbox.append(self.notebook)

        # Keep track of unsaved changes per tab using tab indices.
        self.tab_dirty = {}  # e.g. {0: False, 1: False, ...}
        # We'll also track the current tab index.
        self.current_page_index = 0

        # Create tabs.
        self.create_general_tab()
        self.create_eleven_labs_tts_tab()
        self.create_google_tab()
        self.create_openai_tab()

        # Connect a handler to detect tab switches.
        self.notebook.connect("switch-page", self.on_switch_page)

        # Global File Upload (if needed)
        file_upload_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        upload_label = Gtk.Label(label="Upload Audio File:")
        self.file_upload_button = Gtk.Button(label="Select Audio File")
        self.file_upload_button.connect("clicked", self.on_file_choose_clicked)
        self.selected_file_label = Gtk.Label(label="No file selected")
        file_upload_box.append(upload_label)
        file_upload_box.append(self.file_upload_button)
        file_upload_box.append(self.selected_file_label)
        vbox.append(file_upload_box)

        # Transcription History button.
        history_button = Gtk.Button(label="View Transcription History")
        history_button.connect("clicked", self.show_history)
        vbox.append(history_button)

        self.present()

    def on_switch_page(self, notebook, new_page, new_page_index):
        # Check if the current (old) tab has unsaved changes.
        old_index = self.current_page_index
        if self.tab_dirty.get(old_index, False):
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.WARNING,
                buttons=Gtk.ButtonsType.NONE,
                text="You have unsaved changes in this tab."
            )
            dialog.format_secondary_text("Do you want to save your changes before switching tabs?")
            dialog.add_button("Save", Gtk.ResponseType.OK)
            dialog.add_button("Discard", Gtk.ResponseType.CANCEL)
            response = dialog.run()
            dialog.destroy()
            if response == Gtk.ResponseType.OK:
                self.save_tab(old_index)
            # In either case, mark the tab as clean.
            self.tab_dirty[old_index] = False
        # Update the current page index.
        self.current_page_index = new_page_index

    def save_tab(self, tab_index):
        # Dispatch saving based on tab index:
        # 0: General, 1: Eleven Labs TTS, 2: Google, 3: Open AI.
        if tab_index == 0:
            self.save_general_tab()
        elif tab_index == 1:
            self.save_eleven_labs_tab()
        elif tab_index == 2:
            self.save_google_tab()
        elif tab_index == 3:
            self.save_openai_tab()

    def mark_dirty(self, tab_index):
        self.tab_dirty[tab_index] = True

    # ----------------------- General Tab -----------------------
    def create_general_tab(self):
        general_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        general_box.set_margin_top(6)
        general_box.set_margin_bottom(6)
        general_box.set_margin_start(6)
        general_box.set_margin_end(6)

        # TTS Enable (label and switch in one row)
        hbox_tts = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        tts_label = Gtk.Label(label="Enable TTS:")
        self.general_tts_switch = Gtk.Switch()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        self.general_tts_switch.set_active(self.ATLAS.speech_manager.get_tts_status(current_tts_provider))
        hbox_tts.append(tts_label)
        hbox_tts.append(self.general_tts_switch)
        general_box.append(hbox_tts)

        # Default TTS Provider (same row)
        hbox_tts_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        tts_provider_label = Gtk.Label(label="Default TTS Provider:")
        self.default_tts_combo = Gtk.ComboBoxText()
        for key in self.ATLAS.speech_manager.tts_services.keys():
            self.default_tts_combo.append_text(key)
        if current_tts_provider:
            index = list(self.ATLAS.speech_manager.tts_services.keys()).index(current_tts_provider)
            self.default_tts_combo.set_active(index)
        hbox_tts_provider.append(tts_provider_label)
        hbox_tts_provider.append(self.default_tts_combo)
        general_box.append(hbox_tts_provider)

        # STT Enable (same row)
        hbox_stt = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        stt_label = Gtk.Label(label="Enable STT:")
        self.general_stt_switch = Gtk.Switch()
        default_stt = self.ATLAS.speech_manager.get_default_stt_provider()
        self.general_stt_switch.set_active(True if default_stt else False)
        hbox_stt.append(stt_label)
        hbox_stt.append(self.general_stt_switch)
        general_box.append(hbox_stt)

        # Default STT Provider (same row)
        hbox_stt_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        stt_provider_label = Gtk.Label(label="Default STT Provider:")
        self.default_stt_combo = Gtk.ComboBoxText()
        for key in self.ATLAS.speech_manager.stt_services.keys():
            self.default_stt_combo.append_text(key)
        if default_stt:
            index = list(self.ATLAS.speech_manager.stt_services.keys()).index(default_stt)
            self.default_stt_combo.set_active(index)
        hbox_stt_provider.append(stt_provider_label)
        hbox_stt_provider.append(self.default_stt_combo)
        general_box.append(hbox_stt_provider)

        # Save button for General tab.
        save_button = Gtk.Button(label="Save General Settings")
        save_button.connect("clicked", lambda w: self.save_general_tab())
        general_box.append(save_button)

        self.tab_dirty[0] = False
        # Mark changes as dirty.
        self.general_tts_switch.connect("notify::active", lambda w, ps: self.mark_dirty(0))
        self.default_tts_combo.connect("changed", lambda w: self.mark_dirty(0))
        self.general_stt_switch.connect("notify::active", lambda w, ps: self.mark_dirty(0))
        self.default_stt_combo.connect("changed", lambda w: self.mark_dirty(0))

        self.notebook.append_page(general_box, Gtk.Label(label="General"))
        self.current_page_index = 0

    def save_general_tab(self):
        tts_enabled = self.general_tts_switch.get_active()
        stt_enabled = self.general_stt_switch.get_active()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        self.ATLAS.speech_manager.set_tts_status(tts_enabled, current_tts_provider)
        selected_tts_provider = self.default_tts_combo.get_active_text()
        if selected_tts_provider:
            self.ATLAS.speech_manager.set_default_tts_provider(selected_tts_provider)
            logger.info(f"General: Default TTS provider set to: {selected_tts_provider}")
        if not stt_enabled:
            self.ATLAS.speech_manager.active_stt = None
            logger.info("General: STT disabled.")
        else:
            selected_stt_provider = self.default_stt_combo.get_active_text()
            if selected_stt_provider:
                self.ATLAS.speech_manager.set_default_stt_provider(selected_stt_provider)
                logger.info(f"General: Default STT provider set to: {selected_stt_provider}")
        self.tab_dirty[0] = False

    # ----------------------- Eleven Labs TTS Tab -----------------------
    def create_eleven_labs_tts_tab(self):
        eleven_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        eleven_box.set_margin_top(6)
        eleven_box.set_margin_bottom(6)
        eleven_box.set_margin_start(6)
        eleven_box.set_margin_end(6)

        # Voice selection (label and dropdown in one row)
        hbox_voice = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        voice_label = Gtk.Label(label="Select TTS Voice:")
        self.voice_combo = Gtk.ComboBoxText()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        voices = self.ATLAS.speech_manager.get_tts_voices(current_tts_provider)
        for voice in voices:
            voice_name = voice.get('name', 'Unknown')
            self.voice_combo.append_text(voice_name)
        if voices:
            self.voice_combo.set_active(0)
        hbox_voice.append(voice_label)
        hbox_voice.append(self.voice_combo)
        eleven_box.append(hbox_voice)

        # Eleven Labs API key as a password entry.
        hbox_api = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        api_label = Gtk.Label(label="Eleven Labs API Key:")
        self.eleven_api_entry = Gtk.Entry()
        self.eleven_api_entry.set_visibility(False)
        existing_xi_api = self.ATLAS.config_manager.get_config("XI_API_KEY") or ""
        if existing_xi_api:
            self.eleven_api_entry.set_text("")
            self.eleven_api_entry.set_placeholder_text("Saved")
        else:
            self.eleven_api_entry.set_text("")
        hbox_api.append(api_label)
        hbox_api.append(self.eleven_api_entry)
        eleven_box.append(hbox_api)

        save_button = Gtk.Button(label="Save Eleven Labs Settings")
        save_button.connect("clicked", lambda w: self.save_eleven_labs_tab())
        eleven_box.append(save_button)

        self.tab_dirty[1] = False
        self.voice_combo.connect("changed", lambda w: self.mark_dirty(1))
        self.eleven_api_entry.connect("notify::text", lambda w, ps: self.mark_dirty(1))

        self.notebook.append_page(eleven_box, Gtk.Label(label="Eleven Labs TTS"))

    def save_eleven_labs_tab(self):
        eleven_api_key = self.eleven_api_entry.get_text().strip()
        self.ATLAS.config_manager.config["XI_API_KEY"] = eleven_api_key
        os.environ["XI_API_KEY"] = eleven_api_key
        selected_voice_name = self.voice_combo.get_active_text()
        voices = self.ATLAS.speech_manager.get_tts_voices(self.ATLAS.speech_manager.get_default_tts_provider())
        selected_voice = next((v for v in voices if v.get('name') == selected_voice_name), None)
        if selected_voice:
            self.ATLAS.speech_manager.set_tts_voice(selected_voice, self.ATLAS.speech_manager.get_default_tts_provider())
        self.tab_dirty[1] = False

    # ----------------------- Google Tab -----------------------
    def create_google_tab(self):
        google_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        google_box.set_margin_top(6)
        google_box.set_margin_bottom(6)
        google_box.set_margin_start(6)
        google_box.set_margin_end(6)

        hbox_google = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        creds_label = Gtk.Label(label="Google Credentials JSON Path:")
        self.google_credentials_entry = Gtk.Entry()
        existing_google_creds = self.ATLAS.config_manager.get_config("GOOGLE_APPLICATION_CREDENTIALS") or ""
        self.google_credentials_entry.set_text(existing_google_creds)
        hbox_google.append(creds_label)
        hbox_google.append(self.google_credentials_entry)
        google_box.append(hbox_google)

        note_label = Gtk.Label(label="These credentials are used for both Google TTS and STT.")
        google_box.append(note_label)

        save_button = Gtk.Button(label="Save Google Settings")
        save_button.connect("clicked", lambda w: self.save_google_tab())
        google_box.append(save_button)

        self.tab_dirty[2] = False
        self.google_credentials_entry.connect("notify::text", lambda w, ps: self.mark_dirty(2))

        self.notebook.append_page(google_box, Gtk.Label(label="Google"))

    def save_google_tab(self):
        google_creds = self.google_credentials_entry.get_text().strip()
        self.ATLAS.config_manager.config["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds
        self.tab_dirty[2] = False

    # ----------------------- Open AI Tab -----------------------
    def create_openai_tab(self):
        """
        Creates the Open AI settings tab, which consolidates all Open AI functions (both STT and TTS).
        This tab is divided into two sections.
        
        STT Settings:
          - Provider drop–down (options: "Whisper Online", "GPT-4o STT", "GPT-4o Mini STT")
          - Language drop–down
          - Task drop–down (transcribe/translate)
          - Initial prompt entry (label above)
          - (Shared API key field)
        
        TTS Settings:
          - Provider drop–down (currently only "GPT-4o Mini TTS")
          - (Shared API key field)
        """
        openai_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        openai_box.set_margin_top(6)
        openai_box.set_margin_bottom(6)
        openai_box.set_margin_start(6)
        openai_box.set_margin_end(6)
        
        # --- STT Settings Frame ---
        stt_frame = Gtk.Frame(label="Open AI STT Settings")
        stt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        stt_box.set_margin_top(6)
        stt_box.set_margin_bottom(6)
        stt_box.set_margin_start(6)
        stt_box.set_margin_end(6)
        
        # STT Provider selection (dropdown)
        hbox_stt_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        stt_provider_label = Gtk.Label(label="STT Provider:")
        self.openai_stt_combo = Gtk.ComboBoxText()
        # Options: "Whisper Online", "GPT-4o STT", "GPT-4o Mini STT"
        self.openai_stt_combo.append_text("Whisper Online")
        self.openai_stt_combo.append_text("GPT-4o STT")
        self.openai_stt_combo.append_text("GPT-4o Mini STT")
        self.openai_stt_combo.set_active(0)
        hbox_stt_provider.append(stt_provider_label)
        hbox_stt_provider.append(self.openai_stt_combo)
        stt_box.append(hbox_stt_provider)
        
        # Language selection as a drop-down.
        hbox_language = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        language_label = Gtk.Label(label="Language:")
        self.openai_language_combo = Gtk.ComboBoxText()
        languages = [("Auto", ""), ("English (en)", "en"), ("Japanese (ja)", "ja"),
                     ("Spanish (es)", "es"), ("French (fr)", "fr")]
        for label_text, code in languages:
            self.openai_language_combo.append_text(label_text)
        self.openai_language_combo.set_active(0)
        hbox_language.append(language_label)
        hbox_language.append(self.openai_language_combo)
        stt_box.append(hbox_language)
        
        # Task selection (transcribe/translate)
        hbox_task = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        task_label = Gtk.Label(label="Task:")
        self.openai_task_combo = Gtk.ComboBoxText()
        self.openai_task_combo.append_text("transcribe")
        self.openai_task_combo.append_text("translate")
        self.openai_task_combo.set_active(0)
        hbox_task.append(task_label)
        hbox_task.append(self.openai_task_combo)
        stt_box.append(hbox_task)
        
        # Initial Prompt (label above entry)
        prompt_label = Gtk.Label(label="Initial Prompt (optional):")
        self.openai_prompt_entry = Gtk.Entry()
        stt_box.append(prompt_label)
        stt_box.append(self.openai_prompt_entry)
        
        stt_frame.set_child(stt_box)  # Use set_child instead of add
        
        openai_box.append(stt_frame)
        
        # --- TTS Settings Frame ---
        tts_frame = Gtk.Frame(label="Open AI TTS Settings")
        tts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        tts_box.set_margin_top(6)
        tts_box.set_margin_bottom(6)
        tts_box.set_margin_start(6)
        tts_box.set_margin_end(6)
        
        # TTS Provider selection (dropdown)
        hbox_tts_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        tts_provider_label = Gtk.Label(label="TTS Provider:")
        self.openai_tts_combo = Gtk.ComboBoxText()
        # Currently only one option:
        self.openai_tts_combo.append_text("GPT-4o Mini TTS")
        self.openai_tts_combo.set_active(0)
        hbox_tts_provider.append(tts_provider_label)
        hbox_tts_provider.append(self.openai_tts_combo)
        tts_box.append(hbox_tts_provider)
        
        tts_frame.set_child(tts_box)  # Use set_child instead of add
        openai_box.append(tts_frame)
        
        # Shared API Key for Open AI (password entry)
        hbox_api = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        api_label = Gtk.Label(label="Open AI API Key:")
        self.openai_api_entry = Gtk.Entry()
        self.openai_api_entry.set_visibility(False)
        existing_api = self.ATLAS.config_manager.get_config("OPENAI_API_KEY") or ""
        if existing_api:
            self.openai_api_entry.set_text("")
            self.openai_api_entry.set_placeholder_text("Saved")
        else:
            self.openai_api_entry.set_text("")
        hbox_api.append(api_label)
        hbox_api.append(self.openai_api_entry)
        openai_box.append(hbox_api)
        
        # Save button for Open AI tab.
        save_button = Gtk.Button(label="Save Open AI Settings")
        save_button.connect("clicked", lambda w: self.save_openai_tab())
        openai_box.append(save_button)
        
        self.tab_dirty[3] = False
        # Mark changes.
        self.openai_stt_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.openai_language_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.openai_task_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.openai_prompt_entry.connect("notify::text", lambda w, ps: self.mark_dirty(3))
        self.openai_tts_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.openai_api_entry.connect("notify::text", lambda w, ps: self.mark_dirty(3))
        
        self.notebook.append_page(openai_box, Gtk.Label(label="Open AI"))

    def save_openai_tab(self):
        openai_api_key = self.openai_api_entry.get_text().strip()
        self.ATLAS.config_manager.config["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        stt_provider = self.openai_stt_combo.get_active_text()
        language_active = self.openai_language_combo.get_active_text()
        language_code = ""
        for label_text, code in [("Auto", ""), ("English (en)", "en"), ("Japanese (ja)", "ja"), ("Spanish (es)", "es"), ("French (fr)", "fr")]:
            if label_text == language_active:
                language_code = code
                break
        task = self.openai_task_combo.get_active_text().lower()
        initial_prompt = self.openai_prompt_entry.get_text().strip() or None
        
        # For Open AI TTS, we currently only support GPT-4o Mini TTS.
        tts_provider = self.openai_tts_combo.get_active_text()
        
        # Save settings in the configuration manager.
        self.ATLAS.config_manager.config["OPENAI_STT_PROVIDER"] = stt_provider
        self.ATLAS.config_manager.config["OPENAI_LANGUAGE"] = language_code
        self.ATLAS.config_manager.config["OPENAI_TASK"] = task
        self.ATLAS.config_manager.config["OPENAI_INITIAL_PROMPT"] = initial_prompt
        self.ATLAS.config_manager.config["OPENAI_TTS_PROVIDER"] = tts_provider

        # Initialize or update the Open AI providers in the SpeechManager.
        # For STT:
        try:
            from modules.Speech_Services.gpt4o_stt import GPT4oSTT
            if stt_provider == "Whisper Online":
                from modules.Speech_Services.whisper_stt import WhisperSTT
                openai_stt = WhisperSTT(mode="online")
            elif stt_provider == "GPT-4o STT":
                openai_stt = GPT4oSTT(variant="gpt-4o")
            else:  # GPT-4o Mini STT
                openai_stt = GPT4oSTT(variant="gpt-4o-mini")
            provider_key = "openai_stt"
            self.ATLAS.speech_manager.add_stt_provider(provider_key, openai_stt)
            self.ATLAS.speech_manager.set_default_stt_provider(provider_key)
            logger.info(f"Open AI STT provider set to {stt_provider}")
        except Exception as e:
            logger.error(f"Error initializing Open AI STT provider: {e}")
        
        # For TTS:
        try:
            from modules.Speech_Services.gpt4o_tts import GPT4oTTS
            openai_tts = GPT4oTTS(voice="default")
            provider_key_tts = "openai_tts"
            self.ATLAS.speech_manager.add_tts_provider(provider_key_tts, openai_tts)
            self.ATLAS.speech_manager.set_default_tts_provider(provider_key_tts)
            logger.info("Open AI TTS provider (GPT-4o Mini TTS) initialized.")
        except Exception as e:
            logger.error(f"Error initializing Open AI TTS provider: {e}")
        
        self.tab_dirty[3] = False

    def _create_audio_filter(self):
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio Files")
        audio_filter.add_mime_type("audio/wav")
        audio_filter.add_pattern("*.wav")
        audio_filter.add_pattern("*.mp3")
        audio_filter.add_pattern("*.flac")
        return audio_filter

    def on_file_choose_clicked(self, widget):
        dialog = Gtk.FileChooserNative(
            title="Select Audio File",
            action=Gtk.FileChooserAction.OPEN,
            transient_for=self
        )
        audio_filter = self._create_audio_filter()
        dialog.add_filter(audio_filter)
        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:
            file_path = dialog.get_filename()
            self.selected_file_label.set_text(file_path)
            self.selected_audio_file = file_path
        dialog.destroy()

    def show_history(self, widget):
        history = self.ATLAS.speech_manager.transcription_history
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Transcription History"
        )
        history_text = "\n\n".join(
            f"Time: {entry['timestamp']}\nFile: {entry['audio_file']}\nTranscript: {entry['transcript']}"
            for entry in history
        )
        dialog.format_secondary_text(history_text if history_text else "No history available.")
        dialog.run()
        dialog.destroy()
