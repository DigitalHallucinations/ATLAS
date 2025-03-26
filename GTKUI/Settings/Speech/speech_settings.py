# GTKUI/Settings/Speech/speech_settings.py

"""
Module: speech_settings.py
Description:
    Enterprise production–ready Speech Settings window with a tabbed layout.
    Each tab now saves only its own settings. When a user switches tabs,
    if unsaved changes exist, they’re prompted to save or discard.
    In the Whisper STT tab:
      - The title and drop‐down box (for mode, model, device, language, and task) are on the same row.
      - Language is now a drop‐down list.
      - The initial prompt’s label remains above its entry.
      - The noise reduction and “return segments” options are arranged with their labels on the same row as their switches.
      - The API key entry is a password field that never displays its contents after saving.
    
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
        self.create_whisper_stt_tab()

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
        # 0: General, 1: Eleven Labs TTS, 2: Google, 3: Whisper STT.
        if tab_index == 0:
            self.save_general_tab()
        elif tab_index == 1:
            self.save_eleven_labs_tab()
        elif tab_index == 2:
            self.save_google_tab()
        elif tab_index == 3:
            self.save_whisper_tab()

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

    # ----------------------- Whisper STT Tab -----------------------
    def create_whisper_stt_tab(self):
        whisper_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        whisper_box.set_margin_top(6)
        whisper_box.set_margin_bottom(6)
        whisper_box.set_margin_start(6)
        whisper_box.set_margin_end(6)

        # Whisper Mode (title and dropdown on same row)
        hbox_mode = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        mode_label = Gtk.Label(label="Whisper Mode:")
        self.whisper_mode_combo = Gtk.ComboBoxText()
        self.whisper_mode_combo.append_text("Local")
        self.whisper_mode_combo.append_text("Online")
        self.whisper_mode_combo.set_active(0)
        hbox_mode.append(mode_label)
        hbox_mode.append(self.whisper_mode_combo)
        whisper_box.append(hbox_mode)

        # Whisper Model (same row)
        hbox_model = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        model_label = Gtk.Label(label="Whisper Model:")
        self.whisper_model_combo = Gtk.ComboBoxText()
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        for m in models:
            self.whisper_model_combo.append_text(m)
        self.whisper_model_combo.set_active(1)  # default "base"
        hbox_model.append(model_label)
        hbox_model.append(self.whisper_model_combo)
        whisper_box.append(hbox_model)

        # Device selection (same row)
        hbox_device = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        device_label = Gtk.Label(label="Device:")
        self.device_combo = Gtk.ComboBoxText()
        self.device_combo.append_text("Auto")
        self.device_combo.append_text("cuda")
        self.device_combo.append_text("cpu")
        self.device_combo.set_active(0)
        hbox_device.append(device_label)
        hbox_device.append(self.device_combo)
        whisper_box.append(hbox_device)

        # Language selection as a drop-down
        hbox_language = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        language_label = Gtk.Label(label="Language:")
        self.language_combo = Gtk.ComboBoxText()
        languages = [("Auto", ""), ("English (en)", "en"), ("Japanese (ja)", "ja"),
                     ("Spanish (es)", "es"), ("French (fr)", "fr")]
        for label_text, code in languages:
            # Use the label for display.
            self.language_combo.append_text(label_text)
        self.language_combo.set_active(0)
        hbox_language.append(language_label)
        hbox_language.append(self.language_combo)
        whisper_box.append(hbox_language)

        # Task selection (same row)
        hbox_task = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        task_label = Gtk.Label(label="Task:")
        self.task_combo = Gtk.ComboBoxText()
        self.task_combo.append_text("transcribe")
        self.task_combo.append_text("translate")
        self.task_combo.set_active(0)
        hbox_task.append(task_label)
        hbox_task.append(self.task_combo)
        whisper_box.append(hbox_task)

        # Initial Prompt (label above entry)
        prompt_label = Gtk.Label(label="Initial Prompt (optional):")
        self.prompt_entry = Gtk.Entry()
        whisper_box.append(prompt_label)
        whisper_box.append(self.prompt_entry)

        # Noise Reduction (title and switch on same row)
        hbox_noise = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        noise_label = Gtk.Label(label="Apply Noise Reduction:")
        self.noise_switch = Gtk.Switch()
        self.noise_switch.set_active(False)
        hbox_noise.append(noise_label)
        hbox_noise.append(self.noise_switch)
        whisper_box.append(hbox_noise)

        # Return Segments (title and switch on same row)
        hbox_segments = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        segments_label = Gtk.Label(label="Return Segments with Timestamps:")
        self.segments_switch = Gtk.Switch()
        self.segments_switch.set_active(False)
        hbox_segments.append(segments_label)
        hbox_segments.append(self.segments_switch)
        whisper_box.append(hbox_segments)

        # OpenAI API Key as a password entry.
        hbox_api = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        api_label = Gtk.Label(label="OpenAI API Key (for Whisper Online):")
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
        whisper_box.append(hbox_api)

        save_button = Gtk.Button(label="Save Whisper STT Settings")
        save_button.connect("clicked", lambda w: self.save_whisper_tab())
        whisper_box.append(save_button)

        self.tab_dirty[3] = False
        # Mark changes.
        self.whisper_mode_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.whisper_model_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.device_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.language_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.task_combo.connect("changed", lambda w: self.mark_dirty(3))
        self.prompt_entry.connect("notify::text", lambda w, ps: self.mark_dirty(3))
        self.noise_switch.connect("notify::active", lambda w, ps: self.mark_dirty(3))
        self.segments_switch.connect("notify::active", lambda w, ps: self.mark_dirty(3))
        self.openai_api_entry.connect("notify::text", lambda w, ps: self.mark_dirty(3))

        self.notebook.append_page(whisper_box, Gtk.Label(label="Whisper STT"))

    def save_whisper_tab(self):
        openai_api_key = self.openai_api_entry.get_text().strip()
        self.ATLAS.config_manager.config["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        whisper_mode = self.whisper_mode_combo.get_active_text().lower()  # "local" or "online"
        whisper_model = self.whisper_model_combo.get_active_text()
        device_choice = self.device_combo.get_active_text()
        device = None if device_choice.lower() == "auto" else device_choice.lower()
        language_active = self.language_combo.get_active_text()
        language_code = ""
        for label_text, code in [("Auto", ""), ("English (en)", "en"), ("Japanese (ja)", "ja"), ("Spanish (es)", "es"), ("French (fr)", "fr")]:
            if label_text == language_active:
                language_code = code
                break
        task = self.task_combo.get_active_text().lower()
        initial_prompt = self.prompt_entry.get_text().strip() or None
        noise_reduction = self.noise_switch.get_active()
        return_segments = self.segments_switch.get_active()

        self.ATLAS.config_manager.config["WHISPER_MODE"] = whisper_mode
        self.ATLAS.config_manager.config["WHISPER_MODEL"] = whisper_model
        self.ATLAS.config_manager.config["WHISPER_DEVICE"] = device
        self.ATLAS.config_manager.config["WHISPER_NOISE_REDUCTION"] = noise_reduction
        self.ATLAS.config_manager.config["WHISPER_FALLBACK"] = True
        self.ATLAS.config_manager.config["WHISPER_FS"] = 16000

        from modules.Speech_Services.whisper_stt import WhisperSTT
        try:
            whisper_stt = WhisperSTT(
                mode=whisper_mode,
                model_name=whisper_model,
                fs=16000,
                device=device,
                noise_reduction=noise_reduction,
                fallback_online=True
            )
            provider_key = f"whisper_{whisper_mode}"
            self.ATLAS.speech_manager.add_stt_provider(provider_key, whisper_stt)
            self.ATLAS.speech_manager.set_default_stt_provider(provider_key)
            logger.info(f"Whisper STT provider set to '{provider_key}' with model '{whisper_model}' on device '{device}'")
        except Exception as e:
            logger.error(f"Error setting Whisper mode: {e}")
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
