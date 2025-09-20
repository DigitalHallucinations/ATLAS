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
        GPT-4o STT, and GPT-4o Mini STT), language selection, task (transcribe/translate),
        and an initial prompt (with its label above the entry).
      - The TTS settings section includes a provider drop–down (currently only GPT-4o Mini TTS)
        and uses a password field for the API key that hides its content after saving.

Author: Jeremy Shows - Digital Hallucinations
Date: 05-11-2025
"""
import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
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

        # Paths for custom eye icons (fallbacks to themed symbolic icons if not found)
        self._eye_icon_path = self._abs_icon("../../Icons/eye.png")
        self._eye_off_icon_path = self._abs_icon("../../Icons/eye-off.png")

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
        self.current_page_index = 0

        # Create tabs.
        self.create_general_tab()
        self.create_eleven_labs_tts_tab()
        self.create_google_tab()
        self.create_openai_tab()

        # Tab switch handler.
        self.notebook.connect("switch-page", self.on_switch_page)

        # Global File Upload (if needed)
        file_upload_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        upload_label = Gtk.Label(label="Upload Audio File:")
        upload_label.set_tooltip_text("Pick an audio file to test your STT flow (WAV/MP3/FLAC).")
        self.file_upload_button = Gtk.Button(label="Select Audio File")
        self.file_upload_button.set_tooltip_text("Open a file chooser to select an audio file for transcription.")
        self.file_upload_button.connect("clicked", self.on_file_choose_clicked)
        self.selected_file_label = Gtk.Label(label="No file selected")
        self.selected_file_label.set_tooltip_text("Shows the path of the selected audio file.")
        file_upload_box.append(upload_label)
        file_upload_box.append(self.file_upload_button)
        file_upload_box.append(self.selected_file_label)
        vbox.append(file_upload_box)

        # Transcription History button.
        history_button = Gtk.Button(label="View Transcription History")
        history_button.set_tooltip_text("Open a quick log of past transcriptions.")
        history_button.connect("clicked", self.show_history)
        vbox.append(history_button)

        self.present()

    # ----------------------- Helpers -----------------------

    def _abs_icon(self, relative_path: str) -> str:
        """Resolve absolute path for an icon relative to this file."""
        base = os.path.abspath(os.path.dirname(__file__))
        return os.path.abspath(os.path.join(base, relative_path))

    def _load_icon_picture(self, primary_path: str, fallback_icon_name: str, size: int = 18) -> Gtk.Widget:
        """
        Try to load a paintable from a file path; fall back to a themed icon name.
        Returns a Gtk.Picture (file) or Gtk.Image (themed) as a Widget.
        """
        try:
            texture = Gdk.Texture.new_from_filename(primary_path)
            pic = Gtk.Picture.new_for_paintable(texture)
            pic.set_size_request(size, size)
            pic.set_content_fit(Gtk.ContentFit.CONTAIN)
            return pic
        except Exception:
            img = Gtk.Image.new_from_icon_name(fallback_icon_name)
            img.set_pixel_size(size)
            return img

    def _build_secret_row(self, label_text: str, default_visible: bool = False):
        """
        Build a labeled secret row with: Label, Entry (password-mode), and an eye ToggleButton.
        Returns (container_box, entry_widget, toggle_button).
        """
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        label = Gtk.Label(label=label_text)
        label.set_xalign(0.0)
        label.set_tooltip_text("Enter your value. Use the eye to show/hide.")
        vbox.append(label)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        entry = Gtk.Entry()
        entry.set_placeholder_text("Enter value here")
        entry.set_invisible_char('*')
        entry.set_visibility(default_visible)  # default visibility per caller
        entry.set_tooltip_text("Value is hidden when the eye is off.")
        entry.set_hexpand(True)
        hbox.append(entry)

        toggle = Gtk.ToggleButton()
        toggle.set_can_focus(True)
        # Pick a sensible default: if default_visible is True, eye shows "conceal"
        toggle.set_tooltip_text("Hide value" if default_visible else "Show value")
        role = getattr(Gtk.AccessibleRole, "BUTTON", None)
        if role is not None:
            toggle.set_accessible_role(role)
        icon_path = self._eye_off_icon_path if default_visible else self._eye_icon_path
        fallback_icon = "view-conceal-symbolic" if default_visible else "view-reveal-symbolic"
        eye_widget = self._load_icon_picture(icon_path, fallback_icon, 18)
        toggle.set_child(eye_widget)
        toggle.set_active(default_visible)

        def on_toggled(btn: Gtk.ToggleButton):
            visible = btn.get_active()
            entry.set_visibility(visible)
            icon_name = "view-conceal-symbolic" if visible else "view-reveal-symbolic"
            icon_path_local = self._eye_off_icon_path if visible else self._eye_icon_path
            new_widget = self._load_icon_picture(icon_path_local, icon_name, 18)
            btn.set_child(new_widget)
            btn.set_tooltip_text("Hide value" if visible else "Show value")

        toggle.connect("toggled", on_toggled)
        hbox.append(toggle)

        vbox.append(hbox)
        return vbox, entry, toggle

    # ----------------------- Tab Switching & Saving -----------------------

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
            # NOTE: Gtk.Dialog.run() is deprecated in GTK4, but keeping your current flow.
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

        # TTS Enable
        hbox_tts = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        tts_label = Gtk.Label(label="Enable TTS:")
        tts_label.set_tooltip_text("Master switch for Text-to-Speech output.")
        self.general_tts_switch = Gtk.Switch()
        self.general_tts_switch.set_tooltip_text("Turn TTS on/off globally.")
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        self.general_tts_switch.set_active(self.ATLAS.speech_manager.get_tts_status(current_tts_provider))
        hbox_tts.append(tts_label)
        hbox_tts.append(self.general_tts_switch)
        general_box.append(hbox_tts)

        # Default TTS Provider
        hbox_tts_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        tts_provider_label = Gtk.Label(label="Default TTS Provider:")
        tts_provider_label.set_tooltip_text("Choose the service used for TTS by default.")
        self.default_tts_combo = Gtk.ComboBoxText()
        self.default_tts_combo.set_tooltip_text("Select the default TTS service.")
        for key in self.ATLAS.speech_manager.tts_services.keys():
            self.default_tts_combo.append_text(key)
        if current_tts_provider:
            index = list(self.ATLAS.speech_manager.tts_services.keys()).index(current_tts_provider)
            self.default_tts_combo.set_active(index)
        hbox_tts_provider.append(tts_provider_label)
        hbox_tts_provider.append(self.default_tts_combo)
        general_box.append(hbox_tts_provider)

        # STT Enable
        hbox_stt = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        stt_label = Gtk.Label(label="Enable STT:")
        stt_label.set_tooltip_text("Master switch for Speech-to-Text input.")
        self.general_stt_switch = Gtk.Switch()
        self.general_stt_switch.set_tooltip_text("Turn STT on/off globally.")
        default_stt = self.ATLAS.speech_manager.get_default_stt_provider()
        self.general_stt_switch.set_active(True if default_stt else False)
        hbox_stt.append(stt_label)
        hbox_stt.append(self.general_stt_switch)
        general_box.append(hbox_stt)

        # Default STT Provider
        hbox_stt_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        stt_provider_label = Gtk.Label(label="Default STT Provider:")
        stt_provider_label.set_tooltip_text("Choose the service used for STT by default.")
        self.default_stt_combo = Gtk.ComboBoxText()
        self.default_stt_combo.set_tooltip_text("Select the default STT service.")
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
        save_button.set_tooltip_text("Save global speech settings (TTS/STT toggles and defaults).")
        save_button.connect("clicked", lambda w: self.save_general_tab())
        general_box.append(save_button)

        self.tab_dirty[0] = False
        # Mark changes as dirty.
        self.general_tts_switch.connect("notify::active", lambda w, ps: self.mark_dirty(0))
        self.default_tts_combo.connect("changed", lambda w: self.mark_dirty(0))
        self.general_stt_switch.connect("notify::active", lambda w, ps: self.mark_dirty(0))
        self.default_stt_combo.connect("changed", lambda w: self.mark_dirty(0))

        tab_label = Gtk.Label(label="General")
        tab_label.set_tooltip_text("Global TTS/STT switches and default providers.")
        self.notebook.append_page(general_box, tab_label)
        self.current_page_index = 0

    def save_general_tab(self):
        tts_enabled = self.general_tts_switch.get_active()
        stt_enabled = self.general_stt_switch.get_active()
        current_tts_provider = self.ATLAS.speech_manager.get_default_tts_provider()
        self.ATLAS.speech_manager.set_tts_status(tts_enabled, current_tts_provider)
        selected_tts_provider = self.default_tts_combo.get_active_text()
        selected_stt_provider = self.default_stt_combo.get_active_text() if stt_enabled else None

        self.ATLAS.speech_manager.set_default_speech_providers(
            tts_provider=selected_tts_provider,
            stt_provider=selected_stt_provider,
        )

        if not stt_enabled:
            self.ATLAS.speech_manager.disable_stt()
            logger.info("General: STT disabled.")
        else:
            if selected_stt_provider:
                logger.info(f"General: Default STT provider set to: {selected_stt_provider}")

        if selected_tts_provider:
            logger.info(f"General: Default TTS provider set to: {selected_tts_provider}")
        self.tab_dirty[0] = False

    # ----------------------- Eleven Labs TTS Tab -----------------------
    def create_eleven_labs_tts_tab(self):
        eleven_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        eleven_box.set_margin_top(6)
        eleven_box.set_margin_bottom(6)
        eleven_box.set_margin_start(6)
        eleven_box.set_margin_end(6)

        # Voice selection
        hbox_voice = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        voice_label = Gtk.Label(label="Select TTS Voice:")
        voice_label.set_tooltip_text("Pick the voice used by the current TTS provider.")
        self.voice_combo = Gtk.ComboBoxText()
        self.voice_combo.set_tooltip_text("Available voices for the selected TTS provider.")
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

        # Eleven Labs API key (with eye toggle; default hidden)
        api_row, self.eleven_api_entry, self.eleven_api_toggle = self._build_secret_row("Eleven Labs API Key:", default_visible=False)
        existing_xi_api = self.ATLAS.config_manager.get_config("XI_API_KEY") or ""
        if existing_xi_api:
            self.eleven_api_entry.set_text("")  # do not reveal stored
            self.eleven_api_entry.set_placeholder_text("Saved")
        eleven_box.append(api_row)

        save_button = Gtk.Button(label="Save Eleven Labs Settings")
        save_button.set_tooltip_text("Save Eleven Labs voice and API key.")
        save_button.connect("clicked", lambda w: self.save_eleven_labs_tab())
        eleven_box.append(save_button)

        self.tab_dirty[1] = False
        self.voice_combo.connect("changed", lambda w: self.mark_dirty(1))
        self.eleven_api_entry.connect("notify::text", lambda w, ps: self.mark_dirty(1))

        tab_label = Gtk.Label(label="Eleven Labs TTS")
        tab_label.set_tooltip_text("Configure Eleven Labs voice and credentials.")
        self.notebook.append_page(eleven_box, tab_label)

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

        # Row: label + entry + eye toggle (optional) + file picker
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        creds_label = Gtk.Label(label="Google Credentials JSON Path:")
        creds_label.set_xalign(0.0)
        creds_label.set_tooltip_text("Path to your Google Cloud credentials JSON used by Google STT/TTS.")
        row.append(creds_label)

        h = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        # Entry (default visible; it's just a path)
        self.google_credentials_entry = Gtk.Entry()
        self.google_credentials_entry.set_tooltip_text("Enter the absolute path to the Google credentials JSON.")
        self.google_credentials_entry.set_visibility(True)
        self.google_credentials_entry.set_hexpand(True)
        existing_google_creds = self.ATLAS.config_manager.get_config("GOOGLE_APPLICATION_CREDENTIALS") or ""
        self.google_credentials_entry.set_text(existing_google_creds)
        h.append(self.google_credentials_entry)

        # Optional eye toggle to hide/show the path if desired (default visible)
        self.google_creds_toggle = Gtk.ToggleButton()
        self.google_creds_toggle.set_can_focus(True)
        role = getattr(Gtk.AccessibleRole, "BUTTON", None)
        if role is not None:
            self.google_creds_toggle.set_accessible_role(role)
        self.google_creds_toggle.set_tooltip_text("Hide value" if self.google_credentials_entry.get_visibility() else "Show value")
        icon_path = self._eye_off_icon_path if self.google_credentials_entry.get_visibility() else self._eye_icon_path
        fallback_icon = "view-conceal-symbolic" if self.google_credentials_entry.get_visibility() else "view-reveal-symbolic"
        self.google_creds_toggle.set_child(self._load_icon_picture(icon_path, fallback_icon, 18))
        self.google_creds_toggle.set_active(self.google_credentials_entry.get_visibility())

        def on_google_eye_toggled(btn: Gtk.ToggleButton):
            visible = btn.get_active()
            self.google_credentials_entry.set_visibility(visible)
            icon_name = "view-conceal-symbolic" if visible else "view-reveal-symbolic"
            icon_path_local = self._eye_off_icon_path if visible else self._eye_icon_path
            btn.set_child(self._load_icon_picture(icon_path_local, icon_name, 18))
            btn.set_tooltip_text("Hide value" if visible else "Show value")

        self.google_creds_toggle.connect("toggled", on_google_eye_toggled)
        h.append(self.google_creds_toggle)

        # File picker button
        self.google_creds_picker_btn = Gtk.Button()
        self.google_creds_picker_btn.set_tooltip_text("Browse to select your Google credentials JSON file.")
        folder_icon = Gtk.Image.new_from_icon_name("document-open-symbolic")
        folder_icon.set_pixel_size(18)
        self.google_creds_picker_btn.set_child(folder_icon)

        def on_pick_file(_btn):
            dialog = Gtk.FileChooserNative(
                title="Select Google Credentials JSON",
                action=Gtk.FileChooserAction.OPEN,
                transient_for=self
            )
            # Filter for .json files
            json_filter = Gtk.FileFilter()
            json_filter.set_name("JSON files")
            json_filter.add_mime_type("application/json")
            json_filter.add_pattern("*.json")
            dialog.add_filter(json_filter)
            response = dialog.run()
            if response == Gtk.ResponseType.ACCEPT:
                file_path = dialog.get_filename()
                if file_path:
                    self.google_credentials_entry.set_text(file_path)
                    self.mark_dirty(2)
            dialog.destroy()

        self.google_creds_picker_btn.connect("clicked", on_pick_file)
        h.append(self.google_creds_picker_btn)

        row.append(h)
        google_box.append(row)

        note_label = Gtk.Label(label="These credentials are used for both Google TTS and STT.")
        note_label.set_tooltip_text("One set of Google creds powers both text-to-speech and speech-to-text.")
        google_box.append(note_label)

        save_button = Gtk.Button(label="Save Google Settings")
        save_button.set_tooltip_text("Save the Google credentials path.")
        save_button.connect("clicked", lambda w: self.save_google_tab())
        google_box.append(save_button)

        self.tab_dirty[2] = False
        self.google_credentials_entry.connect("notify::text", lambda w, ps: self.mark_dirty(2))

        tab_label = Gtk.Label(label="Google")
        tab_label.set_tooltip_text("Configure Google Cloud credentials for STT/TTS.")
        self.notebook.append_page(google_box, tab_label)

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

        # STT Provider
        hbox_stt_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        stt_provider_label = Gtk.Label(label="STT Provider:")
        stt_provider_label.set_tooltip_text("Pick the OpenAI-based service for speech recognition.")
        self.openai_stt_combo = Gtk.ComboBoxText()
        self.openai_stt_combo.set_tooltip_text("Whisper Online, GPT-4o STT, or GPT-4o Mini STT.")
        self.openai_stt_combo.append_text("Whisper Online")
        self.openai_stt_combo.append_text("GPT-4o STT")
        self.openai_stt_combo.append_text("GPT-4o Mini STT")
        self.openai_stt_combo.set_active(0)
        hbox_stt_provider.append(stt_provider_label)
        hbox_stt_provider.append(self.openai_stt_combo)
        stt_box.append(hbox_stt_provider)

        # Language
        hbox_language = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        language_label = Gtk.Label(label="Language:")
        language_label.set_tooltip_text("Force a language for STT, or keep Auto to let the model detect.")
        self.openai_language_combo = Gtk.ComboBoxText()
        self.openai_language_combo.set_tooltip_text("Preferred recognition language (or Auto).")
        for label_text in ["Auto", "English (en)", "Japanese (ja)", "Spanish (es)", "French (fr)"]:
            self.openai_language_combo.append_text(label_text)
        self.openai_language_combo.set_active(0)
        hbox_language.append(language_label)
        hbox_language.append(self.openai_language_combo)
        stt_box.append(hbox_language)

        # Task
        hbox_task = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        task_label = Gtk.Label(label="Task:")
        task_label.set_tooltip_text("Transcribe = same language; Translate = to English.")
        self.openai_task_combo = Gtk.ComboBoxText()
        self.openai_task_combo.set_tooltip_text("Choose whether to transcribe or translate.")
        self.openai_task_combo.append_text("transcribe")
        self.openai_task_combo.append_text("translate")
        self.openai_task_combo.set_active(0)
        hbox_task.append(task_label)
        hbox_task.append(self.openai_task_combo)
        stt_box.append(hbox_task)

        # Initial Prompt
        prompt_label = Gtk.Label(label="Initial Prompt (optional):")
        prompt_label.set_tooltip_text("Short hint to bias recognition (e.g., vocabulary, names).")
        self.openai_prompt_entry = Gtk.Entry()
        self.openai_prompt_entry.set_tooltip_text("Optional: give context or vocabulary to improve STT.")
        stt_box.append(prompt_label)
        stt_box.append(self.openai_prompt_entry)

        stt_frame.set_child(stt_box)
        openai_box.append(stt_frame)

        # --- TTS Settings Frame ---
        tts_frame = Gtk.Frame(label="Open AI TTS Settings")
        tts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        tts_box.set_margin_top(6)
        tts_box.set_margin_bottom(6)
        tts_box.set_margin_start(6)
        tts_box.set_margin_end(6)

        # TTS Provider
        hbox_tts_provider = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        tts_provider_label = Gtk.Label(label="TTS Provider:")
        tts_provider_label.set_tooltip_text("Pick the OpenAI service for speech synthesis.")
        self.openai_tts_combo = Gtk.ComboBoxText()
        self.openai_tts_combo.set_tooltip_text("Currently supports GPT-4o Mini TTS.")
        self.openai_tts_combo.append_text("GPT-4o Mini TTS")
        self.openai_tts_combo.set_active(0)
        hbox_tts_provider.append(tts_provider_label)
        hbox_tts_provider.append(self.openai_tts_combo)
        tts_box.append(hbox_tts_provider)

        tts_frame.set_child(tts_box)
        openai_box.append(tts_frame)

        # --- Shared OpenAI API Key (with eye toggle; default hidden) ---
        api_row, self.openai_api_entry, self.openai_api_toggle = self._build_secret_row("Open AI API Key:", default_visible=False)
        existing_api = self.ATLAS.config_manager.get_config("OPENAI_API_KEY") or ""
        if existing_api:
            self.openai_api_entry.set_text("")  # do not reveal stored
            self.openai_api_entry.set_placeholder_text("Saved")
        openai_box.append(api_row)

        # Save button for Open AI tab.
        save_button = Gtk.Button(label="Save Open AI Settings")
        save_button.set_tooltip_text("Save OpenAI STT/TTS preferences and API key.")
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

        tab_label = Gtk.Label(label="Open AI")
        tab_label.set_tooltip_text("Configure OpenAI STT/TTS and API key.")
        self.notebook.append_page(openai_box, tab_label)

    def save_openai_tab(self):
        openai_api_key = self.openai_api_entry.get_text().strip()
        stt_provider = self.openai_stt_combo.get_active_text()
        language_active = self.openai_language_combo.get_active_text()
        language_map = {"Auto": "", "English (en)": "en", "Japanese (ja)": "ja", "Spanish (es)": "es", "French (fr)": "fr"}
        language_code = language_map.get(language_active, "")
        task = self.openai_task_combo.get_active_text().lower()
        initial_prompt = self.openai_prompt_entry.get_text().strip() or None

        # For Open AI TTS, we currently only support GPT-4o Mini TTS.
        tts_provider = self.openai_tts_combo.get_active_text()

        self.ATLAS.speech_manager.configure_openai_speech(
            api_key=openai_api_key,
            stt_provider=stt_provider,
            language=language_code,
            task=task,
            initial_prompt=initial_prompt,
            tts_provider=tts_provider,
        )

        logger.info(f"Open AI STT provider set to {stt_provider}")
        logger.info("Open AI TTS provider (GPT-4o Mini TTS) initialized.")

        self.tab_dirty[3] = False

    # ----------------------- File Picker & History -----------------------

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
