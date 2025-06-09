from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox
)


class SpeechSettingsWindow(QDialog):
    """Simplified speech settings dialog for PySide6."""

    def __init__(self, atlas, parent=None):
        super().__init__(parent)
        self.ATLAS = atlas
        self.setWindowTitle("Speech Settings")
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        self.tts_checkbox = QCheckBox("Enable TTS")
        self.tts_checkbox.setChecked(self.ATLAS.config_manager.get_tts_enabled())
        layout.addWidget(self.tts_checkbox)

        tts_box = QHBoxLayout()
        tts_box.addWidget(QLabel("Default TTS Provider:"))
        self.tts_combo = QComboBox()
        for key in self.ATLAS.speech_manager.tts_services.keys():
            self.tts_combo.addItem(key)
        default_tts = self.ATLAS.speech_manager.get_default_tts_provider()
        if default_tts:
            index = list(self.ATLAS.speech_manager.tts_services.keys()).index(default_tts)
            self.tts_combo.setCurrentIndex(index)
        tts_box.addWidget(self.tts_combo)
        layout.addLayout(tts_box)

        self.stt_checkbox = QCheckBox("Enable STT")
        self.stt_checkbox.setChecked(self.ATLAS.speech_manager.active_stt is not None)
        layout.addWidget(self.stt_checkbox)

        stt_box = QHBoxLayout()
        stt_box.addWidget(QLabel("Default STT Provider:"))
        self.stt_combo = QComboBox()
        for key in self.ATLAS.speech_manager.stt_services.keys():
            self.stt_combo.addItem(key)
        default_stt = self.ATLAS.speech_manager.get_default_stt_provider()
        if default_stt:
            index = list(self.ATLAS.speech_manager.stt_services.keys()).index(default_stt)
            self.stt_combo.setCurrentIndex(index)
        stt_box.addWidget(self.stt_combo)
        layout.addLayout(stt_box)

        button_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_settings)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_box.addWidget(save_btn)
        button_box.addWidget(close_btn)
        layout.addLayout(button_box)

    def save_settings(self):
        self.ATLAS.config_manager.set_tts_enabled(self.tts_checkbox.isChecked())
        tts_provider = self.tts_combo.currentText()
        if tts_provider:
            self.ATLAS.speech_manager.set_default_tts_provider(tts_provider)
        stt_provider = self.stt_combo.currentText()
        if stt_provider:
            self.ATLAS.speech_manager.set_default_stt_provider(stt_provider)
        self.accept()
