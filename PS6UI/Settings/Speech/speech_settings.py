from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QComboBox, QWidget, QTabWidget
)
from PySide6.QtCore import Qt

class SpeechSettings(QDialog):
    """Simplified Speech Settings window."""

    def __init__(self, atlas):
        super().__init__(atlas.sidebar if hasattr(atlas, 'sidebar') else None)
        self.setWindowTitle("Speech Settings")
        self.resize(500, 800)
        self.ATLAS = atlas

        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.general_tab = self.create_general_tab()
        self.tab_widget.addTab(self.general_tab, "General")

        # Placeholder tabs for parity with GTK version
        for label in ("Eleven Labs TTS", "Google", "OpenAI"):
            w = QWidget()
            w.setLayout(QVBoxLayout())
            self.tab_widget.addTab(w, label)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_general_tab)
        button_layout.addStretch(1)
        button_layout.addWidget(save_btn)
        main_layout.addLayout(button_layout)

    def create_general_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignTop)

        tts_label = QLabel("Enable TTS:")
        self.general_tts_switch = QCheckBox()
        current_tts = self.ATLAS.speech_manager.get_default_tts_provider()
        self.general_tts_switch.setChecked(
            self.ATLAS.speech_manager.get_tts_status(current_tts)
        )
        tts_row = QHBoxLayout()
        tts_row.addWidget(tts_label)
        tts_row.addWidget(self.general_tts_switch)
        layout.addLayout(tts_row)

        tts_provider_label = QLabel("Default TTS Provider:")
        self.default_tts_combo = QComboBox()
        for key in self.ATLAS.speech_manager.tts_services.keys():
            self.default_tts_combo.addItem(key)
        if current_tts:
            idx = list(self.ATLAS.speech_manager.tts_services.keys()).index(current_tts)
            self.default_tts_combo.setCurrentIndex(idx)
        tts_provider_row = QHBoxLayout()
        tts_provider_row.addWidget(tts_provider_label)
        tts_provider_row.addWidget(self.default_tts_combo)
        layout.addLayout(tts_provider_row)

        stt_label = QLabel("Enable STT:")
        self.general_stt_switch = QCheckBox()
        default_stt = self.ATLAS.speech_manager.get_default_stt_provider()
        self.general_stt_switch.setChecked(bool(default_stt))
        stt_row = QHBoxLayout()
        stt_row.addWidget(stt_label)
        stt_row.addWidget(self.general_stt_switch)
        layout.addLayout(stt_row)

        stt_provider_label = QLabel("Default STT Provider:")
        self.default_stt_combo = QComboBox()
        for key in self.ATLAS.speech_manager.stt_services.keys():
            self.default_stt_combo.addItem(key)
        if default_stt:
            idx = list(self.ATLAS.speech_manager.stt_services.keys()).index(default_stt)
            self.default_stt_combo.setCurrentIndex(idx)
        stt_provider_row = QHBoxLayout()
        stt_provider_row.addWidget(stt_provider_label)
        stt_provider_row.addWidget(self.default_stt_combo)
        layout.addLayout(stt_provider_row)

        return w

    def save_general_tab(self):
        tts_enabled = self.general_tts_switch.isChecked()
        stt_enabled = self.general_stt_switch.isChecked()

        current_tts = self.ATLAS.speech_manager.get_default_tts_provider()
        self.ATLAS.speech_manager.set_tts_status(tts_enabled, current_tts)
        selected_tts = self.default_tts_combo.currentText()
        if selected_tts:
            self.ATLAS.speech_manager.set_default_tts_provider(selected_tts)

        if not stt_enabled:
            self.ATLAS.speech_manager.active_stt = None
        else:
            selected_stt = self.default_stt_combo.currentText()
            if selected_stt:
                self.ATLAS.speech_manager.set_default_stt_provider(selected_stt)

        self.accept()
