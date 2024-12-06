# UI/Provider_manager/Settings/HF_settings.py

import os
import asyncio
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTabWidget, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QWidget, QGridLayout, QMessageBox, QFileDialog, QListWidget, QScrollArea, QSlider, QMenu
)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QDoubleValidator, QAction

class HuggingFaceSettingsWindow(QDialog):
    """
    A settings dialog for configuring HuggingFace-related model settings and optimizations.
    
    This dialog allows the user to adjust parameters such as temperature, top-p, top-k, caching,
    NVMe offloading, fine-tuning parameters, and manage installed models. The user can also search
    for and download new models from the HuggingFace Hub.
    
    Once the user saves the settings, the `settingsSaved` signal is emitted with the settings dictionary.
    """
    # Define a signal that emits the saved settings dictionary
    settingsSaved = Signal(dict)

    def __init__(self, ATLAS, config_manager, parent_window):
        """
        Initialize the HuggingFaceSettingsWindow.

        Args:
            ATLAS: The main ATLAS application instance, providing access to persona, provider, and model management.
            config_manager: The configuration manager for reading and writing settings.
            parent_window: The parent QWidget or QMainWindow for this dialog.
        """
        super().__init__(parent_window)
        self.setWindowTitle("HuggingFace Settings")
        self.setModal(True)
        self.resize(800, 600)

        self.ATLAS = ATLAS
        self.config_manager = config_manager
        self.parent_window = parent_window

        main_layout = QVBoxLayout(self)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create and add tabs
        self.general_tab = self.create_general_settings_tab()
        self.tab_widget.addTab(self.general_tab, "General")

        # Advanced tab with scroll area
        self.advanced_tab = QTabWidget()
        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_container = QWidget()
        advanced_layout = QVBoxLayout(advanced_container)

        self.advanced_optimizations_tab = self.create_advanced_optimizations_tab()
        self.nvme_tab = self.create_nvme_offloading_tab()
        self.misc_tab = self.create_miscellaneous_tab()

        self.advanced_tab.addTab(self.advanced_optimizations_tab, "Optimizations")
        self.advanced_tab.addTab(self.nvme_tab, "NVMe Offloading")
        self.advanced_tab.addTab(self.misc_tab, "Misc")

        advanced_layout.addWidget(self.advanced_tab)
        advanced_container.setLayout(advanced_layout)
        advanced_scroll.setWidget(advanced_container)
        self.tab_widget.addTab(advanced_scroll, "Advanced")

        # Fine-tuning tab
        self.fine_tuning_tab = self.create_fine_tuning_settings_tab()
        fine_tune_button = QPushButton("Fine-Tune Model")
        fine_tune_button.clicked.connect(self.on_fine_tune_clicked)
        self.fine_tuning_layout.addWidget(fine_tune_button, 7, 1)
        self.tab_widget.addTab(self.fine_tuning_tab, "Fine-Tuning")

        # Models tab
        self.models_tab = QTabWidget()
        self.model_management_tab = self.create_model_management_tab()
        self.search_download_tab = self.create_search_download_tab()

        self.models_tab.addTab(self.model_management_tab, "Manage Models")
        self.models_tab.addTab(self.search_download_tab, "Search & Download")
        self.tab_widget.addTab(self.models_tab, "Models")

        # Control buttons at the bottom
        button_layout = QHBoxLayout()
        back_button = QPushButton("Back")
        back_button.clicked.connect(self.on_back_clicked)
        button_layout.addWidget(back_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.on_cancel_clicked)
        button_layout.addWidget(cancel_button)

        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.on_save_clicked)
        button_layout.addWidget(save_button)

        main_layout.addLayout(button_layout)

        self.populate_model_comboboxes()

    def show_message(self, title, message, icon=QMessageBox.Information):
        """
        Display an informational message dialog.

        Args:
            title (str): The title of the message dialog.
            message (str): The text message to display.
            icon: The icon type for the message box (e.g., QMessageBox.Information).
        """
        QMessageBox.information(self, title, message)

    def confirm_dialog(self, message):
        """
        Display a confirmation dialog with Yes/No options.

        Args:
            message (str): The question or message to confirm.

        Returns:
            bool: True if the user clicked Yes, False otherwise.
        """
        reply = QMessageBox.question(self, "Confirm", message, QMessageBox.Yes | QMessageBox.No)
        return reply == QMessageBox.Yes

    def create_general_settings_tab(self):
        """
        Create the "General" settings tab, including parameters like temperature, top-p, top-k,
        max tokens, repetition penalty, presence penalty, length penalty, early stopping,
        and do_sample.

        Returns:
            QWidget: The widget containing the general settings UI.
        """
        w = QWidget()
        layout = QGridLayout(w)
        layout.setSpacing(10)

        # Temperature with QDoubleSpinBox and QSlider
        temp_label = QLabel("Temperature:")
        layout.addWidget(temp_label, 0, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        layout.addWidget(self.temp_spin, 0, 1)

        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)
        self.temp_slider.setValue(int(self.temp_spin.value() * 100))
        layout.addWidget(self.temp_slider, 0, 2)

        self.temp_spin.valueChanged.connect(self.on_temp_spin_changed)
        self.temp_slider.valueChanged.connect(self.on_temp_slider_changed)

        topp_label = QLabel("Top-p:")
        layout.addWidget(topp_label, 1, 0)
        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.0, 1.0)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setValue(1.0)
        layout.addWidget(self.topp_spin, 1, 1)

        topk_label = QLabel("Top-k:")
        layout.addWidget(topk_label, 2, 0)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 1000)
        self.topk_spin.setValue(50)
        layout.addWidget(self.topk_spin, 2, 1)

        maxt_label = QLabel("Max Tokens:")
        layout.addWidget(maxt_label, 3, 0)
        self.maxt_spin = QSpinBox()
        self.maxt_spin.setRange(1, 2048)
        self.maxt_spin.setValue(100)
        layout.addWidget(self.maxt_spin, 3, 1)

        rp_label = QLabel("Repetition Penalty:")
        layout.addWidget(rp_label, 4, 0)
        self.rp_spin = QDoubleSpinBox()
        self.rp_spin.setRange(0.0, 10.0)
        self.rp_spin.setValue(1.0)
        self.rp_spin.setSingleStep(0.1)
        layout.addWidget(self.rp_spin, 4, 1)

        pres_penalty_label = QLabel("Presence Penalty:")
        layout.addWidget(pres_penalty_label, 5, 0)
        self.pres_penalty_spin = QDoubleSpinBox()
        self.pres_penalty_spin.setRange(-2.0, 2.0)
        self.pres_penalty_spin.setValue(0.0)
        self.pres_penalty_spin.setSingleStep(0.1)
        layout.addWidget(self.pres_penalty_spin, 5, 1)

        length_penalty_label = QLabel("Length Penalty:")
        layout.addWidget(length_penalty_label, 6, 0)
        self.length_penalty_spin = QDoubleSpinBox()
        self.length_penalty_spin.setRange(0.0, 10.0)
        self.length_penalty_spin.setValue(1.0)
        self.length_penalty_spin.setSingleStep(0.1)
        layout.addWidget(self.length_penalty_spin, 6, 1)

        self.early_stopping_check = QCheckBox("Early Stopping")
        layout.addWidget(self.early_stopping_check, 7, 0, 1, 2)

        self.do_sample_check = QCheckBox("Do Sample")
        layout.addWidget(self.do_sample_check, 8, 0, 1, 2)

        return w

    def on_temp_spin_changed(self, val):
        """
        Sync the temperature spin box value to the slider.

        Args:
            val (float): The current value of the temperature spin box.
        """
        self.temp_slider.setValue(int(val * 100))

    def on_temp_slider_changed(self, val):
        """
        Sync the temperature slider value to the spin box.

        Args:
            val (int): The current value of the temperature slider.
        """
        self.temp_spin.setValue(val / 100.0)

    def create_advanced_optimizations_tab(self):
        """
        Create the "Optimizations" tab under "Advanced" settings, including quantization and various
        optimization toggles like gradient checkpointing, LoRA, flash attention, pruning, memory mapping,
        bfloat16, and torch compile.

        Returns:
            QWidget: The widget for advanced optimization settings.
        """
        w = QWidget()
        layout = QGridLayout(w)

        quant_label = QLabel("Quantization:")
        layout.addWidget(quant_label, 0, 0)
        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["None", "4bit", "8bit"])
        layout.addWidget(self.quant_combo, 0, 1)

        self.gc_check = QCheckBox("Gradient Checkpointing")
        layout.addWidget(self.gc_check, 1, 0, 1, 2)

        self.lora_check = QCheckBox("Low-Rank Adaptation (LoRA)")
        layout.addWidget(self.lora_check, 2, 0, 1, 2)

        self.fa_check = QCheckBox("FlashAttention Optimization")
        layout.addWidget(self.fa_check, 3, 0, 1, 2)

        self.pruning_check = QCheckBox("Model Pruning")
        layout.addWidget(self.pruning_check, 4, 0, 1, 2)

        self.mem_map_check = QCheckBox("Memory Mapping")
        layout.addWidget(self.mem_map_check, 5, 0, 1, 2)

        self.bfloat16_check = QCheckBox("Use bfloat16")
        layout.addWidget(self.bfloat16_check, 6, 0, 1, 2)

        self.torch_compile_check = QCheckBox("Torch Compile")
        layout.addWidget(self.torch_compile_check, 7, 0, 1, 2)

        return w

    def create_nvme_offloading_tab(self):
        """
        Create the "NVMe Offloading" tab under "Advanced", including toggles and inputs for NVMe path,
        buffer counts, block size, and queue depth.

        Returns:
            QWidget: The widget for NVMe offloading settings.
        """
        w = QWidget()
        layout = QGridLayout(w)

        self.enable_nvme_check = QCheckBox("Enable NVMe Offloading")
        layout.addWidget(self.enable_nvme_check, 0, 0, 1, 2)
        self.enable_nvme_check.toggled.connect(self.on_enable_nvme_toggled)

        nvme_path_label = QLabel("NVMe Path:")
        layout.addWidget(nvme_path_label, 1, 0)
        self.nvme_path_entry = QLineEdit()
        layout.addWidget(self.nvme_path_entry, 1, 1)

        nvme_path_button = QPushButton("Select NVMe Path")
        nvme_path_button.clicked.connect(self.on_nvme_path_button_clicked)
        layout.addWidget(nvme_path_button, 1, 2)

        buffer_param_label = QLabel("NVMe Buffer Count (Parameters):")
        layout.addWidget(buffer_param_label, 2, 0)
        self.buffer_param_spin = QSpinBox()
        self.buffer_param_spin.setRange(1, 10)
        self.buffer_param_spin.setValue(5)
        layout.addWidget(self.buffer_param_spin, 2, 1)

        buffer_opt_label = QLabel("NVMe Buffer Count (Optimizer):")
        layout.addWidget(buffer_opt_label, 3, 0)
        self.buffer_opt_spin = QSpinBox()
        self.buffer_opt_spin.setRange(1, 10)
        self.buffer_opt_spin.setValue(4)
        layout.addWidget(self.buffer_opt_spin, 3, 1)

        block_size_label = QLabel("NVMe Block Size (Bytes):")
        layout.addWidget(block_size_label, 4, 0)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(1024, 10485760)
        self.block_size_spin.setValue(1048576)
        layout.addWidget(self.block_size_spin, 4, 1)

        queue_depth_label = QLabel("NVMe Queue Depth:")
        layout.addWidget(queue_depth_label, 5, 0)
        self.queue_depth_spin = QSpinBox()
        self.queue_depth_spin.setRange(1,64)
        self.queue_depth_spin.setValue(8)
        layout.addWidget(self.queue_depth_spin, 5, 1)

        self.set_nvme_sensitive(False)

        return w

    def set_nvme_sensitive(self, sensitive):
        """
        Enable or disable NVMe-related input fields based on the given boolean.

        Args:
            sensitive (bool): True to enable fields, False to disable.
        """
        self.nvme_path_entry.setEnabled(sensitive)
        self.buffer_param_spin.setEnabled(sensitive)
        self.buffer_opt_spin.setEnabled(sensitive)
        self.block_size_spin.setEnabled(sensitive)
        self.queue_depth_spin.setEnabled(sensitive)

    def on_enable_nvme_toggled(self, checked):
        """
        Handle toggling of NVMe offloading enable checkbox.

        Args:
            checked (bool): True if NVMe offloading is enabled, False otherwise.
        """
        self.set_nvme_sensitive(checked)

    def on_nvme_path_button_clicked(self):
        """
        Open a directory chooser to select the NVMe path.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select NVMe Path")
        if folder:
            self.nvme_path_entry.setText(folder)

    def create_fine_tuning_settings_tab(self):
        """
        Create the "Fine-Tuning" tab, containing parameters such as number of epochs, batch size,
        learning rate, weight decay, save steps, save total limit, and layers to freeze.

        Returns:
            QWidget: The widget for fine-tuning settings.
        """
        w = QWidget()
        self.fine_tuning_layout = QGridLayout(w)

        epochs_label = QLabel("Number of Training Epochs:")
        self.fine_tuning_layout.addWidget(epochs_label, 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1,100)
        self.epochs_spin.setValue(3)
        self.fine_tuning_layout.addWidget(self.epochs_spin, 0, 1)

        batch_size_label = QLabel("Per Device Train Batch Size:")
        self.fine_tuning_layout.addWidget(batch_size_label, 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1,128)
        self.batch_size_spin.setValue(8)
        self.fine_tuning_layout.addWidget(self.batch_size_spin, 1, 1)

        lr_label = QLabel("Learning Rate:")
        self.fine_tuning_layout.addWidget(lr_label, 2, 0)
        self.lr_entry = QLineEdit("5e-5")
        lr_validator = QDoubleValidator(0.0, 1000.0, 5)
        lr_validator.setNotation(QDoubleValidator.StandardNotation)
        self.lr_entry.setValidator(lr_validator)
        self.fine_tuning_layout.addWidget(self.lr_entry, 2, 1)

        wd_label = QLabel("Weight Decay:")
        self.fine_tuning_layout.addWidget(wd_label, 3, 0)
        self.wd_entry = QLineEdit("0.01")
        wd_validator = QDoubleValidator(0.0, 1.0, 5)
        wd_validator.setNotation(QDoubleValidator.StandardNotation)
        self.wd_entry.setValidator(wd_validator)
        self.fine_tuning_layout.addWidget(self.wd_entry, 3, 1)

        save_steps_label = QLabel("Save Steps:")
        self.fine_tuning_layout.addWidget(save_steps_label, 4, 0)
        self.save_steps_spin = QSpinBox()
        self.save_steps_spin.setRange(100,10000)
        self.save_steps_spin.setValue(1000)
        self.fine_tuning_layout.addWidget(self.save_steps_spin, 4, 1)

        save_total_label = QLabel("Save Total Limit:")
        self.fine_tuning_layout.addWidget(save_total_label, 5, 0)
        self.save_total_spin = QSpinBox()
        self.save_total_spin.setRange(1,100)
        self.save_total_spin.setValue(2)
        self.fine_tuning_layout.addWidget(self.save_total_spin, 5, 1)

        layers_freeze_label = QLabel("Number of Layers to Freeze:")
        self.fine_tuning_layout.addWidget(layers_freeze_label, 6, 0)
        self.layers_freeze_spin = QSpinBox()
        self.layers_freeze_spin.setRange(0,100)
        self.layers_freeze_spin.setValue(0)
        self.fine_tuning_layout.addWidget(self.layers_freeze_spin, 6, 1)

        return w

    def create_model_management_tab(self):
        """
        Create the "Manage Models" tab, allowing the user to load/unload models, clear cache,
        and remove or update installed models.

        Returns:
            QWidget: The widget for model management settings.
        """
        w = QWidget()
        layout = QGridLayout(w)

        load_model_button = QPushButton("Load Model")
        load_model_button.clicked.connect(self.on_load_model_clicked)
        layout.addWidget(load_model_button, 0, 0)

        unload_model_button = QPushButton("Unload Model")
        unload_model_button.clicked.connect(self.on_unload_model_clicked)
        layout.addWidget(unload_model_button, 0, 1)

        model_sel_label = QLabel("Select Active Model:")
        layout.addWidget(model_sel_label, 1, 0)
        self.model_combo = QComboBox()
        layout.addWidget(self.model_combo, 1, 1, 1, 2)

        clear_cache_button = QPushButton("Clear Cache")
        clear_cache_button.clicked.connect(self.on_clear_cache_clicked)
        layout.addWidget(clear_cache_button, 2, 0, 1, 3)

        remove_model_label = QLabel("Remove Installed Model:")
        layout.addWidget(remove_model_label, 3, 0)
        self.remove_model_combo = QComboBox()
        layout.addWidget(self.remove_model_combo, 3, 1)
        remove_model_button = QPushButton("Remove Model")
        remove_model_button.clicked.connect(self.on_remove_model_clicked)
        layout.addWidget(remove_model_button, 3, 2)

        update_model_label = QLabel("Update Installed Model:")
        layout.addWidget(update_model_label, 4, 0)
        self.update_model_combo = QComboBox()
        layout.addWidget(self.update_model_combo, 4, 1)
        update_model_button = QPushButton("Update Model")
        update_model_button.clicked.connect(self.on_update_model_clicked)
        layout.addWidget(update_model_button, 4, 2)

        return w

    def create_miscellaneous_tab(self):
        """
        Create the "Misc" tab under "Advanced" for miscellaneous settings, such as enabling caching
        and setting the logging level.

        Returns:
            QWidget: The widget for miscellaneous settings.
        """
        w = QWidget()
        layout = QGridLayout(w)

        self.caching_check = QCheckBox("Enable Caching")
        self.caching_check.setChecked(True)
        layout.addWidget(self.caching_check, 0, 0, 1, 2)

        log_level_label = QLabel("Logging Level:")
        layout.addWidget(log_level_label, 1, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        layout.addWidget(self.log_level_combo, 1, 1)

        return w

    def create_search_download_tab(self):
        """
        Create the "Search & Download" tab under "Models", allowing the user to search the
        HuggingFace Hub for models that match certain criteria, and display the results.
        """
        w = QWidget()
        layout = QGridLayout(w)

        search_label = QLabel("Search Query:")
        layout.addWidget(search_label, 0, 0)
        self.search_entry = QLineEdit()
        layout.addWidget(self.search_entry, 0, 1)

        search_button = QPushButton("Search")
        search_button.clicked.connect(self.on_search_clicked)
        layout.addWidget(search_button, 0, 2)

        task_label = QLabel("Task:")
        layout.addWidget(task_label, 1, 0)
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Any","text-classification","question-answering","summarization","translation","text-generation","fill-mask"])
        layout.addWidget(self.task_combo, 1, 1)

        language_label = QLabel("Language:")
        layout.addWidget(language_label, 2, 0)
        self.language_entry = QLineEdit()
        layout.addWidget(self.language_entry, 2, 1)

        license_label = QLabel("License:")
        layout.addWidget(license_label, 3, 0)
        self.license_combo = QComboBox()
        self.license_combo.addItems(["Any","mit","apache-2.0","bsd-3-clause","bsd-2-clause","cc-by-sa-4.0","cc-by-4.0","wtfpl"])
        layout.addWidget(self.license_combo, 3, 1)

        library_label = QLabel("Library:")
        layout.addWidget(library_label, 4, 0)
        self.library_entry = QLineEdit()
        layout.addWidget(self.library_entry, 4, 1)

        results_label = QLabel("Search Results:")
        layout.addWidget(results_label, 5, 0, 1, 3)

        self.results_list = QListWidget()
        layout.addWidget(self.results_list, 6, 0, 1, 3)

        # Enable context menu on the results list
        self.results_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_list.customContextMenuRequested.connect(self.on_results_list_context_menu)

        return w

    def on_results_list_context_menu(self, pos: QPoint):
        item = self.results_list.itemAt(pos)
        if item is not None:
            menu = QMenu(self)
            download_action = QAction("Download", self)
            download_action.triggered.connect(lambda: self.download_model(item))
            menu.addAction(download_action)
            menu.exec_(self.results_list.mapToGlobal(pos))

    def download_model(self, item):
        model_id = item.data(Qt.UserRole)
        if not model_id:
            self.show_message("Error", "Could not determine model ID to download.", QMessageBox.Critical)
            return

        if self.confirm_dialog(f"Do you want to download the model '{model_id}'?"):
            try:
                # Ensure that your huggingface_generator has a method like:
                # await huggingface_generator.model_manager.load_model(model_id, force_download=True)
                asyncio.run(self.ATLAS.provider_manager.huggingface_generator.model_manager.load_model(model_id, force_download=True))
                self.show_message("Success", f"Model '{model_id}' downloaded successfully.")
                self.populate_model_comboboxes()
            except Exception as e:
                self.show_message("Error", f"Failed to download model '{model_id}': {e}", QMessageBox.Critical)

    def populate_model_comboboxes(self):
        """
        Populate the model selection comboboxes with installed models.
        """
        installed_models = self.ATLAS.provider_manager.huggingface_generator.get_installed_models()

        self.model_combo.clear()
        self.remove_model_combo.clear()
        self.update_model_combo.clear()

        if not installed_models:
            self.show_message("Info", "No installed models found. Please download a model.")
            return

        for model in installed_models:
            self.model_combo.addItem(model)
            self.remove_model_combo.addItem(model)
            self.update_model_combo.addItem(model)

        self.model_combo.setCurrentIndex(0)
        self.remove_model_combo.setCurrentIndex(0)
        self.update_model_combo.setCurrentIndex(0)

    def on_load_model_clicked(self):
        """
        Load the selected model from the model combobox.
        """
        selected_model = self.model_combo.currentText()
        if selected_model:
            asyncio.run(self.ATLAS.provider_manager.load_model(selected_model))
            self.show_message("Success", f"Model '{selected_model}' loaded successfully.")
        else:
            self.show_message("Error", "No model selected to load.", QMessageBox.Critical)

    def on_unload_model_clicked(self):
        """
        Unload the currently loaded model.
        """
        self.ATLAS.provider_manager.huggingface_generator.unload_model()
        self.show_message("Success", "Model unloaded successfully.")

    def on_fine_tune_clicked(self):
        """
        Handle the "Fine-Tune Model" button click (functionality not implemented).
        """
        self.show_message("Info", "Fine-tuning functionality is not yet implemented.")

    def on_save_clicked(self):
        """
        Handle the "Save Settings" button click, saving all settings and emitting the settingsSaved signal.
        """
        self.save_settings()
        self.show_message("Settings Saved", "Your settings have been saved successfully.")

    def on_cancel_clicked(self):
        """
        Handle the "Cancel" button click, closing the dialog without saving.
        """
        self.close()

    def on_back_clicked(self):
        """
        Handle the "Back" button click, closing the dialog and returning to previous view.
        """
        self.close()

    def on_clear_cache_clicked(self):
        if self.confirm_dialog("Are you sure you want to clear all cached files?"):
            try:
                self.ATLAS.provider_manager.huggingface_generator.clear_model_cache()
                self.show_message("Success", "All cache files cleared successfully.")
            except Exception as e:
                self.show_message("Error", f"Error clearing cache: {e}", QMessageBox.Critical)

    def on_remove_model_clicked(self):
        """
        Remove the selected installed model from the system if confirmed.
        """
        selected_model = self.remove_model_combo.currentText()
        if selected_model:
            if self.confirm_dialog(f"Are you sure you want to remove the model '{selected_model}'?"):
                self.ATLAS.provider_manager.huggingface_generator.model_manager.remove_installed_model(selected_model)
                self.populate_model_comboboxes()
                self.show_message("Success", f"Model '{selected_model}' removed successfully.")
        else:
            self.show_message("Error", "No model selected to remove.", QMessageBox.Critical)

    def on_update_model_clicked(self):
        """
        Update the selected installed model if confirmed.
        """
        selected_model = self.update_model_combo.currentText()
        if selected_model:
            if self.confirm_dialog(f"Do you want to update the model '{selected_model}'?"):
                asyncio.run(self.ATLAS.provider_manager.huggingface_generator.load_model(selected_model, force_download=True))
                self.show_message("Success", f"Model '{selected_model}' updated successfully.")
        else:
            self.show_message("Error", "No model selected to update.", QMessageBox.Critical)

    def on_search_clicked(self):
        """
        Perform a search on the HuggingFace Hub based on the entered criteria (query, task, language, license, library).
        """
        search_query = self.search_entry.text()
        task = self.task_combo.currentText()
        language = self.language_entry.text()
        license_txt = self.license_combo.currentText()
        library = self.library_entry.text()

        filter_args = {}
        if task and task != "Any":
            filter_args["pipeline_tag"] = task
        if language:
            filter_args["language"] = language
        if license_txt and license_txt != "Any":
            filter_args["license"] = license_txt
        if library:
            filter_args["library_name"] = library

        self.perform_search(search_query, filter_args)

    def perform_search(self, search_query, filter_args):
        """
        Perform the actual search query using the HuggingFace Hub API and display results.

        Args:
            search_query (str): The text to search for.
            filter_args (dict): Additional filtering parameters such as task, language, license, and library.
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(search=search_query, **filter_args)
            models = list(models)[:10]

            self.results_list.clear()

            if not models:
                self.results_list.addItem("No models found matching the criteria.")
            else:
                for model in models:
                    info_text = (f"Model ID: {model.modelId} | "
                                 f"Tags: {', '.join(model.tags)} | "
                                 f"Downloads: {model.downloads} | "
                                 f"Likes: {model.likes}")
                    item = self.results_list.addItem(info_text)
                    new_item = self.results_list.item(self.results_list.count()-1)
                    # Store model ID in user data for easy retrieval
                    new_item.setData(Qt.UserRole, model.modelId)
        except Exception as e:
            self.show_message("Error", f"An error occurred while searching for models: {str(e)}", QMessageBox.Critical)

    def save_settings(self):
        """
        Collect all settings from the UI and update the ATLAS configuration. Emit the settingsSaved signal afterwards.
        """
        try:
            settings = {
                "temperature": self.temp_spin.value(),
                "top_p": self.topp_spin.value(),
                "top_k": self.topk_spin.value(),
                "max_tokens": self.maxt_spin.value(),
                "repetition_penalty": self.rp_spin.value(),
                "presence_penalty": self.pres_penalty_spin.value(),
                "length_penalty": self.length_penalty_spin.value(),
                "early_stopping": self.early_stopping_check.isChecked(),
                "do_sample": self.do_sample_check.isChecked(),
                "quantization": self.quant_combo.currentText(),
                "use_gradient_checkpointing": self.gc_check.isChecked(),
                "use_lora": self.lora_check.isChecked(),
                "use_flash_attention": self.fa_check.isChecked(),
                "use_pruning": self.pruning_check.isChecked(),
                "use_memory_mapping": self.mem_map_check.isChecked(),
                "use_bfloat16": self.bfloat16_check.isChecked(),
                "use_torch_compile": self.torch_compile_check.isChecked(),
                "offload_nvme": self.enable_nvme_check.isChecked(),
                "nvme_path": self.nvme_path_entry.text(),
                "nvme_buffer_count_param": self.buffer_param_spin.value(),
                "nvme_buffer_count_optimizer": self.buffer_opt_spin.value(),
                "nvme_block_size": self.block_size_spin.value(),
                "nvme_queue_depth": self.queue_depth_spin.value(),
                "num_train_epochs": self.epochs_spin.value(),
                "per_device_train_batch_size": self.batch_size_spin.value(),
                "learning_rate": float(self.lr_entry.text()),
                "weight_decay": float(self.wd_entry.text()),
                "save_steps": self.save_steps_spin.value(),
                "save_total_limit": self.save_total_spin.value(),
                "num_layers_to_freeze": self.layers_freeze_spin.value(),
                "enable_caching": self.caching_check.isChecked(),
                "logging_level": self.log_level_combo.currentText(),
                "current_model": self.model_combo.currentText(),
            }

            # Call update_model_settings on the huggingface_generator
            self.ATLAS.provider_manager.huggingface_generator.update_model_settings(settings)

            # Emit the settingsSaved signal
            self.settingsSaved.emit(settings)

        except ValueError as ve:
            self.show_message("Invalid Input", f"Check field values: {ve}", QMessageBox.Critical)
        except Exception as e:
            self.show_message("Error", f"Error saving settings: {e}", QMessageBox.Critical)
