# /home/bib/Projects/ATLAS/UI/Provider_manager/Settings/HF_settings.py

import gi
import os
import asyncio
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

class HuggingFaceSettingsWindow(Gtk.Window):
    def __init__(self, ATLAS, config_manager):
        super().__init__(title="HuggingFace Settings")
        self.set_default_size(800, 600)
        self.ATLAS = ATLAS
        self.config_manager = config_manager

        # Create a vertical box to hold all widgets
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add(vbox)

        # Create a Notebook (Tabs)
        notebook = Gtk.Notebook()
        vbox.pack_start(notebook, True, True, 0)

        # General Settings Tab
        general_settings = self.create_general_settings_tab()
        notebook.append_page(general_settings, Gtk.Label(label="General Settings"))

        # Advanced Optimizations Tab
        advanced_optimizations = self.create_advanced_optimizations_tab()
        notebook.append_page(advanced_optimizations, Gtk.Label(label="Advanced Optimizations"))

        # NVMe Offloading Tab
        nvme_offloading = self.create_nvme_offloading_tab()
        notebook.append_page(nvme_offloading, Gtk.Label(label="NVMe Offloading"))

        # Fine-Tuning Settings Tab
        fine_tuning_settings = self.create_fine_tuning_settings_tab()
        notebook.append_page(fine_tuning_settings, Gtk.Label(label="Fine-Tuning Settings"))

        # Model Management Tab
        model_management = self.create_model_management_tab()
        notebook.append_page(model_management, Gtk.Label(label="Model Management"))

        # Miscellaneous Settings Tab
        miscellaneous = self.create_miscellaneous_tab()
        notebook.append_page(miscellaneous, Gtk.Label(label="Miscellaneous"))

        # Search and Download Models Tab
        search_download = self.create_search_download_tab()
        notebook.append_page(search_download, Gtk.Label(label="Search & Download Models"))

        # Action Buttons
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        vbox.pack_start(action_box, False, False, 10)

        load_model_button = Gtk.Button(label="Load Model")
        load_model_button.connect("clicked", self.on_load_model_clicked)
        action_box.pack_start(load_model_button, True, True, 0)

        unload_model_button = Gtk.Button(label="Unload Model")
        unload_model_button.connect("clicked", self.on_unload_model_clicked)
        action_box.pack_start(unload_model_button, True, True, 0)

        fine_tune_button = Gtk.Button(label="Fine-Tune Model")
        fine_tune_button.connect("clicked", self.on_fine_tune_clicked)
        action_box.pack_start(fine_tune_button, True, True, 0)

        # Control Buttons
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        vbox.pack_start(control_box, False, False, 10)

        back_button = Gtk.Button(label="Back")
        back_button.connect("clicked", self.on_back_clicked)
        control_box.pack_start(back_button, False, False, 0)

        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", self.on_cancel_clicked)
        control_box.pack_end(cancel_button, False, False, 0)

        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)
        control_box.pack_end(save_button, False, False, 0)

        # Populate model comboboxes
        self.populate_model_comboboxes()

    def create_general_settings_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Temperature
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_halign(Gtk.Align.START)
        grid.attach(temp_label, 0, 0, 1, 1)

        temp_adjustment = Gtk.Adjustment(value=0.7, lower=0.0, upper=2.0, step_increment=0.1, page_increment=0.5)
        self.temp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=temp_adjustment)
        self.temp_scale.set_digits(2)
        grid.attach(self.temp_scale, 1, 0, 1, 1)

        # Top-p
        topp_label = Gtk.Label(label="Top-p:")
        topp_label.set_halign(Gtk.Align.START)
        grid.attach(topp_label, 0, 1, 1, 1)

        topp_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=1.0, step_increment=0.05, page_increment=0.1)
        self.topp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=topp_adjustment)
        self.topp_scale.set_digits(2)
        grid.attach(self.topp_scale, 1, 1, 1, 1)

        # Top-k
        topk_label = Gtk.Label(label="Top-k:")
        topk_label.set_halign(Gtk.Align.START)
        grid.attach(topk_label, 0, 2, 1, 1)

        self.topk_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(50, 1, 1000, 1, 10, 0))
        grid.attach(self.topk_spin, 1, 2, 1, 1)

        # Max Tokens
        maxt_label = Gtk.Label(label="Max Tokens:")
        maxt_label.set_halign(Gtk.Align.START)
        grid.attach(maxt_label, 0, 3, 1, 1)

        self.maxt_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(100, 1, 2048, 1, 10, 0))
        grid.attach(self.maxt_spin, 1, 3, 1, 1)

        # Repetition Penalty
        rp_label = Gtk.Label(label="Repetition Penalty:")
        rp_label.set_halign(Gtk.Align.START)
        grid.attach(rp_label, 0, 4, 1, 1)

        self.rp_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(1.0, 0.0, 10.0, 0.1, 1.0, 0))
        grid.attach(self.rp_spin, 1, 4, 1, 1)

        # Presence Penalty
        pres_penalty_label = Gtk.Label(label="Presence Penalty:")
        pres_penalty_label.set_halign(Gtk.Align.START)
        grid.attach(pres_penalty_label, 0, 5, 1, 1)

        self.pres_penalty_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(0.0, -2.0, 2.0, 0.1, 0.5, 0))
        grid.attach(self.pres_penalty_spin, 1, 5, 1, 1)

        # Length Penalty
        length_penalty_label = Gtk.Label(label="Length Penalty:")
        length_penalty_label.set_halign(Gtk.Align.START)
        grid.attach(length_penalty_label, 0, 6, 1, 1)

        self.length_penalty_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(1.0, 0.0, 10.0, 0.1, 1.0, 0))
        grid.attach(self.length_penalty_spin, 1, 6, 1, 1)

        # Early Stopping
        self.early_stopping_check = Gtk.CheckButton(label="Early Stopping")
        grid.attach(self.early_stopping_check, 0, 7, 2, 1)

        # Do Sample
        self.do_sample_check = Gtk.CheckButton(label="Do Sample")
        grid.attach(self.do_sample_check, 0, 8, 2, 1)

        return grid

    def create_advanced_optimizations_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Quantization
        quant_label = Gtk.Label(label="Quantization:")
        quant_label.set_halign(Gtk.Align.START)
        grid.attach(quant_label, 0, 0, 1, 1)

        self.quant_combo = Gtk.ComboBoxText()
        self.quant_combo.append_text("None")
        self.quant_combo.append_text("4bit")
        self.quant_combo.append_text("8bit")
        self.quant_combo.set_active(0)
        grid.attach(self.quant_combo, 1, 0, 1, 1)

        # Gradient Checkpointing
        self.gc_check = Gtk.CheckButton(label="Gradient Checkpointing")
        grid.attach(self.gc_check, 0, 1, 2, 1)

        # Low-Rank Adaptation (LoRA)
        self.lora_check = Gtk.CheckButton(label="Low-Rank Adaptation (LoRA)")
        grid.attach(self.lora_check, 0, 2, 2, 1)

        # FlashAttention Optimization
        self.fa_check = Gtk.CheckButton(label="FlashAttention Optimization")
        grid.attach(self.fa_check, 0, 3, 2, 1)

        # Model Pruning
        self.pruning_check = Gtk.CheckButton(label="Model Pruning")
        grid.attach(self.pruning_check, 0, 4, 2, 1)

        # Memory Mapping
        self.mem_map_check = Gtk.CheckButton(label="Memory Mapping")
        grid.attach(self.mem_map_check, 0, 5, 2, 1)

        # Use bfloat16
        self.bfloat16_check = Gtk.CheckButton(label="Use bfloat16")
        grid.attach(self.bfloat16_check, 0, 6, 2, 1)

        # Torch Compile
        self.torch_compile_check = Gtk.CheckButton(label="Torch Compile")
        grid.attach(self.torch_compile_check, 0, 7, 2, 1)

        return grid

    def create_nvme_offloading_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Enable NVMe Offloading
        self.enable_nvme_check = Gtk.CheckButton(label="Enable NVMe Offloading")
        grid.attach(self.enable_nvme_check, 0, 0, 2, 1)
        self.enable_nvme_check.connect("toggled", self.on_enable_nvme_toggled)

        # NVMe Path
        nvme_path_label = Gtk.Label(label="NVMe Path:")
        nvme_path_label.set_halign(Gtk.Align.START)
        grid.attach(nvme_path_label, 0, 1, 1, 1)

        self.nvme_path_entry = Gtk.Entry()
        grid.attach(self.nvme_path_entry, 1, 1, 1, 1)

        nvme_path_button = Gtk.FileChooserButton(title="Select NVMe Path", action=Gtk.FileChooserAction.SELECT_FOLDER)
        nvme_path_button.connect("selection-changed", self.on_nvme_path_selected)
        grid.attach(nvme_path_button, 2, 1, 1, 1)

        # NVMe Buffer Count (Parameters)
        buffer_param_label = Gtk.Label(label="NVMe Buffer Count (Parameters):")
        buffer_param_label.set_halign(Gtk.Align.START)
        grid.attach(buffer_param_label, 0, 2, 1, 1)

        self.buffer_param_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(5, 1, 10, 1, 2, 0))
        grid.attach(self.buffer_param_spin, 1, 2, 1, 1)

        # NVMe Buffer Count (Optimizer)
        buffer_opt_label = Gtk.Label(label="NVMe Buffer Count (Optimizer):")
        buffer_opt_label.set_halign(Gtk.Align.START)
        grid.attach(buffer_opt_label, 0, 3, 1, 1)

        self.buffer_opt_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(4, 1, 10, 1, 2, 0))
        grid.attach(self.buffer_opt_spin, 1, 3, 1, 1)

        # NVMe Block Size
        block_size_label = Gtk.Label(label="NVMe Block Size (Bytes):")
        block_size_label.set_halign(Gtk.Align.START)
        grid.attach(block_size_label, 0, 4, 1, 1)

        self.block_size_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(1048576, 1024, 10485760, 1024, 10240, 0))
        grid.attach(self.block_size_spin, 1, 4, 1, 1)

        # NVMe Queue Depth
        queue_depth_label = Gtk.Label(label="NVMe Queue Depth:")
        queue_depth_label.set_halign(Gtk.Align.START)
        grid.attach(queue_depth_label, 0, 5, 1, 1)

        self.queue_depth_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(8, 1, 64, 1, 4, 0))
        grid.attach(self.queue_depth_spin, 1, 5, 1, 1)

        # Initially disable NVMe settings if not enabled
        self.set_nvme_sensitive(self.enable_nvme_check.get_active())

        return grid

    def create_fine_tuning_settings_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Number of Training Epochs
        epochs_label = Gtk.Label(label="Number of Training Epochs:")
        epochs_label.set_halign(Gtk.Align.START)
        grid.attach(epochs_label, 0, 0, 1, 1)

        self.epochs_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(3, 1, 100, 1, 10, 0))
        grid.attach(self.epochs_spin, 1, 0, 1, 1)

        # Per Device Train Batch Size
        batch_size_label = Gtk.Label(label="Per Device Train Batch Size:")
        batch_size_label.set_halign(Gtk.Align.START)
        grid.attach(batch_size_label, 0, 1, 1, 1)

        self.batch_size_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(8, 1, 128, 1, 8, 0))
        grid.attach(self.batch_size_spin, 1, 1, 1, 1)

        # Learning Rate
        lr_label = Gtk.Label(label="Learning Rate:")
        lr_label.set_halign(Gtk.Align.START)
        grid.attach(lr_label, 0, 2, 1, 1)

        self.lr_entry = Gtk.Entry(text="5e-5")
        grid.attach(self.lr_entry, 1, 2, 1, 1)

        # Weight Decay
        wd_label = Gtk.Label(label="Weight Decay:")
        wd_label.set_halign(Gtk.Align.START)
        grid.attach(wd_label, 0, 3, 1, 1)

        self.wd_entry = Gtk.Entry(text="0.01")
        grid.attach(self.wd_entry, 1, 3, 1, 1)

        # Save Steps
        save_steps_label = Gtk.Label(label="Save Steps:")
        save_steps_label.set_halign(Gtk.Align.START)
        grid.attach(save_steps_label, 0, 4, 1, 1)

        self.save_steps_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(1000, 100, 10000, 100, 1000, 0))
        grid.attach(self.save_steps_spin, 1, 4, 1, 1)

        # Save Total Limit
        save_total_label = Gtk.Label(label="Save Total Limit:")
        save_total_label.set_halign(Gtk.Align.START)
        grid.attach(save_total_label, 0, 5, 1, 1)

        self.save_total_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(2, 1, 100, 1, 5, 0))
        grid.attach(self.save_total_spin, 1, 5, 1, 1)

        # Number of Layers to Freeze
        layers_freeze_label = Gtk.Label(label="Number of Layers to Freeze:")
        layers_freeze_label.set_halign(Gtk.Align.START)
        grid.attach(layers_freeze_label, 0, 6, 1, 1)

        self.layers_freeze_spin = Gtk.SpinButton(adjustment=Gtk.Adjustment(0, 0, 100, 1, 5, 0))
        grid.attach(self.layers_freeze_spin, 1, 6, 1, 1)

        return grid

    def create_model_management_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Model Selection
        model_sel_label = Gtk.Label(label="Select Active Model:")
        model_sel_label.set_halign(Gtk.Align.START)
        grid.attach(model_sel_label, 0, 0, 1, 1)

        self.model_combo = Gtk.ComboBoxText()
        grid.attach(self.model_combo, 1, 0, 1, 1)

        # Clear Cache
        clear_cache_button = Gtk.Button(label="Clear Cache")
        clear_cache_button.connect("clicked", self.on_clear_cache_clicked)
        grid.attach(clear_cache_button, 0, 1, 2, 1)

        # Remove Installed Model
        remove_model_label = Gtk.Label(label="Remove Installed Model:")
        remove_model_label.set_halign(Gtk.Align.START)
        grid.attach(remove_model_label, 0, 2, 1, 1)

        self.remove_model_combo = Gtk.ComboBoxText()
        grid.attach(self.remove_model_combo, 1, 2, 1, 1)

        remove_model_button = Gtk.Button(label="Remove Model")
        remove_model_button.connect("clicked", self.on_remove_model_clicked)
        grid.attach(remove_model_button, 2, 2, 1, 1)

        # Update Installed Model
        update_model_label = Gtk.Label(label="Update Installed Model:")
        update_model_label.set_halign(Gtk.Align.START)
        grid.attach(update_model_label, 0, 3, 1, 1)

        self.update_model_combo = Gtk.ComboBoxText()
        grid.attach(self.update_model_combo, 1, 3, 1, 1)

        update_model_button = Gtk.Button(label="Update Model")
        update_model_button.connect("clicked", self.on_update_model_clicked)
        grid.attach(update_model_button, 2, 3, 1, 1)

        return grid

    def create_miscellaneous_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Enable Caching
        self.caching_check = Gtk.CheckButton(label="Enable Caching")
        self.caching_check.set_active(True)
        grid.attach(self.caching_check, 0, 0, 2, 1)

        # Logging Level
        log_level_label = Gtk.Label(label="Logging Level:")
        log_level_label.set_halign(Gtk.Align.START)
        grid.attach(log_level_label, 0, 1, 1, 1)

        self.log_level_combo = Gtk.ComboBoxText()
        self.log_level_combo.append_text("DEBUG")
        self.log_level_combo.append_text("INFO")
        self.log_level_combo.append_text("WARNING")
        self.log_level_combo.append_text("ERROR")
        self.log_level_combo.append_text("CRITICAL")
        self.log_level_combo.set_active(1)  # Default to INFO
        grid.attach(self.log_level_combo, 1, 1, 1, 1)

        return grid

    def create_search_download_tab(self):
        grid = Gtk.Grid(column_spacing=10, row_spacing=10, margin=10)

        # Search Query
        search_label = Gtk.Label(label="Search Query:")
        search_label.set_halign(Gtk.Align.START)
        grid.attach(search_label, 0, 0, 1, 1)

        self.search_entry = Gtk.Entry()
        grid.attach(self.search_entry, 1, 0, 2, 1)

        # Filters
        # Task Filter
        task_label = Gtk.Label(label="Task:")
        task_label.set_halign(Gtk.Align.START)
        grid.attach(task_label, 0, 1, 1, 1)

        self.task_combo = Gtk.ComboBoxText()
        self.task_combo.append_text("Any")
        tasks = ["text-classification", "question-answering", "summarization", "translation", "text-generation", "fill-mask"]
        for task in tasks:
            self.task_combo.append_text(task)
        self.task_combo.set_active(0)
        grid.attach(self.task_combo, 1, 1, 1, 1)

        # Language Filter
        language_label = Gtk.Label(label="Language:")
        language_label.set_halign(Gtk.Align.START)
        grid.attach(language_label, 0, 2, 1, 1)

        self.language_entry = Gtk.Entry()
        grid.attach(self.language_entry, 1, 2, 1, 1)

        # License Filter
        license_label = Gtk.Label(label="License:")
        license_label.set_halign(Gtk.Align.START)
        grid.attach(license_label, 0, 3, 1, 1)

        self.license_combo = Gtk.ComboBoxText()
        self.license_combo.append_text("Any")
        licenses = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc-by-sa-4.0", "cc-by-4.0", "wtfpl"]
        for lic in licenses:
            self.license_combo.append_text(lic)
        self.license_combo.set_active(0)
        grid.attach(self.license_combo, 1, 3, 1, 1)

        # Library Filter
        library_label = Gtk.Label(label="Library:")
        library_label.set_halign(Gtk.Align.START)
        grid.attach(library_label, 0, 4, 1, 1)

        self.library_entry = Gtk.Entry()
        grid.attach(self.library_entry, 1, 4, 1, 1)

        # Search Button
        search_button = Gtk.Button(label="Search")
        search_button.connect("clicked", self.on_search_clicked)
        grid.attach(search_button, 2, 0, 1, 1)

        # Search Results
        results_label = Gtk.Label(label="Search Results:")
        results_label.set_halign(Gtk.Align.START)
        grid.attach(results_label, 0, 5, 3, 1)

        self.results_listbox = Gtk.ListBox()
        self.results_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.add(self.results_listbox)
        grid.attach(scroll, 0, 6, 3, 1)

        return grid

    def set_nvme_sensitive(self, sensitive):
        # Toggle sensitivity of NVMe-related widgets based on the checkbox
        self.nvme_path_entry.set_sensitive(sensitive)
        self.buffer_param_spin.set_sensitive(sensitive)
        self.buffer_opt_spin.set_sensitive(sensitive)
        self.block_size_spin.set_sensitive(sensitive)
        self.queue_depth_spin.set_sensitive(sensitive)

    def on_enable_nvme_toggled(self, widget):
        self.set_nvme_sensitive(widget.get_active())

    def on_nvme_path_selected(self, widget):
        selected_folder = widget.get_filename()
        if selected_folder:
            self.nvme_path_entry.set_text(selected_folder)

    def populate_model_comboboxes(self):
        # Access the installed models via huggingface_generator
        installed_models = self.ATLAS.provider_manager.huggingface_generator.get_installed_models()

        self.model_combo.remove_all()
        self.remove_model_combo.remove_all()
        self.update_model_combo.remove_all()

        if not installed_models:
            # Handle the case when no models are installed
            self.show_message("Info", "No installed models found. Please download a model.")
            return

        for model in installed_models:
            self.model_combo.append_text(model)
            self.remove_model_combo.append_text(model)
            self.update_model_combo.append_text(model)

        self.model_combo.set_active(0)
        self.remove_model_combo.set_active(0)
        self.update_model_combo.set_active(0)

    # Placeholder callback methods
    def on_load_model_clicked(self, widget):
        selected_model = self.model_combo.get_active_text()
        if selected_model:
            asyncio.run(self.ATLAS.provider_manager.load_model(selected_model))
            self.show_message("Success", f"Model '{selected_model}' loaded successfully.")
        else:
            self.show_message("Error", "No model selected to load.")

    def on_unload_model_clicked(self, widget):
        self.ATLAS.provider_manager.huggingface_generator.unload_model()
        self.show_message("Success", "Model unloaded successfully.")

    def on_fine_tune_clicked(self, widget):
        # Implement fine-tuning functionality
        self.show_message("Info", "Fine-tuning functionality is not yet implemented.")

    def on_save_clicked(self, widget):
        # Implement settings saving functionality
        self.save_settings()
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Settings Saved",
        )
        dialog.format_secondary_text(
            "Your settings have been saved successfully."
        )
        dialog.run()
        dialog.destroy()

    def on_cancel_clicked(self, widget):
        # Implement settings cancel functionality
        self.destroy()

    def on_back_clicked(self, widget):
        # Implement back navigation functionality
        self.destroy()

    def on_clear_cache_clicked(self, widget):
        # Implement cache clearing functionality
        cache_file = self.ATLAS.base_config.cache_file
        confirmation = self.confirm_dialog(f"Are you sure you want to clear the cache at {cache_file}?")
        if confirmation:
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    self.show_message("Success", "Cache cleared successfully.")
                else:
                    self.show_message("Info", "Cache file does not exist.")
            except Exception as e:
                self.show_message("Error", f"Error clearing cache: {str(e)}")

    def on_remove_model_clicked(self, widget):
        selected_model = self.remove_model_combo.get_active_text()
        if selected_model:
            confirmation = self.confirm_dialog(f"Are you sure you want to remove the model '{selected_model}'?")
            if confirmation:
                self.ATLAS.provider_manager.huggingface_generator.model_manager.remove_installed_model(selected_model)
                self.populate_model_comboboxes()
                self.show_message("Success", f"Model '{selected_model}' removed successfully.")
        else:
            self.show_message("Error", "No model selected to remove.")

    def on_update_model_clicked(self, widget):
        selected_model = self.update_model_combo.get_active_text()
        if selected_model:
            confirmation = self.confirm_dialog(f"Do you want to update the model '{selected_model}'?")
            if confirmation:
                asyncio.run(self.ATLAS.provider_manager.huggingface_generator.load_model(selected_model, force_download=True))
                self.show_message("Success", f"Model '{selected_model}' updated successfully.")
        else:
            self.show_message("Error", "No model selected to update.")

    def on_search_clicked(self, widget):
        search_query = self.search_entry.get_text()
        task = self.task_combo.get_active_text()
        language = self.language_entry.get_text()
        license = self.license_combo.get_active_text()
        library = self.library_entry.get_text()

        filter_args = {}
        if task and task != "Any":
            filter_args["pipeline_tag"] = task
        if language:
            filter_args["language"] = language
        if license and license != "Any":
            filter_args["license"] = license
        if library:
            filter_args["library_name"] = library

        # Perform search asynchronously
        GLib.idle_add(self.perform_search, search_query, filter_args)

    def perform_search(self, search_query, filter_args):
        try:
            # Fetch models using the HfApi
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(search=search_query, **filter_args)
            models = list(models)[:10]  # Limit to 10 results for display

            # Clear previous results
            for row in self.results_listbox.get_children():
                self.results_listbox.remove(row)

            if not models:
                label = Gtk.Label(label="No models found matching the criteria.")
                self.results_listbox.add(label)
            else:
                for model in models:
                    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
                    box.set_border_width(5)

                    info_label = Gtk.Label()
                    info_label.set_xalign(0)
                    info_text = f"Model ID: {model.modelId}\nTags: {model.tags}\nDownloads: {model.downloads}\nLikes: {model.likes}"
                    info_label.set_text(info_text)
                    box.pack_start(info_label, True, True, 0)

                    download_button = Gtk.Button(label="Download")
                    download_button.connect("clicked", self.on_download_model_clicked, model.modelId)
                    box.pack_start(download_button, False, False, 0)

                    self.results_listbox.add(box)

            self.results_listbox.show_all()
        except Exception as e:
            self.show_message("Error", f"An error occurred while searching for models: {str(e)}")

        return False  # Stop the idle_add

    def on_download_model_clicked(self, widget, model_name):
        confirmation = self.confirm_dialog(f"Do you want to download and install the model '{model_name}'?")
        if confirmation:
            try:
                # Start the download asynchronously
                GLib.idle_add(self.download_model, model_name)
            except Exception as e:
                self.show_message("Error", f"Error downloading model: {str(e)}")

    def download_model(self, model_name):
        try:
            # Trigger the model download using the provider manager
            asyncio.run(self.ATLAS.provider_manager.huggingface_generator.load_model(model_name, force_download=True))
            self.populate_model_comboboxes()
            self.show_message("Success", f"Model '{model_name}' downloaded and installed successfully.")
        except Exception as e:
            self.show_message("Error", f"Error downloading model '{model_name}': {str(e)}")

        return False  # Stop the idle_add

    def save_settings(self):
        # Gather all settings from the UI and save them using the config manager
        settings = {
            "temperature": self.temp_scale.get_value(),
            "top_p": self.topp_scale.get_value(),
            "top_k": self.topk_spin.get_value_as_int(),
            "max_tokens": self.maxt_spin.get_value_as_int(),
            "repetition_penalty": self.rp_spin.get_value(),
            "presence_penalty": self.pres_penalty_spin.get_value(),
            "length_penalty": self.length_penalty_spin.get_value(),
            "early_stopping": self.early_stopping_check.get_active(),
            "do_sample": self.do_sample_check.get_active(),
            "quantization": self.quant_combo.get_active_text(),
            "use_gradient_checkpointing": self.gc_check.get_active(),
            "use_lora": self.lora_check.get_active(),
            "use_flash_attention": self.fa_check.get_active(),
            "use_pruning": self.pruning_check.get_active(),
            "use_memory_mapping": self.mem_map_check.get_active(),
            "use_bfloat16": self.bfloat16_check.get_active(),
            "use_torch_compile": self.torch_compile_check.get_active(),
            "offload_nvme": self.enable_nvme_check.get_active(),
            "nvme_path": self.nvme_path_entry.get_text(),
            "nvme_buffer_count_param": self.buffer_param_spin.get_value_as_int(),
            "nvme_buffer_count_optimizer": self.buffer_opt_spin.get_value_as_int(),
            "nvme_block_size": self.block_size_spin.get_value_as_int(),
            "nvme_queue_depth": self.queue_depth_spin.get_value_as_int(),
            "num_train_epochs": self.epochs_spin.get_value_as_int(),
            "per_device_train_batch_size": self.batch_size_spin.get_value_as_int(),
            "learning_rate": float(self.lr_entry.get_text()),
            "weight_decay": float(self.wd_entry.get_text()),
            "save_steps": self.save_steps_spin.get_value_as_int(),
            "save_total_limit": self.save_total_spin.get_value_as_int(),
            "num_layers_to_freeze": self.layers_freeze_spin.get_value_as_int(),
            "enable_caching": self.caching_check.get_active(),
            "logging_level": self.log_level_combo.get_active_text(),
            "current_model": self.model_combo.get_active_text(),
        }

        # Save settings using the config manager
        self.ATLAS.base_config.update_model_settings(settings)
        self.show_message("Success", "Settings saved successfully.")

    def confirm_dialog(self, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.YES_NO,
            text=message,
        )
        response = dialog.run()
        dialog.destroy()
        return response == Gtk.ResponseType.YES

    def show_message(self, title, message):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text=title,
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def run(self):
        self.show_all()
