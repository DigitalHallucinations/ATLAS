# GTKUI/Provider_manager/Settings/HF_settings.py

"""HuggingFaceSettingsWindow – A complete settings window for the HuggingFace provider.

This GTK 4 window implements a complete enterprise‐level user interface for configuring
the HuggingFace provider settings. It uses a notebook for multiple tabs including General,
Advanced (with sub-tabs for Optimizations, NVMe Offloading and Miscellaneous), Fine‐Tuning,
and Model Management (with a Search & Download sub‐tab).

All asynchronous operations (such as model load/update/download calls) are executed via
a dedicated helper (_run_async) that runs the provided coroutine in a separate thread
and schedules UI updates on the GTK main loop via GLib.idle_add.
"""

import gi
import os
import asyncio
import threading
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, GLib

from GTKUI.Utils.utils import create_box

class HuggingFaceSettingsWindow(Gtk.Window):
    def __init__(self, ATLAS, config_manager, parent_window):
        """
        Initialize the HuggingFaceSettingsWindow.
        
        Args:
            ATLAS: The main ATLAS application instance.
            config_manager: The configuration manager instance.
            parent_window: The parent GTK window.
        """
        super().__init__(title="HuggingFace Settings")
        self.parent_window = parent_window
        self.set_transient_for(parent_window)
        self.set_modal(True)
        
        # Apply the CSS styling for this window.
        self.apply_css_styling()
        
        self.set_default_size(800, 600)
        self.ATLAS = ATLAS
        self.config_manager = config_manager

        # Create a vertical box (with uniform margins) to hold all widgets.
        vbox = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        self.set_child(vbox)
        
        # Create a Notebook widget to hold the different settings tabs.
        notebook = Gtk.Notebook()
        vbox.append(notebook)
        
        # General Settings Tab
        general_settings = self.create_general_settings_tab()
        notebook.append_page(general_settings, Gtk.Label(label="General"))
        
        # Advanced Tab – with sub-notebook for Optimizations, NVMe Offloading and Miscellaneous.
        advanced_optimizations_notebook = Gtk.Notebook()
        advanced_optimizations = self.create_advanced_optimizations_tab()
        nvme_offloading = self.create_nvme_offloading_tab()
        miscellaneous = self.create_miscellaneous_tab()
        advanced_optimizations_notebook.append_page(advanced_optimizations, Gtk.Label(label="Optimizations"))
        advanced_optimizations_notebook.append_page(nvme_offloading, Gtk.Label(label="NVMe Offloading"))
        advanced_optimizations_notebook.append_page(miscellaneous, Gtk.Label(label="Misc"))
        notebook.append_page(advanced_optimizations_notebook, Gtk.Label(label="Advanced"))
        
        # Fine-Tuning Settings Tab – attach the Fine-Tune Model button directly into the grid.
        fine_tuning_settings = self.create_fine_tuning_settings_tab()
        fine_tuning_grid = fine_tuning_settings  # The grid is already created.
        fine_tune_button = Gtk.Button(label="Fine-Tune Model")
        fine_tune_button.connect("clicked", self.on_fine_tune_clicked)
        fine_tuning_grid.attach(fine_tune_button, 0, 7, 2, 1)
        notebook.append_page(fine_tuning_settings, Gtk.Label(label="Fine-Tuning"))
        
        # Model Management Tab – contains a sub-notebook for Manage Models and Search & Download.
        models_notebook = Gtk.Notebook()
        model_management = self.create_model_management_tab()
        # Adjust button positions for Load and Unload Model Buttons.
        load_model_button = Gtk.Button(label="Load Model")
        load_model_button.connect("clicked", self.on_load_model_clicked)
        unload_model_button = Gtk.Button(label="Unload Model")
        unload_model_button.connect("clicked", self.on_unload_model_clicked)
        model_management_grid = model_management
        model_management_grid.attach(load_model_button, 0, 0, 1, 1)
        model_management_grid.attach(unload_model_button, 1, 0, 1, 1)
        search_download = self.create_search_download_tab()
        models_notebook.append_page(model_management, Gtk.Label(label="Manage Models"))
        models_notebook.append_page(search_download, Gtk.Label(label="Search & Download"))
        notebook.append_page(models_notebook, Gtk.Label(label="Models"))
        
        # Control Buttons at the bottom.
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        vbox.append(control_box)
        back_button = Gtk.Button(label="Back")
        back_button.connect("clicked", self.on_back_clicked)
        control_box.append(back_button)
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.connect("clicked", self.on_cancel_clicked)
        control_box.append(cancel_button)
        save_button = Gtk.Button(label="Save Settings")
        save_button.connect("clicked", self.on_save_clicked)
        control_box.append(save_button)
        
        # Populate model comboboxes with installed models.
        self.populate_model_comboboxes()

    def _run_async(self, coro, success_callback=None, error_callback=None):
        """
        Runs an asynchronous coroutine in a separate thread to avoid blocking the UI.

        Args:
            coro (coroutine): The coroutine to execute.
            success_callback (callable, optional): A function to call on successful completion.
                                                   It will be called with the result of the coroutine.
            error_callback (callable, optional): A function to call if an exception occurs.
                                                 It will be called with the exception instance.
        """
        def runner():
            try:
                # Create a new event loop and run the coroutine.
                result = asyncio.run(coro)
                if success_callback:
                    GLib.idle_add(success_callback, result)
            except Exception as e:
                if error_callback:
                    GLib.idle_add(error_callback, e)
                else:
                    GLib.idle_add(self.show_message, "Error", f"An error occurred: {str(e)}", Gtk.MessageType.ERROR)
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

    def apply_css_styling(self):
        """
        Applies CSS styling to the HuggingFace settings window.
        """
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(b"""
            * { 
                background-color: #2b2b2b; 
                color: white; 
            }

            entry, textview {
                background-color: #1c1c1c;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                padding: 5px;
                margin: 0;
            }

            entry {
                min-height: 30px;
            }

            entry:focus {
                outline: none;
            }

            textview {
                caret-color: white;
            }

            textview text {
                background-color: #1c1c1c;
                color: white;
                caret-color: white;
            }

            textview text selection {
                background-color: #4a90d9;
                color: white;
            }

            button {
                background-color: #2b2b2b;
                color: white;
                padding: 8px;
                border-radius: 3px;
                border: 1px solid #4a4a4a;
                font-size: 14px;
            }

            button:hover {
                background-color: #4a90d9;
                border: 1px solid #4a90d9;
            }

            button:active {
                background-color: #357ABD;
                border: 1px solid #357ABD;
            }

            label { 
                margin: 5px;
                color: #ffffff; 
            }

            notebook tab { 
                background-color: #2b2b2b; 
                color: white; 
                padding: 8px; 
            }

            scrolledwindow {
                border: none;
                background-color: transparent;
            }
        """)
        # Use the display from a new temporary window to obtain the proper display instance.
        display = Gtk.Window().get_display()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_USER
        )

    def create_general_settings_tab(self):
        """
        Creates and returns the General Settings tab containing model generation parameters.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # Temperature setting
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_halign(Gtk.Align.START)
        grid.attach(temp_label, 0, 0, 1, 1)
        temp_adjustment = Gtk.Adjustment(value=0.7, lower=0.0, upper=2.0, step_increment=0.1, page_increment=0.5)
        self.temp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=temp_adjustment)
        self.temp_scale.set_digits(2)
        grid.attach(self.temp_scale, 1, 0, 1, 1)

        # Top-p setting
        topp_label = Gtk.Label(label="Top-p:")
        topp_label.set_halign(Gtk.Align.START)
        grid.attach(topp_label, 0, 1, 1, 1)
        topp_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=1.0, step_increment=0.05, page_increment=0.1)
        self.topp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=topp_adjustment)
        self.topp_scale.set_digits(2)
        grid.attach(self.topp_scale, 1, 1, 1, 1)

        # Top-k setting
        topk_label = Gtk.Label(label="Top-k:")
        topk_label.set_halign(Gtk.Align.START)
        grid.attach(topk_label, 0, 2, 1, 1)
        topk_adjustment = Gtk.Adjustment(value=50, lower=1, upper=1000, step_increment=1, page_increment=10)
        self.topk_spin = Gtk.SpinButton(adjustment=topk_adjustment)
        grid.attach(self.topk_spin, 1, 2, 1, 1)

        # Max Tokens setting
        maxt_label = Gtk.Label(label="Max Tokens:")
        maxt_label.set_halign(Gtk.Align.START)
        grid.attach(maxt_label, 0, 3, 1, 1)
        maxt_adjustment = Gtk.Adjustment(value=100, lower=1, upper=2048, step_increment=1, page_increment=10)
        self.maxt_spin = Gtk.SpinButton(adjustment=maxt_adjustment)
        grid.attach(self.maxt_spin, 1, 3, 1, 1)

        # Repetition Penalty setting
        rp_label = Gtk.Label(label="Repetition Penalty:")
        rp_label.set_halign(Gtk.Align.START)
        grid.attach(rp_label, 0, 4, 1, 1)
        rp_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=10.0, step_increment=0.1, page_increment=1.0)
        self.rp_spin = Gtk.SpinButton(adjustment=rp_adjustment, digits=2)
        grid.attach(self.rp_spin, 1, 4, 1, 1)

        # Presence Penalty setting
        pres_penalty_label = Gtk.Label(label="Presence Penalty:")
        pres_penalty_label.set_halign(Gtk.Align.START)
        grid.attach(pres_penalty_label, 0, 5, 1, 1)
        pres_penalty_adjustment = Gtk.Adjustment(value=0.0, lower=-2.0, upper=2.0, step_increment=0.1, page_increment=0.5)
        self.pres_penalty_spin = Gtk.SpinButton(adjustment=pres_penalty_adjustment, digits=2)
        grid.attach(self.pres_penalty_spin, 1, 5, 1, 1)

        # Length Penalty setting
        length_penalty_label = Gtk.Label(label="Length Penalty:")
        length_penalty_label.set_halign(Gtk.Align.START)
        grid.attach(length_penalty_label, 0, 6, 1, 1)
        length_penalty_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=10.0, step_increment=0.1, page_increment=1.0)
        self.length_penalty_spin = Gtk.SpinButton(adjustment=length_penalty_adjustment, digits=2)
        grid.attach(self.length_penalty_spin, 1, 6, 1, 1)

        # Early Stopping checkbutton
        self.early_stopping_check = Gtk.CheckButton(label="Early Stopping")
        grid.attach(self.early_stopping_check, 0, 7, 2, 1)

        # Do Sample checkbutton
        self.do_sample_check = Gtk.CheckButton(label="Do Sample")
        grid.attach(self.do_sample_check, 0, 8, 2, 1)

        return grid

    def create_advanced_optimizations_tab(self):
        """
        Creates and returns the Advanced Optimizations tab containing optimization switches.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

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

        # Gradient Checkpointing switch
        self.gc_check = Gtk.CheckButton(label="Gradient Checkpointing")
        grid.attach(self.gc_check, 0, 1, 2, 1)

        # Low-Rank Adaptation (LoRA) switch
        self.lora_check = Gtk.CheckButton(label="Low-Rank Adaptation (LoRA)")
        grid.attach(self.lora_check, 0, 2, 2, 1)

        # FlashAttention Optimization switch
        self.fa_check = Gtk.CheckButton(label="FlashAttention Optimization")
        grid.attach(self.fa_check, 0, 3, 2, 1)

        # Model Pruning switch
        self.pruning_check = Gtk.CheckButton(label="Model Pruning")
        grid.attach(self.pruning_check, 0, 4, 2, 1)

        # Memory Mapping switch
        self.mem_map_check = Gtk.CheckButton(label="Memory Mapping")
        grid.attach(self.mem_map_check, 0, 5, 2, 1)

        # Use bfloat16 switch
        self.bfloat16_check = Gtk.CheckButton(label="Use bfloat16")
        grid.attach(self.bfloat16_check, 0, 6, 2, 1)

        # Torch Compile switch
        self.torch_compile_check = Gtk.CheckButton(label="Torch Compile")
        grid.attach(self.torch_compile_check, 0, 7, 2, 1)

        return grid

    def create_nvme_offloading_tab(self):
        """
        Creates and returns the NVMe Offloading tab which allows selection of NVMe parameters.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # Enable NVMe Offloading switch
        self.enable_nvme_check = Gtk.CheckButton(label="Enable NVMe Offloading")
        grid.attach(self.enable_nvme_check, 0, 0, 2, 1)
        self.enable_nvme_check.connect("toggled", self.on_enable_nvme_toggled)

        # NVMe Path entry and selection button
        nvme_path_label = Gtk.Label(label="NVMe Path:")
        nvme_path_label.set_halign(Gtk.Align.START)
        grid.attach(nvme_path_label, 0, 1, 1, 1)
        self.nvme_path_entry = Gtk.Entry()
        grid.attach(self.nvme_path_entry, 1, 1, 1, 1)
        nvme_path_button = Gtk.Button(label="Select NVMe Path")
        nvme_path_button.connect("clicked", self.on_nvme_path_button_clicked)
        grid.attach(nvme_path_button, 2, 1, 1, 1)

        # NVMe Buffer Count (Parameters)
        buffer_param_label = Gtk.Label(label="NVMe Buffer Count (Parameters):")
        buffer_param_label.set_halign(Gtk.Align.START)
        grid.attach(buffer_param_label, 0, 2, 1, 1)
        buffer_param_adjustment = Gtk.Adjustment(value=5, lower=1, upper=10, step_increment=1, page_increment=2)
        self.buffer_param_spin = Gtk.SpinButton(adjustment=buffer_param_adjustment)
        grid.attach(self.buffer_param_spin, 1, 2, 1, 1)

        # NVMe Buffer Count (Optimizer)
        buffer_opt_label = Gtk.Label(label="NVMe Buffer Count (Optimizer):")
        buffer_opt_label.set_halign(Gtk.Align.START)
        grid.attach(buffer_opt_label, 0, 3, 1, 1)
        buffer_opt_adjustment = Gtk.Adjustment(value=4, lower=1, upper=10, step_increment=1, page_increment=2)
        self.buffer_opt_spin = Gtk.SpinButton(adjustment=buffer_opt_adjustment)
        grid.attach(self.buffer_opt_spin, 1, 3, 1, 1)

        # NVMe Block Size (Bytes)
        block_size_label = Gtk.Label(label="NVMe Block Size (Bytes):")
        block_size_label.set_halign(Gtk.Align.START)
        grid.attach(block_size_label, 0, 4, 1, 1)
        block_size_adjustment = Gtk.Adjustment(value=1048576, lower=1024, upper=10485760, step_increment=1024, page_increment=10240)
        self.block_size_spin = Gtk.SpinButton(adjustment=block_size_adjustment)
        grid.attach(self.block_size_spin, 1, 4, 1, 1)

        # NVMe Queue Depth
        queue_depth_label = Gtk.Label(label="NVMe Queue Depth:")
        queue_depth_label.set_halign(Gtk.Align.START)
        grid.attach(queue_depth_label, 0, 5, 1, 1)
        queue_depth_adjustment = Gtk.Adjustment(value=8, lower=1, upper=64, step_increment=1, page_increment=4)
        self.queue_depth_spin = Gtk.SpinButton(adjustment=queue_depth_adjustment)
        grid.attach(self.queue_depth_spin, 1, 5, 1, 1)

        # Set the sensitivity of NVMe settings based on the switch state.
        self.set_nvme_sensitive(self.enable_nvme_check.get_active())

        return grid

    def on_nvme_path_button_clicked(self, widget):
        """
        Opens a file chooser dialog for selecting an NVMe folder.
        """
        dialog = Gtk.FileChooserDialog(
            title="Select NVMe Path",
            transient_for=self.parent_window,
            modal=True,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.add_buttons(
            "_Cancel", Gtk.ResponseType.CANCEL,
            "_Open", Gtk.ResponseType.OK
        )
        dialog.connect("response", self.on_nvme_dialog_response)
        dialog.present()

    def on_nvme_dialog_response(self, dialog, response):
        """
        Callback for the NVMe path dialog. Sets the NVMe path entry if a folder is selected.
        """
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                path = folder.get_path()
                self.nvme_path_entry.set_text(path)
        dialog.destroy()

    def create_fine_tuning_settings_tab(self):
        """
        Creates and returns the Fine-Tuning Settings tab with training parameters.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # Number of Training Epochs
        epochs_label = Gtk.Label(label="Number of Training Epochs:")
        epochs_label.set_halign(Gtk.Align.START)
        grid.attach(epochs_label, 0, 0, 1, 1)
        epochs_adjustment = Gtk.Adjustment(value=3, lower=1, upper=100, step_increment=1, page_increment=10)
        self.epochs_spin = Gtk.SpinButton(adjustment=epochs_adjustment)
        grid.attach(self.epochs_spin, 1, 0, 1, 1)

        # Per Device Train Batch Size
        batch_size_label = Gtk.Label(label="Per Device Train Batch Size:")
        batch_size_label.set_halign(Gtk.Align.START)
        grid.attach(batch_size_label, 0, 1, 1, 1)
        batch_size_adjustment = Gtk.Adjustment(value=8, lower=1, upper=128, step_increment=1, page_increment=8)
        self.batch_size_spin = Gtk.SpinButton(adjustment=batch_size_adjustment)
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
        save_steps_adjustment = Gtk.Adjustment(value=1000, lower=100, upper=10000, step_increment=100, page_increment=1000)
        self.save_steps_spin = Gtk.SpinButton(adjustment=save_steps_adjustment)
        grid.attach(self.save_steps_spin, 1, 4, 1, 1)

        # Save Total Limit
        save_total_label = Gtk.Label(label="Save Total Limit:")
        save_total_label.set_halign(Gtk.Align.START)
        grid.attach(save_total_label, 0, 5, 1, 1)
        save_total_adjustment = Gtk.Adjustment(value=2, lower=1, upper=100, step_increment=1, page_increment=5)
        self.save_total_spin = Gtk.SpinButton(adjustment=save_total_adjustment)
        grid.attach(self.save_total_spin, 1, 5, 1, 1)

        # Number of Layers to Freeze
        layers_freeze_label = Gtk.Label(label="Number of Layers to Freeze:")
        layers_freeze_label.set_halign(Gtk.Align.START)
        grid.attach(layers_freeze_label, 0, 6, 1, 1)
        layers_freeze_adjustment = Gtk.Adjustment(value=0, lower=0, upper=100, step_increment=1, page_increment=5)
        self.layers_freeze_spin = Gtk.SpinButton(adjustment=layers_freeze_adjustment)
        grid.attach(self.layers_freeze_spin, 1, 6, 1, 1)

        return grid

    def create_model_management_tab(self):
        """
        Creates and returns the Model Management tab for selecting, loading, unloading,
        removing, and updating installed models.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # Select Active Model
        model_sel_label = Gtk.Label(label="Select Active Model:")
        model_sel_label.set_halign(Gtk.Align.START)
        grid.attach(model_sel_label, 0, 1, 1, 1)
        self.model_combo = Gtk.ComboBoxText()
        grid.attach(self.model_combo, 1, 1, 2, 1)

        # Clear Cache Button
        clear_cache_button = Gtk.Button(label="Clear Cache")
        clear_cache_button.connect("clicked", self.on_clear_cache_clicked)
        grid.attach(clear_cache_button, 0, 2, 3, 1)

        # Remove Installed Model
        remove_model_label = Gtk.Label(label="Remove Installed Model:")
        remove_model_label.set_halign(Gtk.Align.START)
        grid.attach(remove_model_label, 0, 3, 1, 1)
        self.remove_model_combo = Gtk.ComboBoxText()
        grid.attach(self.remove_model_combo, 1, 3, 1, 1)
        remove_model_button = Gtk.Button(label="Remove Model")
        remove_model_button.connect("clicked", self.on_remove_model_clicked)
        grid.attach(remove_model_button, 2, 3, 1, 1)

        # Update Installed Model
        update_model_label = Gtk.Label(label="Update Installed Model:")
        update_model_label.set_halign(Gtk.Align.START)
        grid.attach(update_model_label, 0, 4, 1, 1)
        self.update_model_combo = Gtk.ComboBoxText()
        grid.attach(self.update_model_combo, 1, 4, 1, 1)
        update_model_button = Gtk.Button(label="Update Model")
        update_model_button.connect("clicked", self.on_update_model_clicked)
        grid.attach(update_model_button, 2, 4, 1, 1)

        return grid

    def create_miscellaneous_tab(self):
        """
        Creates and returns the Miscellaneous tab for caching and logging settings.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # Enable Caching checkbutton
        self.caching_check = Gtk.CheckButton(label="Enable Caching")
        self.caching_check.set_active(True)
        grid.attach(self.caching_check, 0, 0, 2, 1)

        # Logging Level setting
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
        """
        Creates and returns the Search & Download tab for querying and installing models.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # Search Query input
        search_label = Gtk.Label(label="Search Query:")
        search_label.set_halign(Gtk.Align.START)
        grid.attach(search_label, 0, 0, 1, 1)
        self.search_entry = Gtk.Entry()
        grid.attach(self.search_entry, 1, 0, 2, 1)

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

        # Search Results area (ListBox within a ScrolledWindow)
        results_label = Gtk.Label(label="Search Results:")
        results_label.set_halign(Gtk.Align.START)
        grid.attach(results_label, 0, 5, 3, 1)
        self.results_listbox = Gtk.ListBox()
        self.results_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_child(self.results_listbox)
        grid.attach(scroll, 0, 6, 3, 1)

        return grid

    def set_nvme_sensitive(self, sensitive):
        """
        Toggles the sensitivity of NVMe-related widgets.
        
        Args:
            sensitive (bool): True to enable NVMe fields; False to disable.
        """
        self.nvme_path_entry.set_sensitive(sensitive)
        self.buffer_param_spin.set_sensitive(sensitive)
        self.buffer_opt_spin.set_sensitive(sensitive)
        self.block_size_spin.set_sensitive(sensitive)
        self.queue_depth_spin.set_sensitive(sensitive)

    def on_enable_nvme_toggled(self, widget):
        """
        Callback for toggling NVMe offloading. Updates widget sensitivity.
        """
        self.set_nvme_sensitive(widget.get_active())

    def on_nvme_path_selected(self, widget):
        """
        Callback when a folder is selected from the NVMe file chooser.
        """
        selected_folder = widget.get_file()
        if selected_folder:
            self.nvme_path_entry.set_text(selected_folder.get_path())

    def populate_model_comboboxes(self):
        """
        Populates the model, remove model, and update model combo boxes with
        the installed models obtained via the HuggingFace generator.
        """
        installed_models = self.ATLAS.provider_manager.huggingface_generator.get_installed_models()
        self.model_combo.remove_all()
        self.remove_model_combo.remove_all()
        self.update_model_combo.remove_all()
        if not installed_models:
            self.show_message("Info", "No installed models found. Please download a model.", Gtk.MessageType.INFO)
            return
        for model in installed_models:
            self.model_combo.append_text(model)
            self.remove_model_combo.append_text(model)
            self.update_model_combo.append_text(model)
        self.model_combo.set_active(0)
        self.remove_model_combo.set_active(0)
        self.update_model_combo.set_active(0)

    # Asynchronous operation wrappers using _run_async to avoid blocking the UI.
    def on_load_model_clicked(self, widget):
        """
        Callback for loading a model. Runs the model load asynchronously.
        """
        selected_model = self.model_combo.get_active_text()
        if selected_model:
            self._run_async(
                self.ATLAS.provider_manager.huggingface_generator.load_model(selected_model),
                success_callback=lambda _: self.show_message("Success", f"Model '{selected_model}' loaded successfully.", Gtk.MessageType.INFO),
                error_callback=lambda e: self.show_message("Error", f"Error loading model: {str(e)}", Gtk.MessageType.ERROR)
            )
        else:
            self.show_message("Error", "No model selected to load.", Gtk.MessageType.ERROR)

    def on_unload_model_clicked(self, widget):
        """
        Callback for unloading the current model.
        """
        try:
            self.ATLAS.provider_manager.huggingface_generator.unload_model()
            self.show_message("Success", "Model unloaded successfully.", Gtk.MessageType.INFO)
        except Exception as e:
            self.show_message("Error", f"Error unloading model: {str(e)}", Gtk.MessageType.ERROR)

    def on_fine_tune_clicked(self, widget):
        """
        Callback for initiating fine-tuning.
        In this production implementation, this functionality is fully disabled.
        """
        # In a production system, this would trigger the actual fine-tuning workflow.
        # For now, we disable the functionality and inform the user.
        self.show_message("Info", "Fine-tuning functionality is currently disabled in this release.", Gtk.MessageType.INFO)

    def on_save_clicked(self, widget):
        """
        Callback for saving the settings.
        Gathers all settings from the UI and saves them using the configuration manager.
        """
        try:
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
            self.ATLAS.base_config.update_model_settings(settings)
            self.show_message("Settings Saved", "Your settings have been saved successfully.", Gtk.MessageType.INFO)
        except ValueError as ve:
            self.show_message("Invalid Input", f"Please ensure all fields are correctly filled: {str(ve)}", Gtk.MessageType.ERROR)
        except Exception as e:
            self.show_message("Error", f"An error occurred while saving settings: {str(e)}", Gtk.MessageType.ERROR)

    def on_cancel_clicked(self, widget):
        """
        Callback for canceling the settings. Closes the window.
        """
        self.close()

    def on_back_clicked(self, widget):
        """
        Callback for navigating back. Closes the settings window.
        """
        self.close()

    def on_clear_cache_clicked(self, widget):
        """
        Callback for clearing the model cache.
        """
        cache_file = self.ATLAS.base_config.cache_file
        confirmation = self.confirm_dialog(f"Are you sure you want to clear the cache at {cache_file}?")
        if confirmation:
            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    self.show_message("Success", "Cache cleared successfully.", Gtk.MessageType.INFO)
                else:
                    self.show_message("Info", "Cache file does not exist.", Gtk.MessageType.INFO)
            except Exception as e:
                self.show_message("Error", f"Error clearing cache: {str(e)}", Gtk.MessageType.ERROR)

    def on_remove_model_clicked(self, widget):
        """
        Callback for removing an installed model.
        """
        selected_model = self.remove_model_combo.get_active_text()
        if selected_model:
            confirmation = self.confirm_dialog(f"Are you sure you want to remove the model '{selected_model}'?")
            if confirmation:
                self.ATLAS.provider_manager.huggingface_generator.model_manager.remove_installed_model(selected_model)
                self.populate_model_comboboxes()
                self.show_message("Success", f"Model '{selected_model}' removed successfully.", Gtk.MessageType.INFO)
        else:
            self.show_message("Error", "No model selected to remove.", Gtk.MessageType.ERROR)

    def on_update_model_clicked(self, widget):
        """
        Callback for updating an installed model.
        Executes the update asynchronously.
        """
        selected_model = self.update_model_combo.get_active_text()
        if selected_model:
            confirmation = self.confirm_dialog(f"Do you want to update the model '{selected_model}'?")
            if confirmation:
                self._run_async(
                    self.ATLAS.provider_manager.huggingface_generator.load_model(selected_model, force_download=True),
                    success_callback=lambda _: self.show_message("Success", f"Model '{selected_model}' updated successfully.", Gtk.MessageType.INFO),
                    error_callback=lambda e: self.show_message("Error", f"Error updating model: {str(e)}", Gtk.MessageType.ERROR)
                )
        else:
            self.show_message("Error", "No model selected to update.", Gtk.MessageType.ERROR)

    def on_search_clicked(self, widget):
        """
        Callback for the Search button in the Search & Download tab.
        Gathers filter parameters and performs the model search asynchronously.
        """
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

        GLib.idle_add(self.perform_search, search_query, filter_args)

    def perform_search(self, search_query, filter_args):
        """
        Performs a search for models using the HuggingFace HfApi, and updates the UI with the results.
        """
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            models = api.list_models(search=search_query, **filter_args)
            models = list(models)[:10]  # Limit to 10 results for display

            # Clear previous results.
            for row in self.results_listbox.get_children():
                self.results_listbox.remove(row)

            if not models:
                label = Gtk.Label(label="No models found matching the criteria.")
                self.results_listbox.append(label)
            else:
                for model in models:
                    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
                    box.set_margin_top(5)
                    box.set_margin_bottom(5)
                    info_label = Gtk.Label()
                    info_label.set_xalign(0)
                    info_text = f"Model ID: {model.modelId}\nTags: {', '.join(model.tags)}\nDownloads: {model.downloads}\nLikes: {model.likes}"
                    info_label.set_text(info_text)
                    box.append(info_label)
                    download_button = Gtk.Button(label="Download")
                    download_button.connect("clicked", self.on_download_model_clicked, model.modelId)
                    box.append(download_button)
                    self.results_listbox.append(box)
            self.results_listbox.show()
        except Exception as e:
            self.show_message("Error", f"An error occurred while searching for models: {str(e)}", Gtk.MessageType.ERROR)
        return False  # Stop the idle_add

    def on_download_model_clicked(self, widget, model_name):
        """
        Callback for the Download button.
        Asks for confirmation and then runs the download asynchronously.
        """
        confirmation = self.confirm_dialog(f"Do you want to download and install the model '{model_name}'?")
        if confirmation:
            self._run_async(
                self.ATLAS.provider_manager.huggingface_generator.load_model(model_name, force_download=True),
                success_callback=lambda _: (self.populate_model_comboboxes(), self.show_message("Success", f"Model '{model_name}' downloaded and installed successfully.", Gtk.MessageType.INFO)),
                error_callback=lambda e: self.show_message("Error", f"Error downloading model '{model_name}': {str(e)}", Gtk.MessageType.ERROR)
            )

    def confirm_dialog(self, message):
        """
        Displays a confirmation dialog with Yes/No buttons.

        Args:
            message (str): The confirmation message.

        Returns:
            bool: True if the user selects Yes, False otherwise.
        """
        dialog = Gtk.MessageDialog(
            transient_for=self.parent_window, 
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.YES_NO,
            text=message,
        )
        response = dialog.run()
        dialog.destroy()
        return response == Gtk.ResponseType.YES

    def show_message(self, title, message, message_type=Gtk.MessageType.INFO):
        """
        Displays an informational or error message dialog.

        Args:
            title (str): The title of the message.
            message (str): The detailed message text.
            message_type (Gtk.MessageType): The type of the message (INFO, ERROR, etc.).
        """
        dialog = Gtk.MessageDialog(
            transient_for=self.parent_window,  
            modal=True,
            message_type=message_type,
            buttons=Gtk.ButtonsType.OK
        )
        dialog.text = title
        dialog.secondary_text = message
        dialog.connect("response", lambda dialog, response: dialog.destroy())
        dialog.present()
