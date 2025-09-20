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
from gi.repository import Gtk, Gdk, GLib

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

        # Icon helpers (for eye toggle)
        self._eye_icon_path = self._abs_icon("Icons/eye.png")
        self._eye_off_icon_path = self._abs_icon("Icons/eye-off.png")

        # Create a vertical box (with uniform margins) to hold all widgets.
        vbox = create_box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin=10)
        self.set_child(vbox)
        
        # Create a Notebook widget to hold the different settings tabs.
        self.notebook = Gtk.Notebook()
        self.notebook.set_tooltip_text("Configure Hugging Face provider behavior, models, and credentials.")
        vbox.append(self.notebook)
        
        # General Settings Tab
        general_settings = self.create_general_settings_tab()
        general_label = Gtk.Label(label="General")
        general_label.set_tooltip_text("Core generation parameters and credentials.")
        self.notebook.append_page(general_settings, general_label)
        
        # Advanced Tab – with sub-notebook for Optimizations, NVMe Offloading and Miscellaneous.
        advanced_optimizations_notebook = Gtk.Notebook()
        advanced_optimizations_notebook.set_tooltip_text("Low-level performance options for advanced use.")
        advanced_optimizations = self.create_advanced_optimizations_tab()
        nvme_offloading = self.create_nvme_offloading_tab()
        miscellaneous = self.create_miscellaneous_tab()
        tab1 = Gtk.Label(label="Optimizations"); tab1.set_tooltip_text("Quantization and performance switches.")
        tab2 = Gtk.Label(label="NVMe Offloading"); tab2.set_tooltip_text("Offload large tensors to NVMe storage.")
        tab3 = Gtk.Label(label="Misc"); tab3.set_tooltip_text("Caching and logging preferences.")
        advanced_optimizations_notebook.append_page(advanced_optimizations, tab1)
        advanced_optimizations_notebook.append_page(nvme_offloading, tab2)
        advanced_optimizations_notebook.append_page(miscellaneous, tab3)
        adv_label = Gtk.Label(label="Advanced")
        adv_label.set_tooltip_text("Advanced performance tuning.")
        self.notebook.append_page(advanced_optimizations_notebook, adv_label)
        
        # Fine-Tuning Settings Tab – attach the Fine-Tune Model button directly into the grid.
        fine_tuning_settings = self.create_fine_tuning_settings_tab()
        fine_tuning_grid = fine_tuning_settings  # The grid is already created.
        fine_tune_button = Gtk.Button(label="Fine-Tune Model")
        fine_tune_button.set_tooltip_text("Start a fine-tuning workflow (currently disabled).")
        fine_tune_button.connect("clicked", self.on_fine_tune_clicked)
        fine_tuning_grid.attach(fine_tune_button, 0, 7, 2, 1)
        ft_label = Gtk.Label(label="Fine-Tuning")
        ft_label.set_tooltip_text("Training parameters (feature disabled in this build).")
        self.notebook.append_page(fine_tuning_settings, ft_label)
        
        # Model Management Tab – contains a sub-notebook for Manage Models and Search & Download.
        models_notebook = Gtk.Notebook()
        models_notebook.set_tooltip_text("Manage installed models, search and download new ones.")
        model_management = self.create_model_management_tab()
        # Adjust button positions for Load and Unload Model Buttons.
        load_model_button = Gtk.Button(label="Load Model")
        load_model_button.set_tooltip_text("Load the selected installed model into memory.")
        load_model_button.connect("clicked", self.on_load_model_clicked)
        unload_model_button = Gtk.Button(label="Unload Model")
        unload_model_button.set_tooltip_text("Unload the currently active model from memory.")
        unload_model_button.connect("clicked", self.on_unload_model_clicked)
        model_management_grid = model_management
        model_management_grid.attach(load_model_button, 0, 0, 1, 1)
        model_management_grid.attach(unload_model_button, 1, 0, 1, 1)
        search_download = self.create_search_download_tab()
        tab_manage = Gtk.Label(label="Manage Models"); tab_manage.set_tooltip_text("Select, remove, update models.")
        tab_search = Gtk.Label(label="Search & Download"); tab_search.set_tooltip_text("Find and install Hugging Face models.")
        models_notebook.append_page(model_management, tab_manage)
        models_notebook.append_page(search_download, tab_search)
        models_label = Gtk.Label(label="Models")
        models_label.set_tooltip_text("Model administration and discovery.")
        self.notebook.append_page(models_notebook, models_label)
        
        # Control Buttons at the bottom.
        control_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        vbox.append(control_box)
        back_button = Gtk.Button(label="Back")
        back_button.set_tooltip_text("Close this window and return to the previous view.")
        back_button.connect("clicked", self.on_back_clicked)
        control_box.append(back_button)
        cancel_button = Gtk.Button(label="Cancel")
        cancel_button.set_tooltip_text("Discard changes and close this window.")
        cancel_button.connect("clicked", self.on_cancel_clicked)
        control_box.append(cancel_button)
        save_button = Gtk.Button(label="Save Settings")
        save_button.set_tooltip_text("Persist the current settings to configuration.")
        save_button.connect("clicked", self.on_save_clicked)
        control_box.append(save_button)
        
        # Populate model comboboxes with installed models.
        self.populate_model_comboboxes()

    # ---------------------------------------------------------------------
    # Helpers: async runner, icon loaders, CSS
    # ---------------------------------------------------------------------

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

    def _dispatch_provider_result(self, result, *, success_message: str, failure_message: str, on_success=None):
        """Display feedback for provider manager operations."""
        if isinstance(result, dict) and result.get("success"):
            if on_success:
                on_success()
            message = result.get("message") or success_message
            self.show_message("Success", message, Gtk.MessageType.INFO)
            return

        error_detail = ""
        if isinstance(result, dict):
            error_detail = result.get("error", "")
        elif result is not None:
            error_detail = str(result)

        message = failure_message
        if error_detail:
            message = f"{failure_message}: {error_detail}"
        self.show_message("Error", message, Gtk.MessageType.ERROR)

    def _get_saved_hf_token(self) -> str:
        """Retrieve a stored Hugging Face token from configuration if available."""

        if hasattr(self.config_manager, "get_huggingface_api_key"):
            return self.config_manager.get_huggingface_api_key() or ""
        if hasattr(self.config_manager, "get_config"):
            return self.config_manager.get_config("HUGGINGFACE_API_KEY") or ""
        return ""

    def _handle_token_test_result(self, result):
        """Present the outcome of a token validation attempt to the user."""

        if not isinstance(result, dict):
            self.show_message(
                "Warning",
                "Token test returned an unexpected response.",
                Gtk.MessageType.WARNING,
            )
            return

        if result.get("success"):
            info = result.get("data") or {}
            display_name = (
                info.get("name")
                or info.get("fullname")
                or info.get("email")
                or "Authenticated"
            )
            message = result.get("message") or f"Token OK. Signed in as: {display_name}"
            self.show_message("Success", message, Gtk.MessageType.INFO)
            return

        error_message = result.get("error") or "Token test failed."
        self.show_message("Error", f"Token test failed: {error_message}", Gtk.MessageType.ERROR)

    def _abs_icon(self, relative_path: str) -> str:
        """Resolve absolute path for an icon relative to the project root."""
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        return os.path.join(base, relative_path)

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
        display = Gtk.Window().get_display()
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_USER
        )

    # ---------------------------------------------------------------------
    # UI Builders
    # ---------------------------------------------------------------------

    def _build_secret_row(self, label_text: str, placeholder: str = "Enter value here", default_visible: bool = False):
        """
        Build a labeled secret row with: Label, Entry (password-mode), and an eye ToggleButton.
        Returns (container_box, entry_widget, toggle_button).
        """
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        label = Gtk.Label(label=label_text)
        label.set_xalign(0.0)
        label.set_tooltip_text("Enter your value. Use the eye to show or hide.")
        vbox.append(label)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        entry = Gtk.Entry()
        entry.set_placeholder_text(placeholder)
        entry.set_invisible_char('*')
        entry.set_visibility(default_visible)  # hidden by default unless requested
        entry.set_tooltip_text("Value is hidden when the eye is off.")
        entry.set_hexpand(True)
        hbox.append(entry)

        toggle = Gtk.ToggleButton()
        toggle.set_can_focus(True)
        # AccessibleRole.TOGGLE_BUTTON not present in GTK4; use BUTTON or omit
        role = getattr(Gtk.AccessibleRole, "BUTTON", None)
        if role is not None:
            toggle.set_accessible_role(role)
        toggle.set_tooltip_text("Hide value" if default_visible else "Show value")
        icon_path = self._eye_off_icon_path if default_visible else self._eye_icon_path
        fallback_icon = "view-conceal-symbolic" if default_visible else "view-reveal-symbolic"
        toggle.set_child(self._load_icon_picture(icon_path, fallback_icon, 18))
        toggle.set_active(default_visible)

        def on_toggled(btn: Gtk.ToggleButton):
            visible = btn.get_active()
            entry.set_visibility(visible)
            icon_name = "view-conceal-symbolic" if visible else "view-reveal-symbolic"
            icon_path_local = self._eye_off_icon_path if visible else self._eye_icon_path
            btn.set_child(self._load_icon_picture(icon_path_local, icon_name, 18))
            btn.set_tooltip_text("Hide value" if visible else "Show value")

        toggle.connect("toggled", on_toggled)
        hbox.append(toggle)

        vbox.append(hbox)
        return vbox, entry, toggle

    def create_general_settings_tab(self):
        """
        Creates and returns the General Settings tab containing model generation parameters
        and Hugging Face API token with eye toggle + actions.
        """
        grid = Gtk.Grid(column_spacing=10, row_spacing=10)
        grid.set_margin_top(10)
        grid.set_margin_bottom(10)
        grid.set_margin_start(10)
        grid.set_margin_end(10)

        # ---------------- Credentials (Hugging Face Token) ----------------
        creds_label = Gtk.Label(label="Hugging Face API Token:")
        creds_label.set_halign(Gtk.Align.START)
        creds_label.set_tooltip_text("Personal access token used for private model access and higher rate limits.")
        grid.attach(creds_label, 0, 0, 1, 1)

        creds_row, self.hf_token_entry, self.hf_token_toggle = self._build_secret_row(
            "Token", placeholder="hf_...", default_visible=False
        )
        # Place the whole row in a single cell spanning two columns
        grid.attach(creds_row, 1, 0, 2, 1)

        # Prefill from config (do not reveal). Show 'Saved' placeholder if present.
        try:
            existing_token = ""
            if hasattr(self.config_manager, "get_huggingface_api_key"):
                existing_token = self.config_manager.get_huggingface_api_key() or ""
            elif hasattr(self.config_manager, "get_config"):
                existing_token = self.config_manager.get_config("HUGGINGFACE_API_KEY") or ""
            if existing_token:
                self.hf_token_entry.set_text("")
                self.hf_token_entry.set_placeholder_text("Saved")
        except Exception:
            pass

        # Buttons: Save Token, Test Connection
        token_buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        save_token_btn = Gtk.Button(label="Save Token")
        save_token_btn.set_tooltip_text("Save the Hugging Face token to your configuration.")
        save_token_btn.connect("clicked", self.on_save_token_clicked)
        test_token_btn = Gtk.Button(label="Test Connection")
        test_token_btn.set_tooltip_text("Verify the token by calling the Hugging Face API.")
        test_token_btn.connect("clicked", self.on_test_token_clicked)
        token_buttons.append(save_token_btn)
        token_buttons.append(test_token_btn)
        grid.attach(token_buttons, 1, 1, 2, 1)

        # ---------------- Generation Parameters ----------------
        # Temperature
        temp_label = Gtk.Label(label="Temperature:")
        temp_label.set_halign(Gtk.Align.START)
        temp_label.set_tooltip_text("Higher = more random generations. Typical range 0.2–1.0.")
        grid.attach(temp_label, 0, 2, 1, 1)
        temp_adjustment = Gtk.Adjustment(value=0.7, lower=0.0, upper=2.0, step_increment=0.1, page_increment=0.5)
        self.temp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=temp_adjustment)
        self.temp_scale.set_digits(2)
        self.temp_scale.set_tooltip_text("Sampling temperature.")
        grid.attach(self.temp_scale, 1, 2, 1, 1)

        # Top-p
        topp_label = Gtk.Label(label="Top-p:")
        topp_label.set_halign(Gtk.Align.START)
        topp_label.set_tooltip_text("Nucleus sampling; consider tokens in the top cumulative probability p.")
        grid.attach(topp_label, 0, 3, 1, 1)
        topp_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=1.0, step_increment=0.05, page_increment=0.1)
        self.topp_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=topp_adjustment)
        self.topp_scale.set_digits(2)
        self.topp_scale.set_tooltip_text("Lower to make output more focused.")
        grid.attach(self.topp_scale, 1, 3, 1, 1)

        # Top-k
        topk_label = Gtk.Label(label="Top-k:")
        topk_label.set_halign(Gtk.Align.START)
        topk_label.set_tooltip_text("Sample only from the top-k most likely tokens (1 disables).")
        grid.attach(topk_label, 0, 4, 1, 1)
        topk_adjustment = Gtk.Adjustment(value=50, lower=1, upper=1000, step_increment=1, page_increment=10)
        self.topk_spin = Gtk.SpinButton(adjustment=topk_adjustment)
        self.topk_spin.set_tooltip_text("Common values: 40–200.")
        grid.attach(self.topk_spin, 1, 4, 1, 1)

        # Max Tokens
        maxt_label = Gtk.Label(label="Max Tokens:")
        maxt_label.set_halign(Gtk.Align.START)
        maxt_label.set_tooltip_text("Maximum number of new tokens to generate.")
        grid.attach(maxt_label, 0, 5, 1, 1)
        maxt_adjustment = Gtk.Adjustment(value=100, lower=1, upper=2048, step_increment=1, page_increment=10)
        self.maxt_spin = Gtk.SpinButton(adjustment=maxt_adjustment)
        self.maxt_spin.set_tooltip_text("Set according to context window and performance needs.")
        grid.attach(self.maxt_spin, 1, 5, 1, 1)

        # Repetition Penalty
        rp_label = Gtk.Label(label="Repetition Penalty:")
        rp_label.set_halign(Gtk.Align.START)
        rp_label.set_tooltip_text("Penalize repeated tokens (>1.0 reduces repetition).")
        grid.attach(rp_label, 0, 6, 1, 1)
        rp_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=10.0, step_increment=0.1, page_increment=1.0)
        self.rp_spin = Gtk.SpinButton(adjustment=rp_adjustment, digits=2)
        self.rp_spin.set_tooltip_text("Typical range 0.9–1.3.")
        grid.attach(self.rp_spin, 1, 6, 1, 1)

        # Presence Penalty
        pres_penalty_label = Gtk.Label(label="Presence Penalty:")
        pres_penalty_label.set_halign(Gtk.Align.START)
        pres_penalty_label.set_tooltip_text("Encourages discussing new topics when positive.")
        grid.attach(pres_penalty_label, 0, 7, 1, 1)
        pres_penalty_adjustment = Gtk.Adjustment(value=0.0, lower=-2.0, upper=2.0, step_increment=0.1, page_increment=0.5)
        self.pres_penalty_spin = Gtk.SpinButton(adjustment=pres_penalty_adjustment, digits=2)
        self.pres_penalty_spin.set_tooltip_text("Negative values encourage staying on topic.")
        grid.attach(self.pres_penalty_spin, 1, 7, 1, 1)

        # Length Penalty
        length_penalty_label = Gtk.Label(label="Length Penalty:")
        length_penalty_label.set_halign(Gtk.Align.START)
        length_penalty_label.set_tooltip_text("Penalize long outputs (>1.0 favors shorter results).")
        grid.attach(length_penalty_label, 0, 8, 1, 1)
        length_penalty_adjustment = Gtk.Adjustment(value=1.0, lower=0.0, upper=10.0, step_increment=0.1, page_increment=1.0)
        self.length_penalty_spin = Gtk.SpinButton(adjustment=length_penalty_adjustment, digits=2)
        self.length_penalty_spin.set_tooltip_text("Set to 1.0 to disable.")
        grid.attach(self.length_penalty_spin, 1, 8, 1, 1)

        # Early Stopping
        self.early_stopping_check = Gtk.CheckButton(label="Early Stopping")
        self.early_stopping_check.set_tooltip_text("Stop generation when a natural stopping point is reached.")
        grid.attach(self.early_stopping_check, 0, 9, 2, 1)

        # Do Sample
        self.do_sample_check = Gtk.CheckButton(label="Do Sample")
        self.do_sample_check.set_tooltip_text("Enable sampling (disable to use greedy decoding).")
        grid.attach(self.do_sample_check, 0, 10, 2, 1)

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
        quant_label.set_tooltip_text("Reduce model precision for memory/perf trade-offs.")
        grid.attach(quant_label, 0, 0, 1, 1)
        self.quant_combo = Gtk.ComboBoxText()
        self.quant_combo.set_tooltip_text("Pick a quantization level (if supported by the backend).")
        self.quant_combo.append_text("None")
        self.quant_combo.append_text("4bit")
        self.quant_combo.append_text("8bit")
        self.quant_combo.set_active(0)
        grid.attach(self.quant_combo, 1, 0, 1, 1)

        # Gradient Checkpointing
        self.gc_check = Gtk.CheckButton(label="Gradient Checkpointing")
        self.gc_check.set_tooltip_text("Reduce memory usage during training (slower compute).")
        grid.attach(self.gc_check, 0, 1, 2, 1)

        # LoRA
        self.lora_check = Gtk.CheckButton(label="Low-Rank Adaptation (LoRA)")
        self.lora_check.set_tooltip_text("Parameter-efficient fine-tuning strategy.")
        grid.attach(self.lora_check, 0, 2, 2, 1)

        # FlashAttention
        self.fa_check = Gtk.CheckButton(label="FlashAttention Optimization")
        self.fa_check.set_tooltip_text("Use optimized attention kernels where available.")
        grid.attach(self.fa_check, 0, 3, 2, 1)

        # Pruning
        self.pruning_check = Gtk.CheckButton(label="Model Pruning")
        self.pruning_check.set_tooltip_text("Remove parameters to speed up inference (experimental).")
        grid.attach(self.pruning_check, 0, 4, 2, 1)

        # Memory Mapping
        self.mem_map_check = Gtk.CheckButton(label="Memory Mapping")
        self.mem_map_check.set_tooltip_text("Memory-map weights to reduce RAM usage.")
        grid.attach(self.mem_map_check, 0, 5, 2, 1)

        # bfloat16
        self.bfloat16_check = Gtk.CheckButton(label="Use bfloat16")
        self.bfloat16_check.set_tooltip_text("Enable bfloat16 if supported by hardware.")
        grid.attach(self.bfloat16_check, 0, 6, 2, 1)

        # Torch Compile
        self.torch_compile_check = Gtk.CheckButton(label="Torch Compile")
        self.torch_compile_check.set_tooltip_text("Optimize PyTorch graphs (may increase compile time).")
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
        self.enable_nvme_check.set_tooltip_text("Offload large tensors to NVMe storage to fit bigger models.")
        grid.attach(self.enable_nvme_check, 0, 0, 2, 1)
        self.enable_nvme_check.connect("toggled", self.on_enable_nvme_toggled)

        # NVMe Path entry and selection button
        nvme_path_label = Gtk.Label(label="NVMe Path:")
        nvme_path_label.set_halign(Gtk.Align.START)
        nvme_path_label.set_tooltip_text("Directory used for NVMe offloading (ensure fast SSD/NVMe).")
        grid.attach(nvme_path_label, 0, 1, 1, 1)

        self.nvme_path_entry = Gtk.Entry()
        self.nvme_path_entry.set_tooltip_text("Absolute path to a writable, high-speed NVMe directory.")
        grid.attach(self.nvme_path_entry, 1, 1, 1, 1)

        nvme_path_button = Gtk.Button(label="Select NVMe Path")
        nvme_path_button.set_tooltip_text("Browse for the NVMe folder to use for offloading.")
        nvme_path_button.connect("clicked", self.on_nvme_path_button_clicked)
        grid.attach(nvme_path_button, 2, 1, 1, 1)

        # NVMe Buffer Count (Parameters)
        buffer_param_label = Gtk.Label(label="NVMe Buffer Count (Parameters):")
        buffer_param_label.set_halign(Gtk.Align.START)
        buffer_param_label.set_tooltip_text("Number of prefetch buffers for parameter tensors.")
        grid.attach(buffer_param_label, 0, 2, 1, 1)
        buffer_param_adjustment = Gtk.Adjustment(value=5, lower=1, upper=10, step_increment=1, page_increment=2)
        self.buffer_param_spin = Gtk.SpinButton(adjustment=buffer_param_adjustment)
        self.buffer_param_spin.set_tooltip_text("Tune for throughput vs. memory usage.")
        grid.attach(self.buffer_param_spin, 1, 2, 1, 1)

        # NVMe Buffer Count (Optimizer)
        buffer_opt_label = Gtk.Label(label="NVMe Buffer Count (Optimizer):")
        buffer_opt_label.set_halign(Gtk.Align.START)
        buffer_opt_label.set_tooltip_text("Number of prefetch buffers for optimizer state.")
        grid.attach(buffer_opt_label, 0, 3, 1, 1)
        buffer_opt_adjustment = Gtk.Adjustment(value=4, lower=1, upper=10, step_increment=1, page_increment=2)
        self.buffer_opt_spin = Gtk.SpinButton(adjustment=buffer_opt_adjustment)
        self.buffer_opt_spin.set_tooltip_text("Increase for smoother optimizer IO on NVMe.")
        grid.attach(self.buffer_opt_spin, 1, 3, 1, 1)

        # NVMe Block Size (Bytes)
        block_size_label = Gtk.Label(label="NVMe Block Size (Bytes):")
        block_size_label.set_halign(Gtk.Align.START)
        block_size_label.set_tooltip_text("IO block size for NVMe operations (bytes).")
        grid.attach(block_size_label, 0, 4, 1, 1)
        block_size_adjustment = Gtk.Adjustment(value=1048576, lower=1024, upper=10485760, step_increment=1024, page_increment=10240)
        self.block_size_spin = Gtk.SpinButton(adjustment=block_size_adjustment)
        self.block_size_spin.set_tooltip_text("Larger blocks may improve throughput on fast SSDs.")
        grid.attach(self.block_size_spin, 1, 4, 1, 1)

        # NVMe Queue Depth
        queue_depth_label = Gtk.Label(label="NVMe Queue Depth:")
        queue_depth_label.set_halign(Gtk.Align.START)
        queue_depth_label.set_tooltip_text("Outstanding IO requests allowed; higher can boost throughput.")
        grid.attach(queue_depth_label, 0, 5, 1, 1)
        queue_depth_adjustment = Gtk.Adjustment(value=8, lower=1, upper=64, step_increment=1, page_increment=4)
        self.queue_depth_spin = Gtk.SpinButton(adjustment=queue_depth_adjustment)
        self.queue_depth_spin.set_tooltip_text("Beware diminishing returns beyond SSD capability.")
        grid.attach(self.queue_depth_spin, 1, 5, 1, 1)

        # Set the sensitivity of NVMe settings based on the switch state.
        self.set_nvme_sensitive(self.enable_nvme_check.get_active())

        return grid

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
        epochs_label.set_tooltip_text("How many full passes over the training data.")
        grid.attach(epochs_label, 0, 0, 1, 1)
        epochs_adjustment = Gtk.Adjustment(value=3, lower=1, upper=100, step_increment=1, page_increment=10)
        self.epochs_spin = Gtk.SpinButton(adjustment=epochs_adjustment)
        self.epochs_spin.set_tooltip_text("Common range: 1–5 for small finetunes.")
        grid.attach(self.epochs_spin, 1, 0, 1, 1)

        # Per Device Train Batch Size
        batch_size_label = Gtk.Label(label="Per Device Train Batch Size:")
        batch_size_label.set_halign(Gtk.Align.START)
        batch_size_label.set_tooltip_text("Samples per device per step; constrained by VRAM.")
        grid.attach(batch_size_label, 0, 1, 1, 1)
        batch_size_adjustment = Gtk.Adjustment(value=8, lower=1, upper=128, step_increment=1, page_increment=8)
        self.batch_size_spin = Gtk.SpinButton(adjustment=batch_size_adjustment)
        self.batch_size_spin.set_tooltip_text("Increase for faster training if memory allows.")
        grid.attach(self.batch_size_spin, 1, 1, 1, 1)

        # Learning Rate
        lr_label = Gtk.Label(label="Learning Rate:")
        lr_label.set_halign(Gtk.Align.START)
        lr_label.set_tooltip_text("Step size for optimizer updates (e.g., 5e-5).")
        grid.attach(lr_label, 0, 2, 1, 1)
        self.lr_entry = Gtk.Entry(text="5e-5")
        self.lr_entry.set_tooltip_text("Use scientific notation or decimal (e.g., 0.00005).")
        grid.attach(self.lr_entry, 1, 2, 1, 1)

        # Weight Decay
        wd_label = Gtk.Label(label="Weight Decay:")
        wd_label.set_halign(Gtk.Align.START)
        wd_label.set_tooltip_text("L2 regularization strength (e.g., 0.01).")
        grid.attach(wd_label, 0, 3, 1, 1)
        self.wd_entry = Gtk.Entry(text="0.01")
        self.wd_entry.set_tooltip_text("Set to 0 to disable weight decay.")
        grid.attach(self.wd_entry, 1, 3, 1, 1)

        # Save Steps
        save_steps_label = Gtk.Label(label="Save Steps:")
        save_steps_label.set_halign(Gtk.Align.START)
        save_steps_label.set_tooltip_text("Checkpoint every N steps.")
        grid.attach(save_steps_label, 0, 4, 1, 1)
        save_steps_adjustment = Gtk.Adjustment(value=1000, lower=100, upper=10000, step_increment=100, page_increment=1000)
        self.save_steps_spin = Gtk.SpinButton(adjustment=save_steps_adjustment)
        self.save_steps_spin.set_tooltip_text("Checkpointing too often can slow training.")
        grid.attach(self.save_steps_spin, 1, 4, 1, 1)

        # Save Total Limit
        save_total_label = Gtk.Label(label="Save Total Limit:")
        save_total_label.set_halign(Gtk.Align.START)
        save_total_label.set_tooltip_text("Max number of checkpoints to keep.")
        grid.attach(save_total_label, 0, 5, 1, 1)
        save_total_adjustment = Gtk.Adjustment(value=2, lower=1, upper=100, step_increment=1, page_increment=5)
        self.save_total_spin = Gtk.SpinButton(adjustment=save_total_adjustment)
        self.save_total_spin.set_tooltip_text("Older checkpoints will be removed beyond this number.")
        grid.attach(self.save_total_spin, 1, 5, 1, 1)

        # Number of Layers to Freeze
        layers_freeze_label = Gtk.Label(label="Number of Layers to Freeze:")
        layers_freeze_label.set_halign(Gtk.Align.START)
        layers_freeze_label.set_tooltip_text("Keep lower layers fixed to speed up/regularize training.")
        grid.attach(layers_freeze_label, 0, 6, 1, 1)
        layers_freeze_adjustment = Gtk.Adjustment(value=0, lower=0, upper=100, step_increment=1, page_increment=5)
        self.layers_freeze_spin = Gtk.SpinButton(adjustment=layers_freeze_adjustment)
        self.layers_freeze_spin.set_tooltip_text("0 = train all layers.")
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
        model_sel_label.set_tooltip_text("Pick which installed model should be active.")
        grid.attach(model_sel_label, 0, 1, 1, 1)
        self.model_combo = Gtk.ComboBoxText()
        self.model_combo.set_tooltip_text("Installed models will appear here.")
        grid.attach(self.model_combo, 1, 1, 2, 1)

        # Clear Cache Button
        clear_cache_button = Gtk.Button(label="Clear Cache")
        clear_cache_button.set_tooltip_text("Remove cached metadata for Hugging Face operations.")
        clear_cache_button.connect("clicked", self.on_clear_cache_clicked)
        grid.attach(clear_cache_button, 0, 2, 3, 1)

        # Remove Installed Model
        remove_model_label = Gtk.Label(label="Remove Installed Model:")
        remove_model_label.set_halign(Gtk.Align.START)
        remove_model_label.set_tooltip_text("Uninstall a model from local storage.")
        grid.attach(remove_model_label, 0, 3, 1, 1)
        self.remove_model_combo = Gtk.ComboBoxText()
        self.remove_model_combo.set_tooltip_text("Choose an installed model to remove.")
        grid.attach(self.remove_model_combo, 1, 3, 1, 1)
        remove_model_button = Gtk.Button(label="Remove Model")
        remove_model_button.set_tooltip_text("Permanently remove the selected model from local storage.")
        remove_model_button.connect("clicked", self.on_remove_model_clicked)
        grid.attach(remove_model_button, 2, 3, 1, 1)

        # Update Installed Model
        update_model_label = Gtk.Label(label="Update Installed Model:")
        update_model_label.set_halign(Gtk.Align.START)
        update_model_label.set_tooltip_text("Download the newest version of an installed model.")
        grid.attach(update_model_label, 0, 4, 1, 1)
        self.update_model_combo = Gtk.ComboBoxText()
        self.update_model_combo.set_tooltip_text("Choose an installed model to update.")
        grid.attach(self.update_model_combo, 1, 4, 1, 1)
        update_model_button = Gtk.Button(label="Update Model")
        update_model_button.set_tooltip_text("Fetch and install the latest model files.")
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
        self.caching_check.set_tooltip_text("Cache results and metadata to speed up repeated operations.")
        grid.attach(self.caching_check, 0, 0, 2, 1)

        # Logging Level setting
        log_level_label = Gtk.Label(label="Logging Level:")
        log_level_label.set_halign(Gtk.Align.START)
        log_level_label.set_tooltip_text("Verbosity of provider logs.")
        grid.attach(log_level_label, 0, 1, 1, 1)
        self.log_level_combo = Gtk.ComboBoxText()
        self.log_level_combo.set_tooltip_text("Select a minimum logging level.")
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
        search_label.set_tooltip_text("Enter keywords or model IDs to find.")
        grid.attach(search_label, 0, 0, 1, 1)
        self.search_entry = Gtk.Entry()
        self.search_entry.set_tooltip_text("Example: 'llama', 'bert-base-uncased'")
        grid.attach(self.search_entry, 1, 0, 2, 1)

        # Task Filter
        task_label = Gtk.Label(label="Task:")
        task_label.set_halign(Gtk.Align.START)
        task_label.set_tooltip_text("Filter by pipeline task (optional).")
        grid.attach(task_label, 0, 1, 1, 1)
        self.task_combo = Gtk.ComboBoxText()
        self.task_combo.set_tooltip_text("Pick a task or leave 'Any'.")
        self.task_combo.append_text("Any")
        tasks = ["text-classification", "question-answering", "summarization", "translation", "text-generation", "fill-mask"]
        for task in tasks:
            self.task_combo.append_text(task)
        self.task_combo.set_active(0)
        grid.attach(self.task_combo, 1, 1, 1, 1)

        # Language Filter
        language_label = Gtk.Label(label="Language:")
        language_label.set_halign(Gtk.Align.START)
        language_label.set_tooltip_text("Filter by language (optional; e.g., 'en').")
        grid.attach(language_label, 0, 2, 1, 1)
        self.language_entry = Gtk.Entry()
        self.language_entry.set_tooltip_text("ISO language code or name.")
        grid.attach(self.language_entry, 1, 2, 1, 1)

        # License Filter
        license_label = Gtk.Label(label="License:")
        license_label.set_halign(Gtk.Align.START)
        license_label.set_tooltip_text("Filter by license (optional).")
        grid.attach(license_label, 0, 3, 1, 1)
        self.license_combo = Gtk.ComboBoxText()
        self.license_combo.set_tooltip_text("Select a license or leave 'Any'.")
        self.license_combo.append_text("Any")
        licenses = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc-by-sa-4.0", "cc-by-4.0", "wtfpl"]
        for lic in licenses:
            self.license_combo.append_text(lic)
        self.license_combo.set_active(0)
        grid.attach(self.license_combo, 1, 3, 1, 1)

        # Library Filter
        library_label = Gtk.Label(label="Library:")
        library_label.set_halign(Gtk.Align.START)
        library_label.set_tooltip_text("Filter by framework/library (optional).")
        grid.attach(library_label, 0, 4, 1, 1)
        self.library_entry = Gtk.Entry()
        self.library_entry.set_tooltip_text("e.g., 'transformers', 'diffusers'")
        grid.attach(self.library_entry, 1, 4, 1, 1)

        # Search Button
        search_button = Gtk.Button(label="Search")
        search_button.set_tooltip_text("Run the search with current filters.")
        search_button.connect("clicked", self.on_search_clicked)
        grid.attach(search_button, 2, 0, 1, 1)

        # Search Results area (ListBox within a ScrolledWindow)
        results_label = Gtk.Label(label="Search Results:")
        results_label.set_halign(Gtk.Align.START)
        results_label.set_tooltip_text("Top results are shown below; download installs locally.")
        grid.attach(results_label, 0, 5, 3, 1)
        self.results_listbox = Gtk.ListBox()
        self.results_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroll.set_child(self.results_listbox)
        grid.attach(scroll, 0, 6, 3, 1)

        return grid

    # ---------------------------------------------------------------------
    # NVMe helpers
    # ---------------------------------------------------------------------

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

    def on_nvme_path_button_clicked(self, widget):
        """
        Opens a folder chooser dialog for selecting an NVMe directory (FileChooserNative).
        """
        dialog = Gtk.FileChooserNative(
            title="Select NVMe Path",
            action=Gtk.FileChooserAction.SELECT_FOLDER,
            transient_for=self
        )
        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:
            file = dialog.get_file()
            if file:
                path = file.get_path()
                if path:
                    self.nvme_path_entry.set_text(path)
        dialog.destroy()

    # ---------------------------------------------------------------------
    # Populate / Model actions
    # ---------------------------------------------------------------------

    def populate_model_comboboxes(self):
        """
        Populates the model, remove model, and update model combo boxes with
        the installed models obtained via the HuggingFace generator.
        """
        result = self.ATLAS.provider_manager.list_hf_models()
        self.model_combo.remove_all()
        self.remove_model_combo.remove_all()
        self.update_model_combo.remove_all()

        if not result.get("success"):
            message = result.get("error", "Failed to retrieve installed models.")
            self.show_message("Error", message, Gtk.MessageType.ERROR)
            return

        installed_models = result.get("data") or []
        if not installed_models:
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
                self.ATLAS.provider_manager.load_hf_model(selected_model),
                success_callback=lambda res, model=selected_model: self._dispatch_provider_result(
                    res,
                    success_message=f"Model '{model}' loaded successfully.",
                    failure_message=f"Error loading model '{model}'",
                ),
                error_callback=lambda e, model=selected_model: self.show_message(
                    "Error",
                    f"Error loading model '{model}': {str(e)}",
                    Gtk.MessageType.ERROR,
                ),
            )
        else:
            self.show_message("Error", "No model selected to load.", Gtk.MessageType.ERROR)

    def on_unload_model_clicked(self, widget):
        """
        Callback for unloading the current model.
        """
        self._run_async(
            self.ATLAS.provider_manager.unload_hf_model(),
            success_callback=lambda res: self._dispatch_provider_result(
                res,
                success_message="Model unloaded successfully.",
                failure_message="Error unloading model",
            ),
            error_callback=lambda e: self.show_message(
                "Error",
                f"Error unloading model: {str(e)}",
                Gtk.MessageType.ERROR,
            ),
        )

    def on_fine_tune_clicked(self, widget):
        """
        Callback for initiating fine-tuning.
        In this production implementation, this functionality is fully disabled.
        """
        self.show_message("Info", "Fine-tuning functionality is currently disabled in this release.", Gtk.MessageType.INFO)

    # ---------------------------------------------------------------------
    # Save / Cancel / Back
    # ---------------------------------------------------------------------

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
            result = self.ATLAS.provider_manager.update_huggingface_settings(settings)
            self._dispatch_provider_result(
                result,
                success_message="Your settings have been saved successfully.",
                failure_message="An error occurred while saving settings.",
            )
        except ValueError as ve:
            self.show_message("Invalid Input", f"Please ensure all fields are correctly filled: {str(ve)}", Gtk.MessageType.ERROR)
        except Exception as e:
            self.show_message("Error", f"An error occurred while saving settings: {str(e)}", Gtk.MessageType.ERROR)

    def on_cancel_clicked(self, widget):
        """Discard changes and close the window."""
        self.close()

    def on_back_clicked(self, widget):
        """Close the settings window."""
        self.close()

    def on_clear_cache_clicked(self, widget):
        """
        Callback for clearing the model cache.
        """
        confirmation = self.confirm_dialog("Are you sure you want to clear the Hugging Face cache?")
        if confirmation:
            try:
                result = self.ATLAS.provider_manager.clear_huggingface_cache()
                self._dispatch_provider_result(
                    result,
                    success_message="Cache cleared successfully.",
                    failure_message="Error clearing cache",
                )
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
                self._run_async(
                    self.ATLAS.provider_manager.remove_hf_model(selected_model),
                    success_callback=lambda res, model=selected_model: self._dispatch_provider_result(
                        res,
                        success_message=f"Model '{model}' removed successfully.",
                        failure_message=f"Error removing model '{model}'",
                        on_success=self.populate_model_comboboxes,
                    ),
                    error_callback=lambda e, model=selected_model: self.show_message(
                        "Error",
                        f"Error removing model '{model}': {str(e)}",
                        Gtk.MessageType.ERROR,
                    ),
                )
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
                    self.ATLAS.provider_manager.download_huggingface_model(selected_model, force=True),
                    success_callback=lambda res, model=selected_model: self._dispatch_provider_result(
                        res,
                        success_message=f"Model '{model}' updated successfully.",
                        failure_message=f"Error updating model '{model}'",
                    ),
                    error_callback=lambda e, model=selected_model: self.show_message(
                        "Error",
                        f"Error updating model '{model}': {str(e)}",
                        Gtk.MessageType.ERROR,
                    ),
                )
        else:
            self.show_message("Error", "No model selected to update.", Gtk.MessageType.ERROR)

    # ---------------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------------

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

        self._run_async(
            self.ATLAS.provider_manager.search_huggingface_models(search_query, filter_args, limit=10),
            success_callback=self._handle_search_results,
            error_callback=lambda e: self.show_message(
                "Error",
                f"An error occurred while searching for models: {str(e)}",
                Gtk.MessageType.ERROR,
            ),
        )

    def _handle_search_results(self, result):
        """Render Hugging Face search results returned from the provider manager."""

        if not isinstance(result, dict):
            self.show_message(
                "Error",
                "Unexpected response from Hugging Face search.",
                Gtk.MessageType.ERROR,
            )
            return

        if not result.get("success"):
            message = result.get("error", "An error occurred while searching for models.")
            self.show_message("Error", message, Gtk.MessageType.ERROR)
            return

        models = result.get("data") or []

        for row in list(self.results_listbox.get_children()):
            self.results_listbox.remove(row)

        if not models:
            label = Gtk.Label(label="No models found matching the criteria.")
            label.set_tooltip_text("Try different keywords or relax filters.")
            self.results_listbox.append(label)
        else:
            for model in models:
                box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
                box.set_margin_top(5)
                box.set_margin_bottom(5)
                info_label = Gtk.Label()
                info_label.set_xalign(0)
                tags = model.get("tags") or []
                tags_str = ", ".join(tags) if tags else "-"
                downloads = model.get("downloads")
                likes = model.get("likes")
                downloads_str = str(downloads) if downloads is not None else "-"
                likes_str = str(likes) if likes is not None else "-"
                model_id = model.get("id", "")
                info_text = (
                    f"Model ID: {model_id}\n"
                    f"Tags: {tags_str}\n"
                    f"Downloads: {downloads_str}\n"
                    f"Likes: {likes_str}"
                )
                info_label.set_text(info_text)
                info_label.set_tooltip_text("Click Download to install this model locally.")
                box.append(info_label)
                download_button = Gtk.Button(label="Download")
                download_button.set_tooltip_text("Download and install this model.")
                download_button.connect("clicked", self.on_download_model_clicked, model_id)
                box.append(download_button)
                self.results_listbox.append(box)
        self.results_listbox.show()
        return False

    def on_download_model_clicked(self, widget, model_name):
        """
        Callback for the Download button.
        Asks for confirmation and then runs the download asynchronously.
        """
        confirmation = self.confirm_dialog(f"Do you want to download and install the model '{model_name}'?")
        if confirmation:
            self._run_async(
                self.ATLAS.provider_manager.download_huggingface_model(model_name, force=True),
                success_callback=lambda res, model=model_name: self._dispatch_provider_result(
                    res,
                    success_message=f"Model '{model}' downloaded and installed successfully.",
                    failure_message=f"Error downloading model '{model}'",
                    on_success=self.populate_model_comboboxes,
                ),
                error_callback=lambda e, model=model_name: self.show_message(
                    "Error",
                    f"Error downloading model '{model}': {str(e)}",
                    Gtk.MessageType.ERROR,
                ),
            )

    # ---------------------------------------------------------------------
    # Token actions
    # ---------------------------------------------------------------------

    def on_save_token_clicked(self, button):
        """
        Persist the Hugging Face token using the provider manager helper.
        """

        backend = getattr(getattr(self.ATLAS, "provider_manager", None), "save_huggingface_token", None)
        if not callable(backend):
            self.show_message(
                "Error",
                "Saving Hugging Face tokens is not supported in this build.",
                Gtk.MessageType.ERROR,
            )
            return

        try:
            result = backend(self.hf_token_entry.get_text())
        except Exception as exc:  # pragma: no cover - defensive UI guard
            self.show_message(
                "Error",
                f"Failed to save token: {str(exc)}",
                Gtk.MessageType.ERROR,
            )
            return

        self._dispatch_provider_result(
            result,
            success_message="Hugging Face token saved.",
            failure_message="Failed to save Hugging Face token",
            on_success=self._mark_hf_token_saved,
        )

    def _mark_hf_token_saved(self):
        """Reset the token entry to hide the persisted secret."""

        self.hf_token_entry.set_text("")
        self.hf_token_entry.set_placeholder_text("Saved")

    def on_test_token_clicked(self, button):
        """
        Try a simple whoami() call using huggingface_hub to validate the token.
        """
        token_input = self.hf_token_entry.get_text().strip()
        saved_token = self._get_saved_hf_token()
        token_to_use = token_input or saved_token

        if not token_to_use:
            self.show_message("Info", "Enter a token (or save one) before testing.", Gtk.MessageType.INFO)
            return

        self._run_async(
            self.ATLAS.provider_manager.test_huggingface_token(token_input or None),
            success_callback=self._handle_token_test_result,
            error_callback=lambda exc: self.show_message(
                "Error",
                f"Token test failed: {str(exc)}",
                Gtk.MessageType.ERROR,
            ),
        )

    # ---------------------------------------------------------------------
    # Dialog helpers
    # ---------------------------------------------------------------------

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
