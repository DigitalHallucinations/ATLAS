"""RAG settings panel for the main ATLAS application.

Provides comprehensive configuration UI for all RAG subsystems:
- Master on/off toggle
- Embedding provider and model settings
- Chunking configuration
- Retrieval settings
- Reranking configuration
- Ingestion settings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager


class RAGSettingsPanel(Gtk.Box):
    """Comprehensive RAG settings panel with collapsible sections."""

    def __init__(
        self,
        config_manager: "ConfigManager",
        *,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the RAG settings panel.
        
        Args:
            config_manager: Configuration manager for reading/writing settings.
            on_change: Optional callback when settings change.
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        
        self._config_manager = config_manager
        self._on_change = on_change
        self._updating = False  # Prevent recursive updates
        
        self.set_margin_start(16)
        self.set_margin_end(16)
        self.set_margin_top(16)
        self.set_margin_bottom(16)
        
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        """Build the settings panel UI."""
        # Header with master switch
        header = self._build_header_section()
        self.append(header)
        
        # Scrollable content area
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(8)
        
        # Collapsible sections
        content.append(self._build_embeddings_section())
        content.append(self._build_chunking_section())
        content.append(self._build_retrieval_section())
        content.append(self._build_reranking_section())
        content.append(self._build_ingestion_section())
        
        scrolled.set_child(content)
        self.append(scrolled)
        
        # Action buttons
        actions = self._build_actions()
        self.append(actions)

    def _build_header_section(self) -> Gtk.Box:
        """Build the header with master toggle."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        
        # Title row with switch
        title_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        
        title = Gtk.Label(label="Knowledge Base & RAG")
        title.set_xalign(0.0)
        title.set_hexpand(True)
        if hasattr(title, "add_css_class"):
            title.add_css_class("title-1")
        title_row.append(title)
        
        self._master_switch = Gtk.Switch()
        self._master_switch.set_valign(Gtk.Align.CENTER)
        self._master_switch.connect("notify::active", self._on_master_changed)
        title_row.append(self._master_switch)
        
        box.append(title_row)
        
        # Description
        desc = Gtk.Label(
            label="Enable RAG to search your knowledge bases and include relevant context in responses."
        )
        desc.set_xalign(0.0)
        desc.set_wrap(True)
        if hasattr(desc, "add_css_class"):
            desc.add_css_class("dim-label")
        box.append(desc)
        
        # Auto-retrieve row
        auto_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        auto_row.set_margin_top(8)
        
        auto_label = Gtk.Label(label="Automatic context retrieval")
        auto_label.set_xalign(0.0)
        auto_label.set_hexpand(True)
        auto_row.append(auto_label)
        
        self._auto_switch = Gtk.Switch()
        self._auto_switch.set_valign(Gtk.Align.CENTER)
        self._auto_switch.connect("notify::active", self._on_setting_changed)
        auto_row.append(self._auto_switch)
        
        box.append(auto_row)
        
        # Max context tokens
        tokens_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        
        tokens_label = Gtk.Label(label="Max context tokens")
        tokens_label.set_xalign(0.0)
        tokens_label.set_hexpand(True)
        tokens_row.append(tokens_label)
        
        self._max_tokens_spin = Gtk.SpinButton.new_with_range(500, 16000, 500)
        self._max_tokens_spin.connect("value-changed", self._on_setting_changed)
        tokens_row.append(self._max_tokens_spin)
        
        box.append(tokens_row)
        
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(8)
        box.append(separator)
        
        return box

    def _build_embeddings_section(self) -> Gtk.Expander:
        """Build the embeddings configuration section."""
        expander = Gtk.Expander(label="Embeddings")
        expander.set_expanded(False)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        
        # Enabled switch
        enabled_row, self._embed_enabled_switch = self._create_switch_row(
            "Enable embeddings", ""
        )
        box.append(enabled_row)
        
        # Provider dropdown
        provider_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        provider_label = Gtk.Label(label="Provider")
        provider_label.set_xalign(0.0)
        provider_label.set_hexpand(True)
        provider_row.append(provider_label)
        
        self._embed_providers = ["huggingface", "openai", "cohere"]
        provider_list = Gtk.StringList.new(["HuggingFace (Local)", "OpenAI", "Cohere"])
        self._embed_provider_dropdown = Gtk.DropDown(model=provider_list)
        self._embed_provider_dropdown.connect("notify::selected", self._on_setting_changed)
        provider_row.append(self._embed_provider_dropdown)
        box.append(provider_row)
        
        # Model entry
        model_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        model_label = Gtk.Label(label="Model")
        model_label.set_xalign(0.0)
        model_label.set_hexpand(True)
        model_row.append(model_label)
        
        self._embed_model_entry = Gtk.Entry()
        self._embed_model_entry.set_placeholder_text("all-MiniLM-L6-v2")
        self._embed_model_entry.connect("changed", self._on_setting_changed)
        model_row.append(self._embed_model_entry)
        box.append(model_row)
        
        # Batch size
        batch_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        batch_label = Gtk.Label(label="Batch size")
        batch_label.set_xalign(0.0)
        batch_label.set_hexpand(True)
        batch_row.append(batch_label)
        
        self._embed_batch_spin = Gtk.SpinButton.new_with_range(1, 256, 8)
        self._embed_batch_spin.connect("value-changed", self._on_setting_changed)
        batch_row.append(self._embed_batch_spin)
        box.append(batch_row)
        
        expander.set_child(box)
        return expander

    def _build_chunking_section(self) -> Gtk.Expander:
        """Build the chunking configuration section."""
        expander = Gtk.Expander(label="Text Chunking")
        expander.set_expanded(False)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        
        # Enabled switch
        enabled_row, self._chunk_enabled_switch = self._create_switch_row(
            "Enable chunking", ""
        )
        box.append(enabled_row)
        
        # Splitter type
        splitter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        splitter_label = Gtk.Label(label="Splitter type")
        splitter_label.set_xalign(0.0)
        splitter_label.set_hexpand(True)
        splitter_row.append(splitter_label)
        
        self._chunk_splitters = ["recursive", "sentence", "semantic"]
        splitter_list = Gtk.StringList.new(["Recursive", "Sentence", "Semantic"])
        self._chunk_splitter_dropdown = Gtk.DropDown(model=splitter_list)
        self._chunk_splitter_dropdown.connect("notify::selected", self._on_setting_changed)
        splitter_row.append(self._chunk_splitter_dropdown)
        box.append(splitter_row)
        
        # Chunk size
        size_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        size_label = Gtk.Label(label="Chunk size (chars)")
        size_label.set_xalign(0.0)
        size_label.set_hexpand(True)
        size_row.append(size_label)
        
        self._chunk_size_spin = Gtk.SpinButton.new_with_range(100, 4000, 50)
        self._chunk_size_spin.connect("value-changed", self._on_setting_changed)
        size_row.append(self._chunk_size_spin)
        box.append(size_row)
        
        # Chunk overlap
        overlap_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        overlap_label = Gtk.Label(label="Chunk overlap (chars)")
        overlap_label.set_xalign(0.0)
        overlap_label.set_hexpand(True)
        overlap_row.append(overlap_label)
        
        self._chunk_overlap_spin = Gtk.SpinButton.new_with_range(0, 500, 10)
        self._chunk_overlap_spin.connect("value-changed", self._on_setting_changed)
        overlap_row.append(self._chunk_overlap_spin)
        box.append(overlap_row)
        
        expander.set_child(box)
        return expander

    def _build_retrieval_section(self) -> Gtk.Expander:
        """Build the retrieval configuration section."""
        expander = Gtk.Expander(label="Retrieval")
        expander.set_expanded(False)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        
        # Enabled switch
        enabled_row, self._retrieval_enabled_switch = self._create_switch_row(
            "Enable retrieval", ""
        )
        box.append(enabled_row)
        
        # Top K
        topk_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        topk_label = Gtk.Label(label="Top K results")
        topk_label.set_xalign(0.0)
        topk_label.set_hexpand(True)
        topk_row.append(topk_label)
        
        self._retrieval_topk_spin = Gtk.SpinButton.new_with_range(1, 50, 1)
        self._retrieval_topk_spin.connect("value-changed", self._on_setting_changed)
        topk_row.append(self._retrieval_topk_spin)
        box.append(topk_row)
        
        # Similarity threshold
        threshold_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        threshold_label = Gtk.Label(label="Similarity threshold")
        threshold_label.set_xalign(0.0)
        threshold_label.set_hexpand(True)
        threshold_row.append(threshold_label)
        
        self._retrieval_threshold_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self._retrieval_threshold_spin.set_digits(2)
        self._retrieval_threshold_spin.connect("value-changed", self._on_setting_changed)
        threshold_row.append(self._retrieval_threshold_spin)
        box.append(threshold_row)
        
        # Max context chunks
        chunks_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        chunks_label = Gtk.Label(label="Max context chunks")
        chunks_label.set_xalign(0.0)
        chunks_label.set_hexpand(True)
        chunks_row.append(chunks_label)
        
        self._retrieval_chunks_spin = Gtk.SpinButton.new_with_range(1, 20, 1)
        self._retrieval_chunks_spin.connect("value-changed", self._on_setting_changed)
        chunks_row.append(self._retrieval_chunks_spin)
        box.append(chunks_row)
        
        expander.set_child(box)
        return expander

    def _build_reranking_section(self) -> Gtk.Expander:
        """Build the reranking configuration section."""
        expander = Gtk.Expander(label="Reranking")
        expander.set_expanded(False)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        
        # Enabled switch
        enabled_row, self._rerank_enabled_switch = self._create_switch_row(
            "Enable reranking", ""
        )
        box.append(enabled_row)
        
        # Provider
        provider_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        provider_label = Gtk.Label(label="Provider")
        provider_label.set_xalign(0.0)
        provider_label.set_hexpand(True)
        provider_row.append(provider_label)
        
        self._rerank_providers = ["none", "cross_encoder", "cohere"]
        provider_list = Gtk.StringList.new(["None", "Cross-Encoder (Local)", "Cohere"])
        self._rerank_provider_dropdown = Gtk.DropDown(model=provider_list)
        self._rerank_provider_dropdown.connect("notify::selected", self._on_setting_changed)
        provider_row.append(self._rerank_provider_dropdown)
        box.append(provider_row)
        
        # Model
        model_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        model_label = Gtk.Label(label="Model")
        model_label.set_xalign(0.0)
        model_label.set_hexpand(True)
        model_row.append(model_label)
        
        self._rerank_model_entry = Gtk.Entry()
        self._rerank_model_entry.set_placeholder_text("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._rerank_model_entry.connect("changed", self._on_setting_changed)
        model_row.append(self._rerank_model_entry)
        box.append(model_row)
        
        # Top N rerank
        topn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        topn_label = Gtk.Label(label="Top N after rerank")
        topn_label.set_xalign(0.0)
        topn_label.set_hexpand(True)
        topn_row.append(topn_label)
        
        self._rerank_topn_spin = Gtk.SpinButton.new_with_range(1, 50, 1)
        self._rerank_topn_spin.connect("value-changed", self._on_setting_changed)
        topn_row.append(self._rerank_topn_spin)
        box.append(topn_row)
        
        expander.set_child(box)
        return expander

    def _build_ingestion_section(self) -> Gtk.Expander:
        """Build the ingestion configuration section."""
        expander = Gtk.Expander(label="Document Ingestion")
        expander.set_expanded(False)
        
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        
        # Enabled switch
        enabled_row, self._ingest_enabled_switch = self._create_switch_row(
            "Enable ingestion", ""
        )
        box.append(enabled_row)
        
        # Auto-detect type
        detect_row, self._ingest_detect_switch = self._create_switch_row(
            "Auto-detect file type", ""
        )
        box.append(detect_row)
        
        # Extract metadata
        meta_row, self._ingest_meta_switch = self._create_switch_row(
            "Extract metadata", ""
        )
        box.append(meta_row)
        
        # Max file size
        size_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        size_label = Gtk.Label(label="Max file size (MB)")
        size_label.set_xalign(0.0)
        size_label.set_hexpand(True)
        size_row.append(size_label)
        
        self._ingest_size_spin = Gtk.SpinButton.new_with_range(1, 500, 10)
        self._ingest_size_spin.connect("value-changed", self._on_setting_changed)
        size_row.append(self._ingest_size_spin)
        box.append(size_row)
        
        expander.set_child(box)
        return expander

    def _build_actions(self) -> Gtk.Box:
        """Build action buttons."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_halign(Gtk.Align.END)
        box.set_margin_top(8)
        
        reset_btn = Gtk.Button(label="Reset to Defaults")
        reset_btn.connect("clicked", self._on_reset_clicked)
        box.append(reset_btn)
        
        save_btn = Gtk.Button(label="Save")
        save_btn.connect("clicked", self._on_save_clicked)
        if hasattr(save_btn, "add_css_class"):
            save_btn.add_css_class("suggested-action")
        box.append(save_btn)
        
        return box

    def _create_switch_row(
        self, title: str, subtitle: str
    ) -> tuple[Gtk.Box, Gtk.Switch]:
        """Create a row with a switch."""
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        
        label = Gtk.Label(label=title)
        label.set_xalign(0.0)
        label.set_hexpand(True)
        row.append(label)
        
        switch = Gtk.Switch()
        switch.set_valign(Gtk.Align.CENTER)
        switch.connect("notify::active", self._on_setting_changed)
        row.append(switch)
        
        return row, switch

    def _load_settings(self) -> None:
        """Load current settings into UI."""
        self._updating = True
        try:
            settings = self._config_manager.get_rag_settings()
            
            # Master settings
            self._master_switch.set_active(settings.enabled)
            self._auto_switch.set_active(settings.auto_retrieve)
            self._max_tokens_spin.set_value(settings.max_context_tokens)
            
            # Embeddings
            self._embed_enabled_switch.set_active(settings.embeddings.enabled)
            try:
                idx = self._embed_providers.index(settings.embeddings.default_provider.value)
                self._embed_provider_dropdown.set_selected(idx)
            except ValueError:
                pass
            
            # Get model based on provider
            provider = settings.embeddings.default_provider.value
            if provider == "huggingface":
                self._embed_model_entry.set_text(settings.embeddings.huggingface.model)
                self._embed_batch_spin.set_value(settings.embeddings.huggingface.batch_size)
            elif provider == "openai":
                self._embed_model_entry.set_text(settings.embeddings.openai.model)
                self._embed_batch_spin.set_value(settings.embeddings.openai.batch_size)
            elif provider == "cohere":
                self._embed_model_entry.set_text(settings.embeddings.cohere.model)
                self._embed_batch_spin.set_value(settings.embeddings.cohere.batch_size)
            
            # Chunking
            self._chunk_enabled_switch.set_active(settings.chunking.enabled)
            try:
                idx = self._chunk_splitters.index(settings.chunking.default_splitter.value)
                self._chunk_splitter_dropdown.set_selected(idx)
            except ValueError:
                pass
            self._chunk_size_spin.set_value(settings.chunking.chunk_size)
            self._chunk_overlap_spin.set_value(settings.chunking.chunk_overlap)
            
            # Retrieval
            self._retrieval_enabled_switch.set_active(settings.retrieval.enabled)
            self._retrieval_topk_spin.set_value(settings.retrieval.top_k)
            self._retrieval_threshold_spin.set_value(settings.retrieval.similarity_threshold)
            self._retrieval_chunks_spin.set_value(settings.retrieval.max_context_chunks)
            
            # Reranking
            self._rerank_enabled_switch.set_active(settings.reranking.enabled)
            try:
                idx = self._rerank_providers.index(settings.reranking.provider.value)
                self._rerank_provider_dropdown.set_selected(idx)
            except ValueError:
                pass
            self._rerank_model_entry.set_text(settings.reranking.model)
            self._rerank_topn_spin.set_value(settings.reranking.top_n_rerank)
            
            # Ingestion
            self._ingest_enabled_switch.set_active(settings.ingestion.enabled)
            self._ingest_detect_switch.set_active(settings.ingestion.auto_detect_type)
            self._ingest_meta_switch.set_active(settings.ingestion.extract_metadata)
            self._ingest_size_spin.set_value(settings.ingestion.max_file_size_mb)
            
            # Update sensitivity
            self._update_sensitivity()
            
        finally:
            self._updating = False

    def _update_sensitivity(self) -> None:
        """Update widget sensitivity based on master switch."""
        enabled = self._master_switch.get_active()
        self._auto_switch.set_sensitive(enabled)
        self._max_tokens_spin.set_sensitive(enabled)

    def _on_master_changed(self, switch: Gtk.Switch, _pspec) -> None:
        """Handle master switch change."""
        self._update_sensitivity()
        if not self._updating:
            self._on_setting_changed(switch, _pspec)

    def _on_setting_changed(self, widget: Gtk.Widget, *args) -> None:
        """Handle any setting change."""
        if self._updating:
            return
        # Mark as dirty - actual save happens on Save button
        if self._on_change:
            self._on_change()

    def _on_reset_clicked(self, button: Gtk.Button) -> None:
        """Reset to default settings."""
        from ATLAS.config import DEFAULT_RAG_SETTINGS
        self._config_manager.set_rag_settings(DEFAULT_RAG_SETTINGS)
        self._load_settings()

    def _on_save_clicked(self, button: Gtk.Button) -> None:
        """Save current settings."""
        self._save_settings()

    def _save_settings(self) -> None:
        """Save UI values to config."""
        from dataclasses import replace
        from ATLAS.config.rag import (
            RAGSettings,
            EmbeddingSettings,
            ChunkingSettings,
            RetrievalSettings,
            RerankingSettings,
            IngestionSettings,
            EmbeddingProviderType,
            TextSplitterType,
            RerankerType,
        )
        
        settings = self._config_manager.get_rag_settings()
        
        # Master settings
        settings = replace(
            settings,
            enabled=self._master_switch.get_active(),
            auto_retrieve=self._auto_switch.get_active(),
            max_context_tokens=int(self._max_tokens_spin.get_value()),
        )
        
        # Embeddings
        provider_idx = self._embed_provider_dropdown.get_selected()
        provider = EmbeddingProviderType(self._embed_providers[provider_idx])
        settings = replace(
            settings,
            embeddings=replace(
                settings.embeddings,
                enabled=self._embed_enabled_switch.get_active(),
                default_provider=provider,
            ),
        )
        
        # Chunking
        splitter_idx = self._chunk_splitter_dropdown.get_selected()
        splitter = TextSplitterType(self._chunk_splitters[splitter_idx])
        settings = replace(
            settings,
            chunking=replace(
                settings.chunking,
                enabled=self._chunk_enabled_switch.get_active(),
                default_splitter=splitter,
                chunk_size=int(self._chunk_size_spin.get_value()),
                chunk_overlap=int(self._chunk_overlap_spin.get_value()),
            ),
        )
        
        # Retrieval
        settings = replace(
            settings,
            retrieval=replace(
                settings.retrieval,
                enabled=self._retrieval_enabled_switch.get_active(),
                top_k=int(self._retrieval_topk_spin.get_value()),
                similarity_threshold=self._retrieval_threshold_spin.get_value(),
                max_context_chunks=int(self._retrieval_chunks_spin.get_value()),
            ),
        )
        
        # Reranking
        rerank_idx = self._rerank_provider_dropdown.get_selected()
        reranker = RerankerType(self._rerank_providers[rerank_idx])
        settings = replace(
            settings,
            reranking=replace(
                settings.reranking,
                enabled=self._rerank_enabled_switch.get_active(),
                provider=reranker,
                model=self._rerank_model_entry.get_text(),
                top_n_rerank=int(self._rerank_topn_spin.get_value()),
            ),
        )
        
        # Ingestion
        settings = replace(
            settings,
            ingestion=replace(
                settings.ingestion,
                enabled=self._ingest_enabled_switch.get_active(),
                auto_detect_type=self._ingest_detect_switch.get_active(),
                extract_metadata=self._ingest_meta_switch.get_active(),
                max_file_size_mb=self._ingest_size_spin.get_value(),
            ),
        )
        
        self._config_manager.set_rag_settings(settings)


__all__ = ["RAGSettingsPanel"]
