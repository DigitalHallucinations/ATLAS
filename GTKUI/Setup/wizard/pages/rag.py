"""RAG configuration page builder for setup wizard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

if TYPE_CHECKING:
    from GTKUI.Setup.setup_wizard import SetupWizardWindow


def _create_switch_row(
    title: str,
    subtitle: str,
    active: bool = False,
    sensitive: bool = True,
) -> tuple[Gtk.Box, Gtk.Switch]:
    """Create a row with label and switch."""
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
    row.set_hexpand(True)
    
    label_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
    label_box.set_hexpand(True)
    
    title_label = Gtk.Label(label=title)
    title_label.set_xalign(0.0)
    title_label.set_hexpand(True)
    label_box.append(title_label)
    
    if subtitle:
        subtitle_label = Gtk.Label(label=subtitle)
        subtitle_label.set_xalign(0.0)
        subtitle_label.set_wrap(True)
        if hasattr(subtitle_label, "add_css_class"):
            subtitle_label.add_css_class("dim-label")
        label_box.append(subtitle_label)
    
    row.append(label_box)
    
    switch = Gtk.Switch()
    switch.set_active(active)
    switch.set_sensitive(sensitive)
    switch.set_valign(Gtk.Align.CENTER)
    row.append(switch)
    
    return row, switch


def _create_dropdown_row(
    title: str,
    subtitle: str,
    options: list[str],
    selected: int = 0,
    sensitive: bool = True,
) -> tuple[Gtk.Box, Gtk.DropDown]:
    """Create a row with label and dropdown."""
    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
    row.set_hexpand(True)
    
    label_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
    label_box.set_hexpand(True)
    
    title_label = Gtk.Label(label=title)
    title_label.set_xalign(0.0)
    title_label.set_hexpand(True)
    label_box.append(title_label)
    
    if subtitle:
        subtitle_label = Gtk.Label(label=subtitle)
        subtitle_label.set_xalign(0.0)
        subtitle_label.set_wrap(True)
        if hasattr(subtitle_label, "add_css_class"):
            subtitle_label.add_css_class("dim-label")
        label_box.append(subtitle_label)
    
    row.append(label_box)
    
    string_list = Gtk.StringList.new(options)
    dropdown = Gtk.DropDown(model=string_list)
    dropdown.set_selected(selected)
    dropdown.set_sensitive(sensitive)
    dropdown.set_valign(Gtk.Align.CENTER)
    row.append(dropdown)
    
    return row, dropdown


def _create_info_box(message: str, style: str = "info") -> Gtk.Box:
    """Create an information box with icon and message."""
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
    box.set_hexpand(True)
    
    icon_name = {
        "info": "dialog-information-symbolic",
        "warning": "dialog-warning-symbolic", 
        "success": "emblem-ok-symbolic",
    }.get(style, "dialog-information-symbolic")
    
    icon = Gtk.Image.new_from_icon_name(icon_name)
    icon.set_valign(Gtk.Align.START)
    box.append(icon)
    
    label = Gtk.Label(label=message)
    label.set_xalign(0.0)
    label.set_wrap(True)
    label.set_hexpand(True)
    box.append(label)
    
    if hasattr(box, "add_css_class"):
        css_class = f"{style}-box" if style != "info" else "info-box"
        box.add_css_class(css_class)
    
    return box


def build_rag_intro_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    """Build the RAG introduction page."""
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
    box.set_hexpand(True)
    box.set_vexpand(True)
    
    heading = Gtk.Label(label="Knowledge Base & RAG")
    heading.set_wrap(True)
    heading.set_xalign(0.0)
    if hasattr(heading, "add_css_class"):
        heading.add_css_class("heading")
    box.append(heading)
    
    description = Gtk.Label(
        label=(
            "Retrieval-Augmented Generation (RAG) lets ATLAS search your documents "
            "and knowledge bases to provide more accurate, contextual responses. "
            "When enabled, relevant content is automatically retrieved and included "
            "in conversations."
        )
    )
    description.set_wrap(True)
    description.set_xalign(0.0)
    box.append(description)
    
    # Show preflight recommendations if available
    caps = getattr(wizard.controller.state, "rag_capabilities", None)
    if caps:
        if caps.can_use_local_embeddings:
            rec_box = _create_info_box(
                f"Detected: {caps.gpu_info.name or 'GPU'} with {caps.gpu_info.vram_gb:.1f}GB VRAM. "
                f"Local embedding models recommended.",
                "success"
            )
        else:
            rec_box = _create_info_box(
                "No GPU detected. Cloud-based embedding APIs (OpenAI, Cohere) recommended.",
                "info"
            )
        box.append(rec_box)
    
    features = Gtk.Label(
        label=(
            "\n• Semantic search across your documents\n"
            "• Automatic context injection into conversations\n"
            "• Support for local and cloud-based embeddings\n"
            "• Optional reranking for improved relevance"
        )
    )
    features.set_wrap(True)
    features.set_xalign(0.0)
    box.append(features)
    
    wizard._register_instructions(
        box,
        "Enable RAG to give ATLAS access to your knowledge bases during conversations."
    )
    
    return box


def build_rag_page(wizard: "SetupWizardWindow") -> Gtk.Widget:
    """Build the main RAG configuration page."""
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
    box.set_hexpand(True)
    box.set_vexpand(True)
    
    # Get current state
    rag_state = wizard.controller.state.rag
    caps = getattr(wizard.controller.state, "rag_capabilities", None)
    
    # Master switch section
    master_heading = Gtk.Label(label="Enable RAG")
    master_heading.set_xalign(0.0)
    if hasattr(master_heading, "add_css_class"):
        master_heading.add_css_class("heading")
    box.append(master_heading)
    
    master_row, master_switch = _create_switch_row(
        "Enable Knowledge Base & RAG",
        "Allow ATLAS to search and retrieve context from your documents",
        active=rag_state.enabled,
    )
    box.append(master_row)
    
    # Auto-retrieve switch
    auto_row, auto_switch = _create_switch_row(
        "Automatic Context Retrieval",
        "Automatically search knowledge bases for each user message",
        active=rag_state.auto_retrieve,
    )
    box.append(auto_row)
    
    # Separator
    separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
    box.append(separator)
    
    # Embedding provider section
    embed_heading = Gtk.Label(label="Embedding Provider")
    embed_heading.set_xalign(0.0)
    if hasattr(embed_heading, "add_css_class"):
        embed_heading.add_css_class("heading")
    box.append(embed_heading)
    
    # Build provider options based on capabilities
    providers = ["HuggingFace (Local)", "OpenAI", "Cohere"]
    provider_values = ["huggingface", "openai", "cohere"]
    
    # Determine default selection
    current_provider = rag_state.embedding_provider
    try:
        selected_idx = provider_values.index(current_provider)
    except ValueError:
        # Default based on capabilities
        if caps and caps.can_use_local_embeddings:
            selected_idx = 0  # HuggingFace
        else:
            selected_idx = 1  # OpenAI
    
    provider_row, provider_dropdown = _create_dropdown_row(
        "Provider",
        "Local models run on your hardware; cloud APIs require API keys",
        providers,
        selected=selected_idx,
    )
    box.append(provider_row)
    
    # Model selection section
    model_heading = Gtk.Label(label="Embedding Model")
    model_heading.set_xalign(0.0)
    if hasattr(model_heading, "add_css_class"):
        model_heading.add_css_class("heading")
    box.append(model_heading)
    
    # Model options based on provider (simplified for wizard)
    if caps and caps.recommended_models:
        # Use detected recommendations
        model_names = []
        model_values = []
        for rec in caps.recommended_models[:5]:
            display = f"{rec.model} ({rec.dimensions}d, {rec.quality_tier})"
            model_names.append(display)
            model_values.append(rec.model)
    else:
        # Default options
        model_names = [
            "all-MiniLM-L6-v2 (384d, fast)",
            "all-mpnet-base-v2 (768d, balanced)",
            "text-embedding-3-small (1536d, OpenAI)",
        ]
        model_values = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "text-embedding-3-small",
        ]
    
    try:
        model_idx = model_values.index(rag_state.embedding_model)
    except ValueError:
        model_idx = 0
    
    model_row, model_dropdown = _create_dropdown_row(
        "Model",
        "Higher dimensions = better quality but more storage/compute",
        model_names,
        selected=model_idx,
    )
    box.append(model_row)
    
    # Recommendation info
    if caps:
        if caps.can_use_hnsw_index:
            index_info = _create_info_box(
                f"pgvector {caps.pgvector_info.version} detected with HNSW support for fast searches.",
                "success"
            )
        elif caps.pgvector_info.available:
            index_info = _create_info_box(
                f"pgvector {caps.pgvector_info.version} detected. Consider upgrading to 0.5+ for HNSW indexing.",
                "info"
            )
        else:
            index_info = _create_info_box(
                "pgvector not detected. RAG requires PostgreSQL with the vector extension.",
                "warning"
            )
        box.append(index_info)
    
    # Separator
    separator2 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
    box.append(separator2)
    
    # Advanced options expander
    advanced_expander = Gtk.Expander(label="Advanced Options")
    advanced_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
    advanced_box.set_margin_start(12)
    
    # Reranking
    rerank_row, rerank_switch = _create_switch_row(
        "Enable Reranking",
        "Use cross-encoder to improve result relevance (requires GPU)",
        active=rag_state.reranking_enabled,
        sensitive=caps.can_use_reranking if caps else False,
    )
    advanced_box.append(rerank_row)
    
    # Chunking settings info
    chunk_label = Gtk.Label(
        label=f"Chunk size: {rag_state.chunk_size} chars, Overlap: {rag_state.chunk_overlap} chars"
    )
    chunk_label.set_xalign(0.0)
    if hasattr(chunk_label, "add_css_class"):
        chunk_label.add_css_class("dim-label")
    advanced_box.append(chunk_label)
    
    advanced_expander.set_child(advanced_box)
    box.append(advanced_expander)
    
    # Store widget references for apply function
    wizard._rag_master_switch = master_switch
    wizard._rag_auto_switch = auto_switch
    wizard._rag_provider_dropdown = provider_dropdown
    wizard._rag_provider_values = provider_values
    wizard._rag_model_dropdown = model_dropdown
    wizard._rag_model_values = model_values
    wizard._rag_rerank_switch = rerank_switch
    
    # Wire up master switch to enable/disable other controls
    def on_master_toggled(switch: Gtk.Switch, _pspec) -> None:
        enabled = switch.get_active()
        auto_switch.set_sensitive(enabled)
        provider_dropdown.set_sensitive(enabled)
        model_dropdown.set_sensitive(enabled)
        if caps and caps.can_use_reranking:
            rerank_switch.set_sensitive(enabled)
    
    master_switch.connect("notify::active", on_master_toggled)
    
    wizard._register_instructions(
        box,
        "Configure how ATLAS generates embeddings and retrieves context from your knowledge bases."
    )
    
    return box


def apply_rag_page(wizard: "SetupWizardWindow") -> str:
    """Apply RAG configuration from wizard page."""
    import dataclasses
    
    state = wizard.controller.state.rag
    caps = getattr(wizard.controller.state, "rag_capabilities", None)
    
    # Get values from widgets
    enabled = wizard._rag_master_switch.get_active()
    auto_retrieve = wizard._rag_auto_switch.get_active()
    
    provider_idx = wizard._rag_provider_dropdown.get_selected()
    provider = wizard._rag_provider_values[provider_idx]
    
    model_idx = wizard._rag_model_dropdown.get_selected()
    model = wizard._rag_model_values[model_idx]
    
    reranking_enabled = wizard._rag_rerank_switch.get_active()
    
    # Determine dimensions based on model
    dimensions = 384  # Default
    if caps and caps.recommended_models:
        for rec in caps.recommended_models:
            if rec.model == model:
                dimensions = rec.dimensions
                break
    else:
        dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "e5-base-v2": 768,
            "e5-large-v2": 1024,
            "bge-large-en-v1.5": 1024,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "embed-english-v3.0": 1024,
        }
        dimensions = dimension_map.get(model, 384)
    
    # Update state
    wizard.controller.state.rag = dataclasses.replace(
        state,
        enabled=enabled,
        auto_retrieve=auto_retrieve,
        embedding_provider=provider,
        embedding_model=model,
        embedding_dimensions=dimensions,
        reranking_enabled=reranking_enabled,
    )
    
    return ""


__all__ = [
    "build_rag_intro_page",
    "build_rag_page",
    "apply_rag_page",
]
