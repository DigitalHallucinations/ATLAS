"""Knowledge Base Manager - Standalone window for managing RAG knowledge bases.

Provides a comprehensive management interface for:
- Listing and selecting knowledge bases
- Creating and deleting knowledge bases
- Viewing documents within a knowledge base
- Browsing chunks within documents
- Advanced features like query testing, import/export
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk

from GTKUI.KnowledgeBase.embedding_visualization import EmbeddingVisualization
from GTKUI.Utils.styled_window import AtlasWindow
from GTKUI.Utils.utils import apply_css
from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from ATLAS.config import ConfigManager
    from ATLAS.services.rag import RAGService
    from modules.storage.knowledge import KnowledgeStore
    from modules.storage.knowledge.base import KnowledgeBase, Document, Chunk, SearchResult

logger = setup_logger(__name__)


@dataclass
class KBManagerState:
    """State tracking for the KB Manager."""
    
    selected_kb_id: Optional[str] = None
    selected_document_id: Optional[str] = None
    is_loading: bool = False


class KnowledgeBaseManager(AtlasWindow):
    """Standalone window for managing RAG knowledge bases.

    Features:
    - Knowledge base list with create/delete
    - Document list for selected KB
    - Chunk browser for selected document
    - Query testing panel
    - Import/export functionality

    Usage:
        manager = KnowledgeBaseManager(
            config_manager=config_manager,
            knowledge_store=knowledge_store,
            rag_service=rag_service,
        )
        manager.present()
    """

    def __init__(
        self,
        *,
        config_manager: Optional["ConfigManager"] = None,
        knowledge_store: Optional["KnowledgeStore"] = None,
        rag_service: Optional["RAGService"] = None,
        parent: Optional[Gtk.Window] = None,
    ) -> None:
        """Initialize the Knowledge Base Manager.

        Args:
            config_manager: Configuration manager for settings.
            knowledge_store: Knowledge store for KB operations.
            rag_service: RAG service for retrieval testing.
            parent: Optional parent window.
        """
        super().__init__(
            title="Knowledge Base Manager",
            default_size=(1000, 700),
            modal=False,
            transient_for=parent,
            css_classes=["kb-manager-window"],
        )

        self._config_manager = config_manager
        self._knowledge_store = knowledge_store
        self._rag_service = rag_service
        self._state = KBManagerState()

        # Data caches
        self._knowledge_bases: List["KnowledgeBase"] = []
        self._documents: List["Document"] = []
        self._chunks: List["Chunk"] = []

        self._build_ui()
        self._load_knowledge_bases()

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        # Main horizontal paned layout
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(280)

        # Left sidebar - KB list
        left_panel = self._build_kb_list_panel()
        paned.set_start_child(left_panel)

        # Right area - content panels
        right_panel = self._build_content_area()
        paned.set_end_child(right_panel)

        # Wrap in box with header
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        
        # Header bar
        header = self._build_header()
        main_box.append(header)
        
        main_box.append(paned)
        self.set_child(main_box)

    def _build_header(self) -> Gtk.Box:
        """Build the header bar with actions."""
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header.add_css_class("toolbar")
        header.set_margin_start(12)
        header.set_margin_end(12)
        header.set_margin_top(8)
        header.set_margin_bottom(8)

        # Title
        title = Gtk.Label(label="Knowledge Base Manager")
        title.add_css_class("title-3")
        title.set_hexpand(True)
        title.set_xalign(0)
        header.append(title)

        # Refresh button
        refresh_btn = Gtk.Button.new_from_icon_name("view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Refresh")
        refresh_btn.connect("clicked", self._on_refresh_clicked)
        header.append(refresh_btn)

        # Import button
        import_btn = Gtk.Button(label="Import")
        import_btn.set_tooltip_text("Import knowledge base from file")
        import_btn.connect("clicked", self._on_import_clicked)
        header.append(import_btn)

        # Close button
        close_btn = Gtk.Button(label="Close")
        close_btn.connect("clicked", lambda b: self.close())
        header.append(close_btn)

        return header

    def _build_kb_list_panel(self) -> Gtk.Box:
        """Build the knowledge base list sidebar."""
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        panel.set_margin_start(12)
        panel.set_margin_end(8)
        panel.set_margin_top(12)
        panel.set_margin_bottom(12)
        panel.set_size_request(250, -1)

        # Header with create button
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        
        label = Gtk.Label(label="Knowledge Bases")
        label.add_css_class("heading")
        label.set_xalign(0)
        label.set_hexpand(True)
        header.append(label)

        create_btn = Gtk.Button.new_from_icon_name("list-add-symbolic")
        create_btn.add_css_class("flat")
        create_btn.set_tooltip_text("Create new knowledge base")
        create_btn.connect("clicked", self._on_create_kb_clicked)
        header.append(create_btn)

        panel.append(header)

        # Search entry
        self._kb_search = Gtk.SearchEntry()
        self._kb_search.set_placeholder_text("Search knowledge bases…")
        self._kb_search.connect("search-changed", self._on_kb_search_changed)
        panel.append(self._kb_search)

        # KB list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._kb_list = Gtk.ListBox()
        self._kb_list.add_css_class("navigation-sidebar")
        self._kb_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._kb_list.connect("row-selected", self._on_kb_selected)
        scrolled.set_child(self._kb_list)

        panel.append(scrolled)

        # Loading indicator
        self._kb_loading_spinner = Gtk.Spinner()
        self._kb_loading_spinner.set_visible(False)
        panel.append(self._kb_loading_spinner)

        # Empty state
        self._kb_empty_label = Gtk.Label(label="No knowledge bases found")
        self._kb_empty_label.add_css_class("dim-label")
        self._kb_empty_label.set_visible(False)
        panel.append(self._kb_empty_label)

        return panel

    def _build_content_area(self) -> Gtk.Box:
        """Build the main content area with document/chunk views."""
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        panel.set_margin_start(8)
        panel.set_margin_end(12)
        panel.set_margin_top(12)
        panel.set_margin_bottom(12)

        # Stack for different content views
        self._content_stack = Gtk.Stack()
        self._content_stack.set_vexpand(True)
        self._content_stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)

        # Empty state (no KB selected)
        empty_page = self._build_empty_state()
        self._content_stack.add_named(empty_page, "empty")

        # KB details page
        kb_page = self._build_kb_details_page()
        self._content_stack.add_named(kb_page, "kb_details")

        self._content_stack.set_visible_child_name("empty")
        panel.append(self._content_stack)

        return panel

    def _build_empty_state(self) -> Gtk.Box:
        """Build the empty state when no KB is selected."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_valign(Gtk.Align.CENTER)
        box.set_halign(Gtk.Align.CENTER)

        icon = Gtk.Image.new_from_icon_name("system-search-symbolic")
        icon.set_pixel_size(64)
        icon.add_css_class("dim-label")
        box.append(icon)

        label = Gtk.Label(label="Select a knowledge base")
        label.add_css_class("title-2")
        label.add_css_class("dim-label")
        box.append(label)

        hint = Gtk.Label(label="Choose a knowledge base from the sidebar to view its documents")
        hint.add_css_class("dim-label")
        box.append(hint)

        return box

    def _build_kb_details_page(self) -> Gtk.Box:
        """Build the KB details page with documents and chunks."""
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        # KB info header
        self._kb_info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._kb_info_box.add_css_class("card")
        self._kb_info_box.set_margin_bottom(8)
        
        self._kb_name_label = Gtk.Label(label="")
        self._kb_name_label.add_css_class("title-2")
        self._kb_name_label.set_xalign(0)
        self._kb_info_box.append(self._kb_name_label)

        self._kb_description_label = Gtk.Label(label="")
        self._kb_description_label.set_xalign(0)
        self._kb_description_label.set_wrap(True)
        self._kb_description_label.add_css_class("dim-label")
        self._kb_info_box.append(self._kb_description_label)

        self._kb_stats_label = Gtk.Label(label="")
        self._kb_stats_label.set_xalign(0)
        self._kb_stats_label.add_css_class("caption")
        self._kb_info_box.append(self._kb_stats_label)

        # Action buttons row
        actions_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions_box.set_margin_top(8)

        upload_btn = Gtk.Button(label="Upload Documents")
        upload_btn.add_css_class("suggested-action")
        upload_btn.connect("clicked", self._on_upload_to_kb_clicked)
        actions_box.append(upload_btn)

        configure_btn = Gtk.Button.new_from_icon_name("emblem-system-symbolic")
        configure_btn.set_tooltip_text("Configure knowledge base settings")
        configure_btn.connect("clicked", self._on_configure_kb_clicked)
        actions_box.append(configure_btn)

        export_btn = Gtk.Button(label="Export")
        export_btn.connect("clicked", self._on_export_kb_clicked)
        actions_box.append(export_btn)

        delete_btn = Gtk.Button(label="Delete")
        delete_btn.add_css_class("destructive-action")
        delete_btn.connect("clicked", self._on_delete_kb_clicked)
        actions_box.append(delete_btn)

        self._kb_info_box.append(actions_box)
        page.append(self._kb_info_box)

        # Notebook for browse/visualize/query tabs
        self._details_notebook = Gtk.Notebook()
        self._details_notebook.set_vexpand(True)

        # Tab 1: Browse (Documents + Chunks)
        browse_tab = self._build_browse_tab()
        browse_label = Gtk.Label(label="Browse")
        self._details_notebook.append_page(browse_tab, browse_label)

        # Tab 2: Embedding Visualization
        self._embedding_viz = EmbeddingVisualization(
            knowledge_store=self._knowledge_store,
            on_chunk_selected=self._on_viz_chunk_selected,
        )
        self._embedding_viz.set_margin_start(8)
        self._embedding_viz.set_margin_end(8)
        self._embedding_viz.set_margin_top(8)
        self._embedding_viz.set_margin_bottom(8)
        viz_label = Gtk.Label(label="Embeddings")
        self._details_notebook.append_page(self._embedding_viz, viz_label)

        # Tab 3: Query Testing
        query_tab = self._build_query_testing_tab()
        query_label = Gtk.Label(label="Query Test")
        self._details_notebook.append_page(query_tab, query_label)

        # Connect tab switch to refresh visualization
        self._details_notebook.connect("switch-page", self._on_details_tab_switched)

        page.append(self._details_notebook)

        return page

    def _build_browse_tab(self) -> Gtk.Paned:
        """Build the browse tab with documents and chunks paned view."""
        # Paned for documents and chunks
        content_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        content_paned.set_vexpand(True)
        content_paned.set_position(350)

        # Documents list
        docs_panel = self._build_documents_panel()
        content_paned.set_start_child(docs_panel)

        # Chunks panel
        chunks_panel = self._build_chunks_panel()
        content_paned.set_end_child(chunks_panel)

        return content_paned

    def _build_query_testing_tab(self) -> Gtk.Box:
        """Build the query testing tab for RAG retrieval testing."""
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        panel.set_margin_start(12)
        panel.set_margin_end(12)
        panel.set_margin_top(12)
        panel.set_margin_bottom(12)

        # Query input section
        query_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        
        query_label = Gtk.Label(label="Test Query")
        query_label.add_css_class("heading")
        query_label.set_xalign(0)
        query_section.append(query_label)

        query_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        
        self._query_entry = Gtk.Entry()
        self._query_entry.set_placeholder_text("Enter a query to test retrieval...")
        self._query_entry.set_hexpand(True)
        self._query_entry.connect("activate", self._on_query_activate)
        query_row.append(self._query_entry)

        # Top-K selector
        topk_label = Gtk.Label(label="Results:")
        query_row.append(topk_label)

        self._topk_spin = Gtk.SpinButton.new_with_range(1, 20, 1)
        self._topk_spin.set_value(5)
        self._topk_spin.set_tooltip_text("Number of results to retrieve")
        query_row.append(self._topk_spin)

        self._query_btn = Gtk.Button(label="Search")
        self._query_btn.add_css_class("suggested-action")
        self._query_btn.connect("clicked", self._on_query_clicked)
        query_row.append(self._query_btn)

        query_section.append(query_row)
        panel.append(query_section)

        # Results section
        results_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        results_section.set_vexpand(True)

        results_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        results_label = Gtk.Label(label="Retrieved Chunks")
        results_label.add_css_class("heading")
        results_label.set_xalign(0)
        results_label.set_hexpand(True)
        results_header.append(results_label)

        self._query_results_count = Gtk.Label(label="")
        self._query_results_count.add_css_class("dim-label")
        self._query_results_count.add_css_class("caption")
        results_header.append(self._query_results_count)

        results_section.append(results_header)

        # Results list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._query_results_list = Gtk.ListBox()
        self._query_results_list.add_css_class("boxed-list")
        self._query_results_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._query_results_list.connect("row-selected", self._on_query_result_selected)
        scrolled.set_child(self._query_results_list)

        results_section.append(scrolled)

        # Query spinner
        self._query_spinner = Gtk.Spinner()
        self._query_spinner.set_visible(False)
        results_section.append(self._query_spinner)

        # Empty state
        self._query_empty_label = Gtk.Label(label="Enter a query to test RAG retrieval")
        self._query_empty_label.add_css_class("dim-label")
        self._query_empty_label.set_valign(Gtk.Align.CENTER)
        results_section.append(self._query_empty_label)

        panel.append(results_section)

        # Result detail section
        detail_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        
        detail_label = Gtk.Label(label="Selected Chunk Detail")
        detail_label.add_css_class("heading")
        detail_label.set_xalign(0)
        detail_section.append(detail_label)

        detail_scroll = Gtk.ScrolledWindow()
        detail_scroll.set_min_content_height(120)
        detail_scroll.set_max_content_height(200)
        detail_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self._query_result_detail = Gtk.TextView()
        self._query_result_detail.set_editable(False)
        self._query_result_detail.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._query_result_detail.set_top_margin(8)
        self._query_result_detail.set_bottom_margin(8)
        self._query_result_detail.set_left_margin(8)
        self._query_result_detail.set_right_margin(8)
        self._query_result_detail.add_css_class("monospace")
        detail_scroll.set_child(self._query_result_detail)

        detail_section.append(detail_scroll)
        panel.append(detail_section)

        return panel

    def _on_query_activate(self, entry: Gtk.Entry) -> None:
        """Handle Enter key in query entry."""
        self._on_query_clicked(None)

    def _on_query_clicked(self, button: Optional[Gtk.Button]) -> None:
        """Handle query button click."""
        query = self._query_entry.get_text().strip()
        if not query or not self._state.selected_kb_id:
            return

        top_k = int(self._topk_spin.get_value())
        self._run_query(query, top_k)

    def _run_query(self, query: str, top_k: int) -> None:
        """Execute a retrieval query."""
        if not self._knowledge_store:
            return

        self._query_btn.set_sensitive(False)
        self._query_spinner.set_visible(True)
        self._query_spinner.start()
        self._query_empty_label.set_visible(False)

        # Clear previous results
        while child := self._query_results_list.get_first_child():
            self._query_results_list.remove(child)

        async def do_query():
            try:
                from modules.storage.knowledge.base import SearchQuery

                search_query = SearchQuery(
                    query_text=query,
                    knowledge_base_ids=[self._state.selected_kb_id],
                    top_k=top_k,
                )

                results = await self._knowledge_store.search(search_query)

                GLib.idle_add(self._on_query_results, results)

            except Exception as e:
                logger.error("Query failed: %s", e)
                GLib.idle_add(self._on_query_error, str(e))

        asyncio.create_task(do_query())

    def _on_query_results(self, results: List["SearchResult"]) -> None:
        """Handle query results."""
        self._query_btn.set_sensitive(True)
        self._query_spinner.stop()
        self._query_spinner.set_visible(False)

        self._query_results_count.set_label(f"{len(results)} results")

        if not results:
            self._query_empty_label.set_text("No results found")
            self._query_empty_label.set_visible(True)
            return

        for i, result in enumerate(results):
            row = self._create_query_result_row(result, i)
            self._query_results_list.append(row)

    def _create_query_result_row(self, result: "SearchResult", index: int) -> Gtk.ListBoxRow:
        """Create a row for a query result."""
        row = Gtk.ListBoxRow()
        row.result_content = result.chunk.content
        row.result_score = result.score
        row.result_document = result.document_title if hasattr(result, "document_title") else "Unknown"

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # Header row with score and document
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        
        rank_label = Gtk.Label(label=f"#{index + 1}")
        rank_label.add_css_class("caption")
        rank_label.add_css_class("dim-label")
        header.append(rank_label)

        score_label = Gtk.Label(label=f"Score: {result.score:.4f}")
        score_label.add_css_class("caption")
        header.append(score_label)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        header.append(spacer)

        doc_label = Gtk.Label(label=row.result_document)
        doc_label.add_css_class("caption")
        doc_label.add_css_class("dim-label")
        doc_label.set_ellipsize(3)  # Pango.EllipsizeMode.END
        header.append(doc_label)

        box.append(header)

        # Content preview
        preview = result.chunk.content[:150] + "..." if len(result.chunk.content) > 150 else result.chunk.content
        content_label = Gtk.Label(label=preview.replace("\n", " "))
        content_label.set_xalign(0)
        content_label.set_wrap(True)
        content_label.set_max_width_chars(80)
        content_label.add_css_class("body")
        box.append(content_label)

        row.set_child(box)
        return row

    def _on_query_result_selected(self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        """Handle query result selection."""
        if row is None:
            self._query_result_detail.get_buffer().set_text("")
            return

        content = getattr(row, "result_content", "")
        score = getattr(row, "result_score", 0)
        doc = getattr(row, "result_document", "Unknown")

        detail_text = f"Document: {doc}\nScore: {score:.4f}\n\n---\n\n{content}"
        self._query_result_detail.get_buffer().set_text(detail_text)

    def _on_query_error(self, error: str) -> None:
        """Handle query error."""
        self._query_btn.set_sensitive(True)
        self._query_spinner.stop()
        self._query_spinner.set_visible(False)
        self._query_empty_label.set_text(f"Error: {error}")
        self._query_empty_label.set_visible(True)

    def _on_details_tab_switched(self, notebook: Gtk.Notebook, page: Gtk.Widget, page_num: int) -> None:
        """Handle tab switch - refresh visualization when switching to embeddings tab."""
        if page_num == 1 and self._state.selected_kb_id:
            # Switching to embeddings tab - refresh visualization
            self._embedding_viz.set_knowledge_base(self._state.selected_kb_id)

    def _on_viz_chunk_selected(self, chunk_id: str) -> None:
        """Handle chunk selection from visualization."""
        # Find and select the corresponding chunk in the list
        for row in self._chunk_list:
            if hasattr(row, "chunk_id") and row.chunk_id == chunk_id:
                self._chunk_list.select_row(row)
                # Switch to browse tab to show the chunk
                self._details_notebook.set_current_page(0)
                break

    def _build_documents_panel(self) -> Gtk.Box:
        """Build the documents list panel."""
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        panel.set_size_request(300, -1)

        # Header
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        
        label = Gtk.Label(label="Documents")
        label.add_css_class("heading")
        label.set_xalign(0)
        label.set_hexpand(True)
        header.append(label)

        self._doc_count_label = Gtk.Label(label="0")
        self._doc_count_label.add_css_class("caption")
        self._doc_count_label.add_css_class("dim-label")
        header.append(self._doc_count_label)

        panel.append(header)

        # Search
        self._doc_search = Gtk.SearchEntry()
        self._doc_search.set_placeholder_text("Search documents…")
        self._doc_search.connect("search-changed", self._on_doc_search_changed)
        panel.append(self._doc_search)

        # Documents list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._doc_list = Gtk.ListBox()
        self._doc_list.add_css_class("boxed-list")
        self._doc_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._doc_list.connect("row-selected", self._on_document_selected)
        scrolled.set_child(self._doc_list)

        panel.append(scrolled)

        # Empty state
        self._doc_empty_label = Gtk.Label(label="No documents in this knowledge base")
        self._doc_empty_label.add_css_class("dim-label")
        self._doc_empty_label.set_visible(False)
        panel.append(self._doc_empty_label)

        return panel

    def _build_chunks_panel(self) -> Gtk.Box:
        """Build the chunks browser panel."""
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        panel.set_margin_start(8)

        # Header
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        
        label = Gtk.Label(label="Chunks")
        label.add_css_class("heading")
        label.set_xalign(0)
        label.set_hexpand(True)
        header.append(label)

        self._chunk_count_label = Gtk.Label(label="0")
        self._chunk_count_label.add_css_class("caption")
        self._chunk_count_label.add_css_class("dim-label")
        header.append(self._chunk_count_label)

        panel.append(header)

        # Chunks list
        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._chunk_list = Gtk.ListBox()
        self._chunk_list.add_css_class("boxed-list")
        self._chunk_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._chunk_list.connect("row-selected", self._on_chunk_selected)
        scrolled.set_child(self._chunk_list)

        panel.append(scrolled)

        # Chunk editor section
        editor_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        editor_header.set_margin_top(8)

        editor_label = Gtk.Label(label="Chunk Editor")
        editor_label.add_css_class("heading")
        editor_label.set_xalign(0)
        editor_label.set_hexpand(True)
        editor_header.append(editor_label)

        # Edit mode toggle
        self._edit_mode_switch = Gtk.Switch()
        self._edit_mode_switch.set_valign(Gtk.Align.CENTER)
        self._edit_mode_switch.set_tooltip_text("Toggle edit mode")
        self._edit_mode_switch.connect("notify::active", self._on_edit_mode_toggled)
        editor_header.append(self._edit_mode_switch)

        panel.append(editor_header)

        # Editor text view
        editor_scrolled = Gtk.ScrolledWindow()
        editor_scrolled.set_min_content_height(150)
        editor_scrolled.set_max_content_height(250)
        editor_scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)

        self._chunk_preview = Gtk.TextView()
        self._chunk_preview.set_editable(False)
        self._chunk_preview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._chunk_preview.set_top_margin(8)
        self._chunk_preview.set_bottom_margin(8)
        self._chunk_preview.set_left_margin(8)
        self._chunk_preview.set_right_margin(8)
        self._chunk_preview.add_css_class("monospace")
        editor_scrolled.set_child(self._chunk_preview)

        panel.append(editor_scrolled)

        # Editor action buttons (hidden by default)
        self._editor_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._editor_actions.set_halign(Gtk.Align.END)
        self._editor_actions.set_margin_top(4)
        self._editor_actions.set_visible(False)

        revert_btn = Gtk.Button(label="Revert")
        revert_btn.add_css_class("flat")
        revert_btn.connect("clicked", self._on_revert_chunk_clicked)
        self._editor_actions.append(revert_btn)

        save_btn = Gtk.Button(label="Save Changes")
        save_btn.add_css_class("suggested-action")
        save_btn.connect("clicked", self._on_save_chunk_clicked)
        self._editor_actions.append(save_btn)

        panel.append(self._editor_actions)

        # Track original content for revert
        self._original_chunk_content: Optional[str] = None
        self._selected_chunk_id: Optional[str] = None

        # Empty state
        self._chunk_empty_label = Gtk.Label(label="Select a document to view chunks")
        self._chunk_empty_label.add_css_class("dim-label")
        self._chunk_empty_label.set_visible(False)
        panel.append(self._chunk_empty_label)

        return panel

    def _on_edit_mode_toggled(self, switch: Gtk.Switch, _pspec: Any) -> None:
        """Handle edit mode toggle."""
        is_editing = switch.get_active()
        self._chunk_preview.set_editable(is_editing)
        self._editor_actions.set_visible(is_editing)
        
        if is_editing:
            self._chunk_preview.add_css_class("editing")
        else:
            self._chunk_preview.remove_css_class("editing")

    def _on_revert_chunk_clicked(self, button: Gtk.Button) -> None:
        """Revert chunk content to original."""
        if self._original_chunk_content is not None:
            self._chunk_preview.get_buffer().set_text(self._original_chunk_content)

    def _on_save_chunk_clicked(self, button: Gtk.Button) -> None:
        """Save edited chunk content."""
        if not self._selected_chunk_id or not self._knowledge_store:
            return

        buffer = self._chunk_preview.get_buffer()
        start, end = buffer.get_bounds()
        new_content = buffer.get_text(start, end, False)

        async def do_save():
            try:
                await self._knowledge_store.update_chunk(
                    chunk_id=self._selected_chunk_id,
                    content=new_content,
                )
                # Update original content after successful save
                self._original_chunk_content = new_content
                GLib.idle_add(self._on_chunk_saved)
            except Exception as e:
                logger.error("Failed to save chunk: %s", e)
                GLib.idle_add(self._on_chunk_save_error, str(e))

        asyncio.create_task(do_save())

    def _on_chunk_saved(self) -> None:
        """Handle successful chunk save."""
        # Update the chunk in the list
        if self._state.selected_document_id:
            self._load_chunks(self._state.selected_document_id)
        # Turn off edit mode
        self._edit_mode_switch.set_active(False)

    def _on_chunk_save_error(self, error: str) -> None:
        """Handle chunk save error."""
        logger.warning("Chunk save failed: %s", error)

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_knowledge_bases(self) -> None:
        """Load knowledge bases from store."""
        if not self._knowledge_store:
            self._show_no_store_warning()
            return

        self._kb_loading_spinner.set_visible(True)
        self._kb_loading_spinner.start()

        async def fetch():
            try:
                kbs = await self._knowledge_store.list_knowledge_bases()
                GLib.idle_add(self._on_kbs_loaded, kbs)
            except Exception as e:
                logger.error("Failed to load knowledge bases: %s", e)
                GLib.idle_add(self._on_kbs_load_error, str(e))

        asyncio.create_task(fetch())

    def _on_kbs_loaded(self, kbs: List["KnowledgeBase"]) -> None:
        """Handle loaded knowledge bases."""
        self._kb_loading_spinner.stop()
        self._kb_loading_spinner.set_visible(False)
        
        self._knowledge_bases = kbs
        self._refresh_kb_list()

    def _on_kbs_load_error(self, error: str) -> None:
        """Handle KB loading error."""
        self._kb_loading_spinner.stop()
        self._kb_loading_spinner.set_visible(False)
        
        self._kb_empty_label.set_label(f"Error: {error[:50]}...")
        self._kb_empty_label.set_visible(True)

    def _show_no_store_warning(self) -> None:
        """Show warning when knowledge store is not available."""
        self._kb_empty_label.set_label("Knowledge store not available")
        self._kb_empty_label.set_visible(True)

    def _refresh_kb_list(self, filter_text: str = "") -> None:
        """Refresh the KB list display."""
        # Clear existing rows
        while True:
            child = self._kb_list.get_first_child()
            if child is None:
                break
            self._kb_list.remove(child)

        # Filter and add KBs
        filtered = self._knowledge_bases
        if filter_text:
            filter_lower = filter_text.lower()
            filtered = [
                kb for kb in self._knowledge_bases
                if filter_lower in kb.name.lower() or filter_lower in (kb.description or "").lower()
            ]

        if not filtered:
            self._kb_empty_label.set_visible(True)
            return

        self._kb_empty_label.set_visible(False)

        for kb in filtered:
            row = self._create_kb_row(kb)
            self._kb_list.append(row)

    def _create_kb_row(self, kb: "KnowledgeBase") -> Gtk.ListBoxRow:
        """Create a row for the KB list."""
        row = Gtk.ListBoxRow()
        row.kb_id = kb.id  # Store KB ID on row for selection handling

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # Name
        name_label = Gtk.Label(label=kb.name)
        name_label.set_xalign(0)
        name_label.add_css_class("heading")
        box.append(name_label)

        # Stats
        stats = f"{kb.document_count} docs • {kb.chunk_count} chunks"
        stats_label = Gtk.Label(label=stats)
        stats_label.set_xalign(0)
        stats_label.add_css_class("caption")
        stats_label.add_css_class("dim-label")
        box.append(stats_label)

        row.set_child(box)
        return row

    def _load_documents(self, kb_id: str) -> None:
        """Load documents for a knowledge base."""
        if not self._knowledge_store:
            return

        async def fetch():
            try:
                docs = await self._knowledge_store.list_documents(kb_id)
                GLib.idle_add(self._on_documents_loaded, docs)
            except Exception as e:
                logger.error("Failed to load documents: %s", e)
                GLib.idle_add(self._on_documents_load_error, str(e))

        asyncio.create_task(fetch())

    def _on_documents_loaded(self, docs: List["Document"]) -> None:
        """Handle loaded documents."""
        self._documents = docs
        self._refresh_doc_list()

    def _on_documents_load_error(self, error: str) -> None:
        """Handle document loading error."""
        self._doc_empty_label.set_label(f"Error: {error[:50]}...")
        self._doc_empty_label.set_visible(True)

    def _refresh_doc_list(self, filter_text: str = "") -> None:
        """Refresh the document list display."""
        # Clear existing rows
        while True:
            child = self._doc_list.get_first_child()
            if child is None:
                break
            self._doc_list.remove(child)

        # Filter
        filtered = self._documents
        if filter_text:
            filter_lower = filter_text.lower()
            filtered = [
                doc for doc in self._documents
                if filter_lower in doc.title.lower()
            ]

        self._doc_count_label.set_label(str(len(filtered)))

        if not filtered:
            self._doc_empty_label.set_visible(True)
            return

        self._doc_empty_label.set_visible(False)

        for doc in filtered:
            row = self._create_doc_row(doc)
            self._doc_list.append(row)

    def _create_doc_row(self, doc: "Document") -> Gtk.ListBoxRow:
        """Create a row for the document list."""
        row = Gtk.ListBoxRow()
        row.document_id = doc.id

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(6)
        box.set_margin_bottom(6)

        # Title
        title_label = Gtk.Label(label=doc.title)
        title_label.set_xalign(0)
        title_label.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        box.append(title_label)

        # Meta
        chunk_info = f"{doc.chunk_count} chunks • {doc.token_count} tokens"
        meta_label = Gtk.Label(label=chunk_info)
        meta_label.set_xalign(0)
        meta_label.add_css_class("caption")
        meta_label.add_css_class("dim-label")
        box.append(meta_label)

        row.set_child(box)
        return row

    def _load_chunks(self, document_id: str) -> None:
        """Load chunks for a document."""
        if not self._knowledge_store:
            return

        async def fetch():
            try:
                chunks = await self._knowledge_store.get_chunks_by_document(document_id)
                GLib.idle_add(self._on_chunks_loaded, chunks)
            except Exception as e:
                logger.error("Failed to load chunks: %s", e)
                GLib.idle_add(self._on_chunks_load_error, str(e))

        asyncio.create_task(fetch())

    def _on_chunks_loaded(self, chunks: List["Chunk"]) -> None:
        """Handle loaded chunks."""
        self._chunks = chunks
        self._refresh_chunk_list()

    def _on_chunks_load_error(self, error: str) -> None:
        """Handle chunk loading error."""
        self._chunk_empty_label.set_label(f"Error: {error[:50]}...")
        self._chunk_empty_label.set_visible(True)

    def _refresh_chunk_list(self) -> None:
        """Refresh the chunk list display."""
        # Clear existing rows
        while True:
            child = self._chunk_list.get_first_child()
            if child is None:
                break
            self._chunk_list.remove(child)

        self._chunk_count_label.set_label(str(len(self._chunks)))

        if not self._chunks:
            self._chunk_empty_label.set_visible(True)
            return

        self._chunk_empty_label.set_visible(False)

        for i, chunk in enumerate(self._chunks):
            row = self._create_chunk_row(chunk, i)
            self._chunk_list.append(row)

    def _create_chunk_row(self, chunk: "Chunk", index: int) -> Gtk.ListBoxRow:
        """Create a row for the chunk list."""
        row = Gtk.ListBoxRow()
        row.chunk_id = chunk.id
        row.chunk_content = chunk.content

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(6)
        box.set_margin_bottom(6)

        # Index
        index_label = Gtk.Label(label=f"#{index + 1}")
        index_label.add_css_class("caption")
        index_label.add_css_class("dim-label")
        index_label.set_size_request(40, -1)
        box.append(index_label)

        # Preview
        preview = chunk.content[:60].replace("\n", " ")
        if len(chunk.content) > 60:
            preview += "..."
        preview_label = Gtk.Label(label=preview)
        preview_label.set_xalign(0)
        preview_label.set_ellipsize(3)
        preview_label.set_hexpand(True)
        box.append(preview_label)

        # Token count
        tokens_label = Gtk.Label(label=f"{chunk.token_count} tok")
        tokens_label.add_css_class("caption")
        tokens_label.add_css_class("dim-label")
        box.append(tokens_label)

        row.set_child(box)
        return row

    # -------------------------------------------------------------------------
    # Event handlers
    # -------------------------------------------------------------------------

    def _on_refresh_clicked(self, button: Gtk.Button) -> None:
        """Handle refresh button click."""
        self._load_knowledge_bases()

    def _on_import_clicked(self, button: Gtk.Button) -> None:
        """Handle import button click - import KB from zip file."""
        if not self._knowledge_store:
            return

        # Open file chooser dialog
        dialog = Gtk.FileDialog()
        dialog.set_title("Import Knowledge Base")

        # Add zip filter
        zip_filter = Gtk.FileFilter()
        zip_filter.set_name("ZIP Archives")
        zip_filter.add_pattern("*.zip")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(zip_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(zip_filter)

        dialog.open(self, None, self._on_import_file_selected)

    def _on_import_file_selected(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        """Handle import file selection."""
        try:
            file = dialog.open_finish(result)
            if file:
                import_path = file.get_path()
                self._import_knowledge_base(import_path)
        except GLib.Error as e:
            if "Dismissed" not in str(e):
                logger.error("Import dialog error: %s", e)

    def _import_knowledge_base(self, import_path: str) -> None:
        """Import a knowledge base from a zip file."""
        import json
        import zipfile
        from modules.storage.knowledge.base import DocumentType

        if not self._knowledge_store:
            return

        async def do_import():
            try:
                # Read zip file
                with zipfile.ZipFile(import_path, "r") as zf:
                    manifest_data = zf.read("manifest.json")
                    export_data = json.loads(manifest_data.decode("utf-8"))

                # Validate version
                version = export_data.get("version", "1.0")
                if version not in ("1.0",):
                    GLib.idle_add(
                        self._show_import_error, f"Unsupported export version: {version}"
                    )
                    return

                kb_data = export_data["knowledge_base"]

                # Create new knowledge base
                kb = await self._knowledge_store.create_knowledge_base(
                    name=kb_data["name"] + " (Imported)",
                    description=kb_data.get("description", ""),
                    embedding_model=kb_data.get("embedding_model", "text-embedding-3-small"),
                    metadata={
                        "imported_from": import_path,
                        "imported_at": datetime.now().isoformat(),
                        "original_export_time": export_data.get("exported_at"),
                    },
                )

                # Import documents - they will be auto-chunked and embedded
                doc_count = 0

                for doc_data in export_data.get("documents", []):
                    # Parse document type
                    doc_type_str = doc_data.get("document_type", "text")
                    try:
                        doc_type = DocumentType(doc_type_str)
                    except ValueError:
                        doc_type = DocumentType.TEXT

                    # Add document - will auto-chunk and embed
                    doc = await self._knowledge_store.add_document(
                        kb_id=kb.id,
                        title=doc_data["title"],
                        content=doc_data.get("content", ""),
                        source_uri=doc_data.get("source_uri", ""),
                        document_type=doc_type,
                        metadata=doc_data.get("metadata", {}),
                        auto_chunk=True,
                        auto_embed=True,
                    )
                    doc_count += 1

                # Refresh KB to get updated chunk count
                updated_kb = await self._knowledge_store.get_knowledge_base(kb.id)

                GLib.idle_add(
                    self._on_import_complete,
                    updated_kb or kb,
                    doc_count,
                )

            except zipfile.BadZipFile:
                GLib.idle_add(self._show_import_error, "Invalid or corrupted zip file")
            except json.JSONDecodeError as e:
                GLib.idle_add(self._show_import_error, f"Invalid manifest JSON: {e}")
            except KeyError as e:
                GLib.idle_add(self._show_import_error, f"Missing required field: {e}")
            except Exception as e:
                logger.error("Failed to import KB: %s", e)
                GLib.idle_add(self._show_import_error, str(e))

        asyncio.create_task(do_import())

    def _on_import_complete(
        self, kb: "KnowledgeBase", doc_count: int
    ) -> None:
        """Handle successful import completion."""
        chunk_count = getattr(kb, "chunk_count", 0)
        logger.info("Import complete: %s with %d docs, %d chunks", kb.name, doc_count, chunk_count)
        self._knowledge_bases.append(kb)
        self._refresh_kb_list()
        
        # Show success message
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text="Import Complete",
        )
        dialog.format_secondary_text(
            f"Imported '{kb.name}' with {doc_count} documents and {chunk_count} chunks."
        )
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

    def _show_import_error(self, error: str) -> None:
        """Show import error dialog."""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Import Failed",
        )
        dialog.format_secondary_text(error)
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

    def _on_create_kb_clicked(self, button: Gtk.Button) -> None:
        """Handle create KB button click."""
        dialog = CreateKBDialog(
            knowledge_store=self._knowledge_store,
            parent=self,
            on_created=self._on_kb_created,
        )
        dialog.present()

    def _on_kb_created(self, kb: "KnowledgeBase") -> None:
        """Handle KB creation completion."""
        self._knowledge_bases.append(kb)
        self._refresh_kb_list()

    def _on_kb_search_changed(self, search: Gtk.SearchEntry) -> None:
        """Handle KB search text change."""
        self._refresh_kb_list(search.get_text())

    def _on_kb_selected(self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        """Handle KB selection."""
        if row is None:
            self._state.selected_kb_id = None
            self._content_stack.set_visible_child_name("empty")
            return

        kb_id = getattr(row, "kb_id", None)
        if kb_id:
            self._state.selected_kb_id = kb_id
            self._show_kb_details(kb_id)

    def _show_kb_details(self, kb_id: str) -> None:
        """Show details for the selected KB."""
        kb = next((k for k in self._knowledge_bases if k.id == kb_id), None)
        if not kb:
            return

        self._kb_name_label.set_label(kb.name)
        self._kb_description_label.set_label(kb.description or "No description")
        self._kb_stats_label.set_label(
            f"{kb.document_count} documents • {kb.chunk_count} chunks • "
            f"Model: {kb.embedding_model}"
        )

        self._content_stack.set_visible_child_name("kb_details")
        self._load_documents(kb_id)
        
        # Reset embedding visualization - will load when tab is switched
        self._embedding_viz.set_knowledge_base(None)
        self._details_notebook.set_current_page(0)  # Start on browse tab

    def _on_doc_search_changed(self, search: Gtk.SearchEntry) -> None:
        """Handle document search text change."""
        self._refresh_doc_list(search.get_text())

    def _on_document_selected(self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        """Handle document selection."""
        if row is None:
            self._state.selected_document_id = None
            return

        doc_id = getattr(row, "document_id", None)
        if doc_id:
            self._state.selected_document_id = doc_id
            self._load_chunks(doc_id)

    def _on_chunk_selected(self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]) -> None:
        """Handle chunk selection."""
        if row is None:
            self._chunk_preview.get_buffer().set_text("")
            self._selected_chunk_id = None
            self._original_chunk_content = None
            self._edit_mode_switch.set_active(False)
            return

        content = getattr(row, "chunk_content", "")
        chunk_id = getattr(row, "chunk_id", None)
        
        self._selected_chunk_id = chunk_id
        self._original_chunk_content = content
        self._chunk_preview.get_buffer().set_text(content)
        
        # Reset edit mode when selecting a new chunk
        self._edit_mode_switch.set_active(False)

    def _on_upload_to_kb_clicked(self, button: Gtk.Button) -> None:
        """Handle upload to KB button click."""
        if not self._state.selected_kb_id:
            return

        try:
            from GTKUI.Utils.document_upload_dialog import DocumentUploadDialog

            dialog = DocumentUploadDialog(
                rag_service=self._rag_service,
                knowledge_store=self._knowledge_store,
                parent=self,
                on_complete=self._on_upload_complete,
            )
            # Pre-select the current KB in the dialog
            # TODO: Add method to DocumentUploadDialog to pre-select KB
            dialog.present()

        except Exception as e:
            logger.error("Failed to open upload dialog: %s", e)

    def _on_upload_complete(self, results: List[Dict[str, Any]]) -> None:
        """Handle upload completion - refresh documents."""
        if self._state.selected_kb_id:
            self._load_documents(self._state.selected_kb_id)

    def _on_configure_kb_clicked(self, button: Gtk.Button) -> None:
        """Handle configure KB button click."""
        if not self._state.selected_kb_id:
            return

        kb = next((k for k in self._knowledge_bases if k.id == self._state.selected_kb_id), None)
        if not kb:
            return

        dialog = KBConfigDialog(
            knowledge_base=kb,
            knowledge_store=self._knowledge_store,
            parent=self,
            on_updated=self._on_kb_updated,
        )
        dialog.present()

    def _on_kb_updated(self, kb: "KnowledgeBase") -> None:
        """Handle KB configuration update."""
        # Update local cache
        for i, cached_kb in enumerate(self._knowledge_bases):
            if cached_kb.id == kb.id:
                self._knowledge_bases[i] = kb
                break
        
        # Refresh display
        self._show_kb_details(kb.id)
        self._refresh_kb_list()

    def _on_export_kb_clicked(self, button: Gtk.Button) -> None:
        """Handle export KB button click - export KB to zip file."""
        if not self._state.selected_kb_id or not self._knowledge_store:
            return

        kb = next((k for k in self._knowledge_bases if k.id == self._state.selected_kb_id), None)
        if not kb:
            return

        # Open file save dialog
        dialog = Gtk.FileDialog()
        dialog.set_title(f"Export '{kb.name}'")
        dialog.set_initial_name(f"{kb.name.replace(' ', '_')}_export.zip")
        
        # Add zip filter
        zip_filter = Gtk.FileFilter()
        zip_filter.set_name("ZIP Archives")
        zip_filter.add_pattern("*.zip")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(zip_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(zip_filter)

        dialog.save(self, None, self._on_export_file_selected, kb.id)

    def _on_export_file_selected(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult, kb_id: str
    ) -> None:
        """Handle export file selection."""
        try:
            file = dialog.save_finish(result)
            if file:
                export_path = file.get_path()
                self._export_knowledge_base(kb_id, export_path)
        except GLib.Error as e:
            if "Dismissed" not in str(e):
                logger.error("Export dialog error: %s", e)

    def _export_knowledge_base(self, kb_id: str, export_path: str) -> None:
        """Export a knowledge base to a zip file."""
        import json
        import zipfile
        from datetime import datetime

        if not self._knowledge_store:
            return

        async def do_export():
            try:
                # Get KB metadata
                kb = await self._knowledge_store.get_knowledge_base(kb_id)
                if not kb:
                    GLib.idle_add(self._show_export_error, "Knowledge base not found")
                    return

                # Get all documents
                documents = await self._knowledge_store.list_documents(kb_id)

                # Build export data
                export_data = {
                    "version": "1.0",
                    "exported_at": datetime.now().isoformat(),
                    "knowledge_base": {
                        "name": kb.name,
                        "description": kb.description,
                        "embedding_model": kb.embedding_model,
                        "chunk_size": getattr(kb, "chunk_size", 500),
                        "chunk_overlap": getattr(kb, "chunk_overlap", 50),
                    },
                    "documents": [],
                }

                # Get document content - prefer full content, fall back to concatenated chunks
                for doc in documents:
                    doc_content = doc.content
                    
                    # If no content stored, reconstruct from chunks
                    if not doc_content:
                        chunks = await self._knowledge_store.get_chunks(doc.id)
                        if chunks:
                            doc_content = "\n\n".join(
                                chunk.content for chunk in sorted(chunks, key=lambda c: c.index)
                            )
                    
                    doc_data = {
                        "title": doc.title,
                        "content": doc_content or "",
                        "source_uri": getattr(doc, "source_uri", None) or doc.source,
                        "document_type": doc.document_type.value if hasattr(doc.document_type, "value") else str(doc.document_type),
                        "metadata": doc.metadata or {},
                    }
                    export_data["documents"].append(doc_data)

                # Write zip file
                with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("manifest.json", json.dumps(export_data, indent=2))

                GLib.idle_add(
                    self._show_export_success,
                    f"Exported {len(documents)} documents to {export_path}",
                )

            except Exception as e:
                logger.error("Failed to export KB: %s", e)
                GLib.idle_add(self._show_export_error, str(e))

        asyncio.create_task(do_export())

    def _show_export_success(self, message: str) -> None:
        """Show export success notification."""
        logger.info("Export complete: %s", message)
        # TODO: Add toast notification

    def _show_export_error(self, error: str) -> None:
        """Show export error dialog."""
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Export Failed",
        )
        dialog.format_secondary_text(error)
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

    def _on_delete_kb_clicked(self, button: Gtk.Button) -> None:
        """Handle delete KB button click."""
        if not self._state.selected_kb_id:
            return

        kb = next((k for k in self._knowledge_bases if k.id == self._state.selected_kb_id), None)
        if not kb:
            return

        # Show confirmation dialog
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.OK_CANCEL,
            text=f"Delete knowledge base '{kb.name}'?",
        )
        dialog.format_secondary_text(
            "This will permanently delete all documents and chunks. This action cannot be undone."
        )
        dialog.connect("response", self._on_delete_confirmed, kb.id)
        dialog.present()

    def _on_delete_confirmed(self, dialog: Gtk.MessageDialog, response: int, kb_id: str) -> None:
        """Handle delete confirmation response."""
        dialog.destroy()

        if response != Gtk.ResponseType.OK:
            return

        if not self._knowledge_store:
            return

        async def do_delete():
            try:
                await self._knowledge_store.delete_knowledge_base(kb_id, delete_documents=True)
                GLib.idle_add(self._on_kb_deleted, kb_id)
            except Exception as e:
                logger.error("Failed to delete KB: %s", e)

        asyncio.create_task(do_delete())

    def _on_kb_deleted(self, kb_id: str) -> None:
        """Handle KB deletion completion."""
        self._knowledge_bases = [kb for kb in self._knowledge_bases if kb.id != kb_id]
        self._refresh_kb_list()
        self._state.selected_kb_id = None
        self._content_stack.set_visible_child_name("empty")


class KBConfigDialog(AtlasWindow):
    """Dialog for configuring knowledge base settings."""

    def __init__(
        self,
        *,
        knowledge_base: "KnowledgeBase",
        knowledge_store: Optional["KnowledgeStore"] = None,
        parent: Optional[Gtk.Window] = None,
        on_updated: Optional[Callable[["KnowledgeBase"], None]] = None,
    ) -> None:
        super().__init__(
            title=f"Configure: {knowledge_base.name}",
            default_size=(450, 450),
            modal=True,
            transient_for=parent,
            css_classes=["kb-config-dialog"],
        )

        self._kb = knowledge_base
        self._knowledge_store = knowledge_store
        self._on_updated = on_updated

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        # Name
        name_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        name_label = Gtk.Label(label="Name")
        name_label.set_xalign(0)
        name_box.append(name_label)

        self._name_entry = Gtk.Entry()
        self._name_entry.set_text(self._kb.name)
        name_box.append(self._name_entry)

        box.append(name_box)

        # Description
        desc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        desc_label = Gtk.Label(label="Description")
        desc_label.set_xalign(0)
        desc_box.append(desc_label)

        self._desc_entry = Gtk.Entry()
        self._desc_entry.set_text(self._kb.description or "")
        desc_box.append(self._desc_entry)

        box.append(desc_box)

        # Settings frame
        settings_frame = Gtk.Frame()
        settings_frame.set_label("Document Processing Settings")
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(12)
        settings_box.set_margin_top(12)
        settings_box.set_margin_bottom(12)

        # Chunk size
        chunk_size_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        chunk_size_label = Gtk.Label(label="Chunk Size (tokens):")
        chunk_size_label.set_xalign(0)
        chunk_size_label.set_hexpand(True)
        chunk_size_box.append(chunk_size_label)

        self._chunk_size_spin = Gtk.SpinButton.new_with_range(100, 2000, 50)
        self._chunk_size_spin.set_value(self._kb.chunk_size)
        self._chunk_size_spin.set_tooltip_text("Target size for document chunks")
        chunk_size_box.append(self._chunk_size_spin)

        settings_box.append(chunk_size_box)

        # Chunk overlap
        overlap_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        overlap_label = Gtk.Label(label="Chunk Overlap (tokens):")
        overlap_label.set_xalign(0)
        overlap_label.set_hexpand(True)
        overlap_box.append(overlap_label)

        self._overlap_spin = Gtk.SpinButton.new_with_range(0, 500, 10)
        self._overlap_spin.set_value(self._kb.chunk_overlap)
        self._overlap_spin.set_tooltip_text("Overlap between adjacent chunks")
        overlap_box.append(self._overlap_spin)

        settings_box.append(overlap_box)

        # Embedding model (read-only for existing KBs)
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        model_label = Gtk.Label(label="Embedding Model:")
        model_label.set_xalign(0)
        model_label.set_hexpand(True)
        model_box.append(model_label)

        model_value = Gtk.Label(label=self._kb.embedding_model)
        model_value.add_css_class("monospace")
        model_value.add_css_class("dim-label")
        model_box.append(model_value)

        settings_box.append(model_box)

        # Info about model being read-only
        model_info = Gtk.Label(label="Note: Embedding model cannot be changed after creation.")
        model_info.add_css_class("dim-label")
        model_info.add_css_class("caption")
        model_info.set_xalign(0)
        model_info.set_wrap(True)
        settings_box.append(model_info)

        settings_frame.set_child(settings_box)
        box.append(settings_frame)

        # Stats (read-only)
        stats_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        stats_box.set_margin_top(8)

        docs_stat = Gtk.Label(label=f"📄 {self._kb.document_count} documents")
        docs_stat.add_css_class("dim-label")
        stats_box.append(docs_stat)

        chunks_stat = Gtk.Label(label=f"📦 {self._kb.chunk_count} chunks")
        chunks_stat.add_css_class("dim-label")
        stats_box.append(chunks_stat)

        box.append(stats_box)

        # Actions
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        actions.set_halign(Gtk.Align.END)
        actions.set_margin_top(16)

        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda b: self.close())
        actions.append(cancel_btn)

        save_btn = Gtk.Button(label="Save Changes")
        save_btn.add_css_class("suggested-action")
        save_btn.connect("clicked", self._on_save_clicked)
        actions.append(save_btn)

        box.append(actions)
        self.set_child(box)

    def _on_save_clicked(self, button: Gtk.Button) -> None:
        """Handle save button click."""
        name = self._name_entry.get_text().strip()
        if not name:
            return

        description = self._desc_entry.get_text().strip()
        chunk_size = int(self._chunk_size_spin.get_value())
        chunk_overlap = int(self._overlap_spin.get_value())

        if not self._knowledge_store:
            self.close()
            return

        async def do_update():
            try:
                # Update basic info through the API
                updated_kb = await self._knowledge_store.update_knowledge_base(
                    self._kb.id,
                    name=name,
                    description=description,
                    metadata={
                        **(self._kb.metadata or {}),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                )
                
                if updated_kb:
                    # Update local chunk settings (for future documents)
                    updated_kb.chunk_size = chunk_size
                    updated_kb.chunk_overlap = chunk_overlap
                    GLib.idle_add(self._on_update_complete, updated_kb)
                else:
                    GLib.idle_add(self.close)

            except Exception as e:
                logger.error("Failed to update KB: %s", e)
                GLib.idle_add(self.close)

        asyncio.create_task(do_update())

    def _on_update_complete(self, kb: "KnowledgeBase") -> None:
        """Handle update completion."""
        if self._on_updated:
            self._on_updated(kb)
        self.close()


class CreateKBDialog(AtlasWindow):
    """Dialog for creating a new knowledge base."""

    def __init__(
        self,
        *,
        knowledge_store: Optional["KnowledgeStore"] = None,
        parent: Optional[Gtk.Window] = None,
        on_created: Optional[Callable[["KnowledgeBase"], None]] = None,
    ) -> None:
        super().__init__(
            title="Create Knowledge Base",
            default_size=(400, 300),
            modal=True,
            transient_for=parent,
            css_classes=["create-kb-dialog"],
        )

        self._knowledge_store = knowledge_store
        self._on_created = on_created

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(24)
        box.set_margin_bottom(24)

        # Name
        name_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        name_label = Gtk.Label(label="Name")
        name_label.set_xalign(0)
        name_box.append(name_label)

        self._name_entry = Gtk.Entry()
        self._name_entry.set_placeholder_text("Enter knowledge base name")
        name_box.append(self._name_entry)

        box.append(name_box)

        # Description
        desc_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        desc_label = Gtk.Label(label="Description (optional)")
        desc_label.set_xalign(0)
        desc_box.append(desc_label)

        self._desc_entry = Gtk.Entry()
        self._desc_entry.set_placeholder_text("Enter description")
        desc_box.append(self._desc_entry)

        box.append(desc_box)

        # Embedding model
        model_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        model_label = Gtk.Label(label="Embedding Model")
        model_label.set_xalign(0)
        model_box.append(model_label)

        self._model_dropdown = Gtk.DropDown()
        models = Gtk.StringList.new([
            "all-MiniLM-L6-v2",
            "text-embedding-3-small",
            "text-embedding-ada-002",
            "embed-english-v3.0",
        ])
        self._model_dropdown.set_model(models)
        model_box.append(self._model_dropdown)

        box.append(model_box)

        # Actions
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        actions.set_halign(Gtk.Align.END)
        actions.set_margin_top(16)

        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda b: self.close())
        actions.append(cancel_btn)

        create_btn = Gtk.Button(label="Create")
        create_btn.add_css_class("suggested-action")
        create_btn.connect("clicked", self._on_create_clicked)
        actions.append(create_btn)

        box.append(actions)
        self.set_child(box)

    def _on_create_clicked(self, button: Gtk.Button) -> None:
        """Handle create button click."""
        name = self._name_entry.get_text().strip()
        if not name:
            return

        description = self._desc_entry.get_text().strip()
        
        model_idx = self._model_dropdown.get_selected()
        model_list = self._model_dropdown.get_model()
        model = model_list.get_string(model_idx) if model_list else "all-MiniLM-L6-v2"

        if not self._knowledge_store:
            self.close()
            return

        async def do_create():
            try:
                kb = await self._knowledge_store.create_knowledge_base(
                    name=name,
                    description=description,
                    embedding_model=model,
                )
                GLib.idle_add(self._on_created_complete, kb)
            except Exception as e:
                logger.error("Failed to create KB: %s", e)
                GLib.idle_add(self.close)

        asyncio.create_task(do_create())

    def _on_created_complete(self, kb: "KnowledgeBase") -> None:
        """Handle creation completion."""
        if self._on_created:
            self._on_created(kb)
        self.close()


__all__ = [
    "KnowledgeBaseManager",
    "CreateKBDialog",
    "KBConfigDialog",
    "KBManagerState",
]
