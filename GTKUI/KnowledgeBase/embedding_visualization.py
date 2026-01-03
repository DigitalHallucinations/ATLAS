"""Embedding visualization components for knowledge base management.

Provides 2D scatter plot visualization of high-dimensional embeddings
using dimensionality reduction (t-SNE or PCA).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk

try:
    import cairo
except ImportError:
    cairo = None  # type: ignore

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from modules.storage.knowledge import KnowledgeStore
    from modules.storage.knowledge.base import KnowledgeChunk

logger = setup_logger(__name__)


class ReductionMethod(Enum):
    """Dimensionality reduction method."""

    TSNE = "t-SNE"
    PCA = "PCA"


@dataclass
class EmbeddingPoint:
    """A point in the 2D embedding visualization."""

    chunk_id: str
    document_id: str
    document_title: str
    x: float
    y: float
    content_preview: str


class EmbeddingVisualization(Gtk.Box):
    """2D scatter plot visualization of chunk embeddings.

    Uses dimensionality reduction to project high-dimensional embeddings
    to 2D for visual exploration.

    Features:
    - t-SNE and PCA reduction methods
    - Interactive point selection
    - Tooltips with chunk content preview
    - Color coding by document
    - Zoom and pan (future)
    """

    def __init__(
        self,
        *,
        knowledge_store: Optional["KnowledgeStore"] = None,
        on_chunk_selected: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initialize the embedding visualization.

        Args:
            knowledge_store: Knowledge store for retrieving embeddings.
            on_chunk_selected: Callback when a point is clicked.
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("embedding-visualization")

        self._knowledge_store = knowledge_store
        self._on_chunk_selected = on_chunk_selected

        # Data
        self._points: List[EmbeddingPoint] = []
        self._document_colors: Dict[str, Tuple[float, float, float]] = {}
        self._selected_point_idx: Optional[int] = None
        self._hovered_point_idx: Optional[int] = None

        # State
        self._is_loading = False
        self._kb_id: Optional[str] = None
        self._reduction_method = ReductionMethod.TSNE

        # Build UI
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the visualization UI."""
        # Toolbar
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        toolbar.set_margin_bottom(4)

        # Reduction method dropdown
        method_label = Gtk.Label(label="Method:")
        toolbar.append(method_label)

        self._method_dropdown = Gtk.DropDown.new_from_strings(["t-SNE", "PCA"])
        self._method_dropdown.set_selected(0)
        self._method_dropdown.connect("notify::selected", self._on_method_changed)
        toolbar.append(self._method_dropdown)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        toolbar.append(spacer)

        # Refresh button
        refresh_btn = Gtk.Button.new_from_icon_name("view-refresh-symbolic")
        refresh_btn.set_tooltip_text("Regenerate visualization")
        refresh_btn.connect("clicked", self._on_refresh_clicked)
        toolbar.append(refresh_btn)

        self.append(toolbar)

        # Drawing area with overlay
        self._overlay = Gtk.Overlay()
        self._overlay.set_hexpand(True)
        self._overlay.set_vexpand(True)

        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_hexpand(True)
        self._drawing_area.set_vexpand(True)
        self._drawing_area.set_content_width(400)
        self._drawing_area.set_content_height(300)
        self._drawing_area.set_draw_func(self._on_draw)
        self._overlay.set_child(self._drawing_area)

        # Motion controller for hover
        motion_ctrl = Gtk.EventControllerMotion()
        motion_ctrl.connect("motion", self._on_motion)
        motion_ctrl.connect("leave", self._on_leave)
        self._drawing_area.add_controller(motion_ctrl)

        # Click controller for selection
        click_ctrl = Gtk.GestureClick()
        click_ctrl.connect("pressed", self._on_click)
        self._drawing_area.add_controller(click_ctrl)

        # Loading spinner overlay
        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(48, 48)
        self._spinner.set_halign(Gtk.Align.CENTER)
        self._spinner.set_valign(Gtk.Align.CENTER)
        self._overlay.add_overlay(self._spinner)

        # Empty state label
        self._empty_label = Gtk.Label(label="Select a knowledge base to visualize embeddings")
        self._empty_label.add_css_class("dim-label")
        self._empty_label.set_halign(Gtk.Align.CENTER)
        self._empty_label.set_valign(Gtk.Align.CENTER)
        self._overlay.add_overlay(self._empty_label)

        # Tooltip
        self._tooltip = Gtk.Label()
        self._tooltip.add_css_class("tooltip")
        self._tooltip.add_css_class("osd")
        self._tooltip.set_wrap(True)
        self._tooltip.set_max_width_chars(50)
        self._tooltip.set_visible(False)
        self._tooltip.set_halign(Gtk.Align.START)
        self._tooltip.set_valign(Gtk.Align.START)
        self._overlay.add_overlay(self._tooltip)

        self.append(self._overlay)

        # Stats bar
        self._stats_label = Gtk.Label(label="")
        self._stats_label.add_css_class("caption")
        self._stats_label.add_css_class("dim-label")
        self._stats_label.set_xalign(0)
        self.append(self._stats_label)

    def set_knowledge_base(self, kb_id: Optional[str]) -> None:
        """Set the knowledge base to visualize.

        Args:
            kb_id: Knowledge base ID or None to clear.
        """
        self._kb_id = kb_id
        self._points = []
        self._document_colors = {}
        self._selected_point_idx = None

        if kb_id:
            self._load_embeddings()
        else:
            self._empty_label.set_visible(True)
            self._drawing_area.queue_draw()

    def _on_method_changed(self, dropdown: Gtk.DropDown, _pspec: Any) -> None:
        """Handle reduction method change."""
        selected = dropdown.get_selected()
        self._reduction_method = ReductionMethod.TSNE if selected == 0 else ReductionMethod.PCA

        if self._kb_id and self._points:
            # Re-compute with new method
            self._load_embeddings()

    def _on_refresh_clicked(self, button: Gtk.Button) -> None:
        """Handle refresh button click."""
        if self._kb_id:
            self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load and process embeddings from the knowledge store."""
        if not self._knowledge_store or not self._kb_id:
            return

        self._is_loading = True
        self._empty_label.set_visible(False)
        self._spinner.set_spinning(True)
        self._spinner.set_visible(True)

        async def do_load():
            try:
                # Get all documents in the KB
                documents = await self._knowledge_store.list_documents(self._kb_id)

                # Collect all chunks with embeddings
                all_chunks: List[Tuple["KnowledgeChunk", str]] = []
                doc_titles: Dict[str, str] = {}

                for doc in documents:
                    doc_titles[doc.id] = doc.title
                    chunks = await self._knowledge_store.get_chunks(
                        doc.id, include_embeddings=True
                    )
                    for chunk in chunks:
                        if chunk.embedding:
                            all_chunks.append((chunk, doc.title))

                if not all_chunks:
                    GLib.idle_add(self._on_no_embeddings)
                    return

                # Extract embeddings for reduction
                embeddings = [chunk.embedding for chunk, _ in all_chunks]

                # Perform dimensionality reduction
                coords = await asyncio.to_thread(
                    self._reduce_dimensions, embeddings
                )

                # Create points
                points: List[EmbeddingPoint] = []
                for i, ((chunk, doc_title), (x, y)) in enumerate(zip(all_chunks, coords)):
                    preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    points.append(EmbeddingPoint(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        document_title=doc_title,
                        x=x,
                        y=y,
                        content_preview=preview,
                    ))

                # Generate colors for documents
                doc_ids = list(set(p.document_id for p in points))
                colors = self._generate_colors(len(doc_ids))
                doc_colors = dict(zip(doc_ids, colors))

                GLib.idle_add(self._on_embeddings_loaded, points, doc_colors)

            except Exception as e:
                logger.error("Failed to load embeddings: %s", e)
                GLib.idle_add(self._on_load_error, str(e))

        asyncio.create_task(do_load())

    def _reduce_dimensions(self, embeddings: List[List[float]]) -> List[Tuple[float, float]]:
        """Reduce embeddings to 2D coordinates.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            List of (x, y) coordinates.
        """
        try:
            import numpy as np

            embeddings_array = np.array(embeddings)

            if self._reduction_method == ReductionMethod.TSNE:
                from sklearn.manifold import TSNE

                # Adjust perplexity based on sample size
                n_samples = len(embeddings)
                perplexity = min(30, max(5, n_samples // 4))

                reducer = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=1000,
                )
            else:
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=2, random_state=42)

            reduced = reducer.fit_transform(embeddings_array)

            # Normalize to [0, 1] range
            min_vals = reduced.min(axis=0)
            max_vals = reduced.max(axis=0)
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1  # Avoid division by zero
            normalized = (reduced - min_vals) / ranges

            return [(float(x), float(y)) for x, y in normalized]

        except ImportError as e:
            logger.error("Missing sklearn for dimensionality reduction: %s", e)
            # Fallback: random placement
            import random
            return [(random.random(), random.random()) for _ in embeddings]

    def _generate_colors(self, n: int) -> List[Tuple[float, float, float]]:
        """Generate n distinct colors using HSV color space.

        Args:
            n: Number of colors to generate.

        Returns:
            List of RGB tuples.
        """
        colors = []
        for i in range(n):
            hue = i / max(1, n)
            # Convert HSV to RGB (saturation=0.7, value=0.8)
            h = hue * 6
            c = 0.8 * 0.7
            x = c * (1 - abs(h % 2 - 1))
            m = 0.8 - c

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors.append((r + m, g + m, b + m))

        return colors

    def _on_embeddings_loaded(
        self,
        points: List[EmbeddingPoint],
        doc_colors: Dict[str, Tuple[float, float, float]],
    ) -> None:
        """Handle successful embedding load."""
        self._points = points
        self._document_colors = doc_colors
        self._is_loading = False
        self._spinner.set_spinning(False)
        self._spinner.set_visible(False)

        n_docs = len(set(p.document_id for p in points))
        self._stats_label.set_text(
            f"{len(points)} chunks from {n_docs} documents"
        )

        self._drawing_area.queue_draw()

    def _on_no_embeddings(self) -> None:
        """Handle case with no embeddings."""
        self._is_loading = False
        self._spinner.set_spinning(False)
        self._spinner.set_visible(False)
        self._empty_label.set_text("No embeddings found. Documents may not be indexed.")
        self._empty_label.set_visible(True)

    def _on_load_error(self, error: str) -> None:
        """Handle embedding load error."""
        self._is_loading = False
        self._spinner.set_spinning(False)
        self._spinner.set_visible(False)
        self._empty_label.set_text(f"Error: {error}")
        self._empty_label.set_visible(True)

    def _on_draw(self, area: Gtk.DrawingArea, ctx: "cairo.Context", width: int, height: int) -> None:
        """Draw the embedding scatter plot."""
        if cairo is None:
            return

        # Background
        ctx.set_source_rgb(0.12, 0.12, 0.14)
        ctx.rectangle(0, 0, width, height)
        ctx.fill()

        if not self._points:
            return

        # Calculate margins and plotting area
        margin = 20
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        if plot_width <= 0 or plot_height <= 0:
            return

        # Draw grid
        ctx.set_source_rgba(0.3, 0.3, 0.3, 0.3)
        ctx.set_line_width(1)
        for i in range(5):
            # Vertical lines
            x = margin + (plot_width * i / 4)
            ctx.move_to(x, margin)
            ctx.line_to(x, height - margin)
            # Horizontal lines
            y = margin + (plot_height * i / 4)
            ctx.move_to(margin, y)
            ctx.line_to(width - margin, y)
        ctx.stroke()

        # Draw points
        point_radius = 6
        for i, point in enumerate(self._points):
            # Convert normalized coords to screen coords
            x = margin + point.x * plot_width
            y = margin + (1 - point.y) * plot_height  # Flip Y axis

            # Get color for document
            color = self._document_colors.get(point.document_id, (0.4, 0.6, 0.9))

            # Draw point
            if i == self._selected_point_idx:
                # Selected point - larger with outline
                ctx.set_source_rgb(1, 1, 1)
                ctx.arc(x, y, point_radius + 3, 0, 2 * math.pi)
                ctx.fill()
                ctx.set_source_rgb(*color)
                ctx.arc(x, y, point_radius + 1, 0, 2 * math.pi)
                ctx.fill()
            elif i == self._hovered_point_idx:
                # Hovered point - slightly larger
                ctx.set_source_rgba(*color, 1.0)
                ctx.arc(x, y, point_radius + 2, 0, 2 * math.pi)
                ctx.fill()
            else:
                # Normal point
                ctx.set_source_rgba(*color, 0.8)
                ctx.arc(x, y, point_radius, 0, 2 * math.pi)
                ctx.fill()

    def _point_at_position(self, x: float, y: float) -> Optional[int]:
        """Find point at screen position.

        Args:
            x: Screen X coordinate.
            y: Screen Y coordinate.

        Returns:
            Index of point or None.
        """
        if not self._points:
            return None

        width = self._drawing_area.get_width()
        height = self._drawing_area.get_height()
        margin = 20
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        if plot_width <= 0 or plot_height <= 0:
            return None

        hit_radius = 10  # Slightly larger than point for easier clicking

        for i, point in enumerate(self._points):
            px = margin + point.x * plot_width
            py = margin + (1 - point.y) * plot_height

            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist <= hit_radius:
                return i

        return None

    def _on_motion(self, controller: Gtk.EventControllerMotion, x: float, y: float) -> None:
        """Handle mouse motion for hover effects."""
        idx = self._point_at_position(x, y)

        if idx != self._hovered_point_idx:
            self._hovered_point_idx = idx
            self._drawing_area.queue_draw()

            if idx is not None:
                point = self._points[idx]
                self._tooltip.set_text(
                    f"📄 {point.document_title}\n\n{point.content_preview}"
                )
                self._tooltip.set_visible(True)

                # Position tooltip near cursor
                self._tooltip.set_margin_start(int(x) + 15)
                self._tooltip.set_margin_top(int(y) + 15)
            else:
                self._tooltip.set_visible(False)

    def _on_leave(self, controller: Gtk.EventControllerMotion) -> None:
        """Handle mouse leaving the drawing area."""
        if self._hovered_point_idx is not None:
            self._hovered_point_idx = None
            self._tooltip.set_visible(False)
            self._drawing_area.queue_draw()

    def _on_click(self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float) -> None:
        """Handle click to select a point."""
        idx = self._point_at_position(x, y)

        if idx != self._selected_point_idx:
            self._selected_point_idx = idx
            self._drawing_area.queue_draw()

            if idx is not None and self._on_chunk_selected:
                point = self._points[idx]
                self._on_chunk_selected(point.chunk_id)
