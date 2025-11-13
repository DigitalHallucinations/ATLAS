"""Reusable analytics chart widgets for persona management."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk
import cairo


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Attempt to convert a payload value into a timezone-aware datetime."""

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        result = datetime.fromtimestamp(float(value), timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        candidate = text
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            result = datetime.fromisoformat(candidate)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    result = datetime.strptime(candidate, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    else:
        return None

    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    else:
        result = result.astimezone(timezone.utc)
    return result


class _AnalyticsChartBase(Gtk.Box):
    """Convenience base class that renders a titled drawing area."""

    def __init__(self, title: str, placeholder: str) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("analytics-chart")
        self.set_hexpand(True)
        self.set_vexpand(False)

        self._title_label = Gtk.Label(label=title)
        self._title_label.set_xalign(0.0)
        self._title_label.add_css_class("chart-title")
        self.append(self._title_label)

        self._overlay = Gtk.Overlay()
        self._overlay.set_hexpand(True)
        self._overlay.set_vexpand(True)
        self.append(self._overlay)

        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_hexpand(True)
        self._drawing_area.set_vexpand(True)
        self._overlay.set_child(self._drawing_area)

        self._placeholder_label = Gtk.Label(label=placeholder)
        self._placeholder_label.set_halign(Gtk.Align.CENTER)
        self._placeholder_label.set_valign(Gtk.Align.CENTER)
        self._placeholder_label.set_wrap(True)
        self._placeholder_label.add_css_class("chart-placeholder")
        self._overlay.add_overlay(self._placeholder_label)

        self._dark_mode = False

    # ------------------------ Theme helpers ------------------------

    def set_dark_mode(self, enabled: bool) -> None:
        self._dark_mode = bool(enabled)
        self.remove_css_class("dark")
        self.remove_css_class("light")
        self.add_css_class("dark" if self._dark_mode else "light")
        self.queue_draw()

    def queue_draw(self) -> None:  # type: ignore[override]
        self._drawing_area.queue_draw()

    def _show_placeholder(self, visible: bool, message: Optional[str] = None) -> None:
        if message:
            self._placeholder_label.set_text(message)
        self._placeholder_label.set_visible(visible)

    # ------------------------- Event binding -------------------------

    def set_activate_handler(self, callback: Optional[Callable[[datetime], None]]) -> None:
        self._activate_handler = callback

    def _emit_activate(self, timestamp: Optional[datetime]) -> None:
        if timestamp is None:
            return
        handler = getattr(self, "_activate_handler", None)
        if callable(handler):
            handler(timestamp)


class AnomalyHeatmap(_AnalyticsChartBase):
    """Calendar-style heatmap visualising anomaly frequency per day."""

    def __init__(self) -> None:
        super().__init__("Anomaly Frequency", "No anomalies detected for the selected range.")
        self._drawing_area.set_content_height(160)
        self._drawing_area.set_content_width(360)
        self._drawing_area.set_draw_func(self._on_draw)  # type: ignore[arg-type]

        self._grid: List[Tuple[datetime, int, bool]] = []
        self._range_start: Optional[datetime] = None
        self._range_end: Optional[datetime] = None
        self._max_count: int = 0
        self._highlight: Optional[datetime] = None

        click = Gtk.GestureClick()
        click.set_button(0)
        click.connect("released", self._on_click)
        self._drawing_area.add_controller(click)

    # --------------------------- Public API ---------------------------

    def update_anomalies(
        self,
        entries: Iterable[Dict[str, Any]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> None:
        counts: Dict[datetime, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            timestamp = _coerce_datetime(entry.get("timestamp"))
            if timestamp is None:
                continue
            day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            counts[day] = counts.get(day, 0) + 1

        in_range_days: List[datetime] = []
        if start is not None:
            start_day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        elif counts:
            start_day = min(counts)
        else:
            start_day = None

        if end is not None:
            end_day = end.replace(hour=0, minute=0, second=0, microsecond=0)
        elif counts:
            end_day = max(counts)
        else:
            end_day = None

        if start_day and end_day and start_day <= end_day:
            current = start_day
            while current <= end_day:
                in_range_days.append(current)
                current += timedelta(days=1)
        else:
            in_range_days.extend(sorted(counts.keys()))

        self._max_count = max(counts.values(), default=0)
        if not in_range_days:
            self._grid = []
            self._range_start = None
            self._range_end = None
            self._highlight = None
            self._show_placeholder(True)
            self.queue_draw()
            return

        grid_start = in_range_days[0]
        grid_end = in_range_days[-1]
        grid_start -= timedelta(days=grid_start.weekday())
        grid_end += timedelta(days=(6 - grid_end.weekday()))

        grid: List[Tuple[datetime, int, bool]] = []
        current = grid_start
        while current <= grid_end:
            in_range = current in counts or (in_range_days[0] <= current <= in_range_days[-1])
            value = counts.get(current, 0) if in_range else 0
            grid.append((current, value, in_range and (in_range_days[0] <= current <= in_range_days[-1])))
            current += timedelta(days=1)

        self._grid = grid
        self._range_start = in_range_days[0]
        self._range_end = in_range_days[-1]
        self._highlight = None
        self._show_placeholder(False)
        self.queue_draw()

    def highlight_date(self, date_value: Optional[datetime]) -> None:
        if date_value is None:
            self._highlight = None
            self.queue_draw()
            return
        highlight = date_value.replace(hour=0, minute=0, second=0, microsecond=0)
        self._highlight = highlight
        self.queue_draw()

    # ---------------------------- Rendering ----------------------------

    def _heat_color(self, value: int) -> Tuple[float, float, float]:
        if self._max_count <= 0:
            intensity = 0.0
        else:
            intensity = max(0.0, min(1.0, value / self._max_count))
        if self._dark_mode:
            start = (0.15, 0.20, 0.24)
            end = (0.18, 0.55, 0.92)
        else:
            start = (0.90, 0.93, 0.98)
            end = (0.20, 0.45, 0.85)
        r = start[0] + (end[0] - start[0]) * intensity
        g = start[1] + (end[1] - start[1]) * intensity
        b = start[2] + (end[2] - start[2]) * intensity
        return (r, g, b)

    def _on_draw(self, _area: Gtk.DrawingArea, ctx: cairo.Context, width: int, height: int) -> None:
        if not self._grid:
            return

        total_columns = max(1, (len(self._grid) + 6) // 7)
        padding = 4.0
        usable_width = max(0.0, width - (total_columns + 1) * padding)
        usable_height = max(0.0, height - (7 + 1) * padding)
        if usable_width <= 0 or usable_height <= 0:
            return

        cell_width = usable_width / total_columns
        cell_height = usable_height / 7
        cell_size = min(cell_width, cell_height)
        cell_width = cell_height = cell_size

        for index, (day, value, in_range) in enumerate(self._grid):
            column = index // 7
            row = index % 7
            x = padding + column * (cell_width + padding)
            y = padding + row * (cell_height + padding)

            if in_range:
                r, g, b = self._heat_color(value)
                ctx.set_source_rgb(r, g, b)
            else:
                if self._dark_mode:
                    ctx.set_source_rgba(0.25, 0.27, 0.30, 0.35)
                else:
                    ctx.set_source_rgba(0.80, 0.82, 0.86, 0.35)
            ctx.rectangle(x, y, cell_width, cell_height)
            ctx.fill()

            if self._highlight and day == self._highlight:
                if self._dark_mode:
                    ctx.set_source_rgb(0.99, 0.80, 0.20)
                else:
                    ctx.set_source_rgb(0.20, 0.35, 0.70)
                ctx.set_line_width(2.0)
                ctx.rectangle(x - 1, y - 1, cell_width + 2, cell_height + 2)
                ctx.stroke()

    # --------------------------- Interaction ---------------------------

    def _on_click(self, gesture: Gtk.GestureClick, _n_press: int, x: float, y: float) -> None:
        if not self._grid:
            return
        allocation = self._drawing_area.get_allocation()
        width = allocation.width
        height = allocation.height
        total_columns = max(1, (len(self._grid) + 6) // 7)
        padding = 4.0
        usable_width = max(0.0, width - (total_columns + 1) * padding)
        usable_height = max(0.0, height - (7 + 1) * padding)
        if usable_width <= 0 or usable_height <= 0:
            return
        cell_width = usable_width / total_columns
        cell_height = usable_height / 7
        cell_size = min(cell_width, cell_height)
        cell_width = cell_height = cell_size

        column = int((x - padding) // (cell_width + padding))
        row = int((y - padding) // (cell_height + padding))
        if column < 0 or row < 0:
            return
        index = column * 7 + row
        if index < 0 or index >= len(self._grid):
            return
        day, _, in_range = self._grid[index]
        if not in_range:
            return
        self.highlight_date(day)
        self._emit_activate(day)


class LatencyTimeline(_AnalyticsChartBase):
    """Simple line chart plotting latency samples over time."""

    def __init__(self) -> None:
        super().__init__("Latency Trend", "No latency samples recorded for the selected range.")
        self._drawing_area.set_content_height(180)
        self._drawing_area.set_content_width(420)
        self._drawing_area.set_draw_func(self._on_draw)  # type: ignore[arg-type]

        self._points: List[Tuple[datetime, float]] = []
        self._highlight_index: Optional[int] = None
        self._min_latency: float = 0.0
        self._max_latency: float = 0.0

        click = Gtk.GestureClick()
        click.set_button(0)
        click.connect("released", self._on_click)
        self._drawing_area.add_controller(click)

    # --------------------------- Public API ---------------------------

    def update_samples(
        self,
        entries: Iterable[Dict[str, Any]],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> None:
        points: List[Tuple[datetime, float]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            timestamp = _coerce_datetime(entry.get("timestamp"))
            latency = entry.get("latency_ms")
            if timestamp is None or not isinstance(latency, (int, float)):
                continue
            points.append((timestamp, float(latency)))

        points.sort(key=lambda pair: pair[0])
        self._points = points
        if not points:
            self._highlight_index = None
            self._show_placeholder(True)
            self.queue_draw()
            return

        latencies = [latency for _, latency in points]
        self._min_latency = min(latencies)
        self._max_latency = max(latencies)
        if self._min_latency == self._max_latency:
            self._max_latency = self._min_latency + 1.0

        self._highlight_index = None
        self._show_placeholder(False)
        self.queue_draw()

    def highlight_timestamp(self, timestamp: Optional[datetime]) -> None:
        if not timestamp or not self._points:
            self._highlight_index = None
            self.queue_draw()
            return
        target = timestamp.astimezone(timezone.utc)
        best_index: Optional[int] = None
        best_delta: Optional[float] = None
        for index, (point_time, _latency) in enumerate(self._points):
            delta = abs((point_time - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = index
        if best_index is not None and best_delta is not None and best_delta <= 3600:
            self._highlight_index = best_index
        else:
            self._highlight_index = None
        self.queue_draw()

    # ---------------------------- Rendering ----------------------------

    def _stroke_color(self) -> Tuple[float, float, float]:
        return (0.35, 0.80, 0.95) if self._dark_mode else (0.16, 0.35, 0.70)

    def _grid_color(self) -> Tuple[float, float, float, float]:
        return (0.8, 0.82, 0.86, 0.35) if not self._dark_mode else (0.45, 0.48, 0.52, 0.45)

    def _highlight_color(self) -> Tuple[float, float, float]:
        return (0.99, 0.80, 0.20) if self._dark_mode else (0.20, 0.35, 0.70)

    def _on_draw(self, _area: Gtk.DrawingArea, ctx: cairo.Context, width: int, height: int) -> None:
        if not self._points:
            return

        padding = 24.0
        plot_width = max(1.0, width - 2 * padding)
        plot_height = max(1.0, height - 2 * padding)

        min_time = self._points[0][0]
        max_time = self._points[-1][0]
        total_seconds = max(1.0, (max_time - min_time).total_seconds())

        ctx.set_line_width(1.0)
        grid_r, grid_g, grid_b, grid_a = self._grid_color()
        ctx.set_source_rgba(grid_r, grid_g, grid_b, grid_a)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = padding + plot_height * (1.0 - fraction)
            ctx.move_to(padding, y)
            ctx.line_to(padding + plot_width, y)
        ctx.stroke()

        ctx.set_line_width(2.0)
        stroke_r, stroke_g, stroke_b = self._stroke_color()
        ctx.set_source_rgb(stroke_r, stroke_g, stroke_b)
        previous_point: Optional[Tuple[float, float]] = None

        for index, (timestamp, latency) in enumerate(self._points):
            x = padding + ((timestamp - min_time).total_seconds() / total_seconds) * plot_width
            normalized = (latency - self._min_latency) / (self._max_latency - self._min_latency)
            y = padding + plot_height * (1.0 - normalized)

            if previous_point is not None:
                ctx.move_to(previous_point[0], previous_point[1])
                ctx.line_to(x, y)
                ctx.stroke()

            previous_point = (x, y)

        ctx.set_source_rgb(stroke_r, stroke_g, stroke_b)
        ctx.set_line_width(2.0)
        previous_point = None
        for timestamp, latency in self._points:
            x = padding + ((timestamp - min_time).total_seconds() / total_seconds) * plot_width
            normalized = (latency - self._min_latency) / (self._max_latency - self._min_latency)
            y = padding + plot_height * (1.0 - normalized)
            ctx.arc(x, y, 3.5, 0, math.tau)
            ctx.fill()

        if self._highlight_index is not None and 0 <= self._highlight_index < len(self._points):
            timestamp, latency = self._points[self._highlight_index]
            x = padding + ((timestamp - min_time).total_seconds() / total_seconds) * plot_width
            normalized = (latency - self._min_latency) / (self._max_latency - self._min_latency)
            y = padding + plot_height * (1.0 - normalized)
            highlight_r, highlight_g, highlight_b = self._highlight_color()
            ctx.set_source_rgb(highlight_r, highlight_g, highlight_b)
            ctx.arc(x, y, 6.0, 0, math.tau)
            ctx.set_line_width(2.5)
            ctx.stroke()

    # --------------------------- Interaction ---------------------------

    def _on_click(self, gesture: Gtk.GestureClick, _n_press: int, x: float, _y: float) -> None:
        if not self._points:
            return
        allocation = self._drawing_area.get_allocation()
        width = allocation.width
        height = allocation.height
        padding = 24.0
        plot_width = max(1.0, width - 2 * padding)
        min_time = self._points[0][0]
        max_time = self._points[-1][0]
        total_seconds = max(1.0, (max_time - min_time).total_seconds())
        ratio = (x - padding) / plot_width
        ratio = max(0.0, min(1.0, ratio))
        target_time = min_time + timedelta(seconds=total_seconds * ratio)

        best_index: Optional[int] = None
        best_delta: Optional[float] = None
        for index, (timestamp, _latency) in enumerate(self._points):
            delta = abs((timestamp - target_time).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = index
        if best_index is None:
            return
        self._highlight_index = best_index
        self.queue_draw()
        self._emit_activate(self._points[best_index][0])
