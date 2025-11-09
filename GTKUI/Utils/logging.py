"""GTK UI helpers for integrating Python logging with Gtk widgets."""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GLib", "2.0")

from gi.repository import GLib, Gtk


class GTKUILogHandler(logging.Handler):
    """Logging handler that dispatches records to a :class:`Gtk.TextBuffer`.

    The handler defers buffer updates to the GTK main loop via ``GLib.idle_add``
    to keep UI interactions thread-safe.
    """

    def __init__(
        self,
        buffer: Gtk.TextBuffer,
        *,
        text_view: Optional[Gtk.TextView] = None,
        max_lines: int = 2000,
        idle_priority: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._buffer = buffer
        self._text_view = text_view
        self._max_lines = max_lines if max_lines >= 0 else 0
        default_priority = getattr(GLib, "PRIORITY_DEFAULT_IDLE", GLib.PRIORITY_DEFAULT)
        self._idle_priority = idle_priority if idle_priority is not None else default_priority
        self._lock = threading.Lock()
        self._queue: Deque[str] = deque()
        self._idle_scheduled = False
        self._paused = False
        self._closed = False

    # ------------------------- Public API -------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def max_lines(self) -> int:
        with self._lock:
            return self._max_lines

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)
            should_schedule = not self._paused and self._queue and not self._idle_scheduled
        if should_schedule:
            GLib.idle_add(self._dispatch_queue, priority=self._idle_priority)
            with self._lock:
                self._idle_scheduled = True

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
        GLib.idle_add(self._clear_buffer, priority=self._idle_priority)

    def set_max_lines(self, max_lines: int) -> None:
        clamped = max(int(max_lines), 0)
        with self._lock:
            self._max_lines = clamped
            if self._paused:
                self._trim_queue_locked()
        GLib.idle_add(self._enforce_retention, priority=self._idle_priority)

    def close(self) -> None:  # noqa: D401 - standard logging Handler API.
        """Flush pending entries and mark the handler as closed."""

        try:
            with self._lock:
                self._closed = True
                self._queue.clear()
        finally:
            super().close()

    # ------------------------- logging.Handler overrides -------------------------

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        if not message:
            return

        with self._lock:
            if self._paused:
                self._queue.append(message)
                self._trim_queue_locked()
                return

            self._queue.append(message)
            if self._idle_scheduled:
                return
            self._idle_scheduled = True

        GLib.idle_add(self._dispatch_queue, priority=self._idle_priority)

    # ------------------------- Internal helpers -------------------------

    def _dispatch_queue(self) -> bool:
        if self._closed:
            with self._lock:
                self._queue.clear()
                self._idle_scheduled = False
            return False

        with self._lock:
            if self._paused or not self._queue:
                self._idle_scheduled = False
                return False
            messages = list(self._queue)
            self._queue.clear()
            self._idle_scheduled = False

        appended = "\n".join(messages) + "\n"
        buf = self._buffer
        buf.insert(buf.get_end_iter(), appended)
        self._enforce_retention()
        self._scroll_to_end()
        return False

    def _enforce_retention(self) -> bool:
        max_lines = self._max_lines
        if max_lines <= 0:
            return False

        buf = self._buffer
        total_lines = buf.get_line_count()
        if total_lines <= max_lines:
            return False

        start_iter = buf.get_start_iter()
        trim_from = total_lines - max_lines
        if trim_from <= 0:
            return False
        try:
            remove_end = buf.get_iter_at_line(trim_from)
        except ValueError:
            return False
        buf.delete(start_iter, remove_end)
        return False

    def _scroll_to_end(self) -> None:
        view = self._text_view
        if view is None:
            return
        buf = self._buffer
        end_iter = buf.get_end_iter()
        view.scroll_to_iter(end_iter, 0.0, False, 0.0, 1.0)

    def _clear_buffer(self) -> bool:
        self._buffer.set_text("")
        return False

    def _trim_queue_locked(self) -> None:
        """Ensure the pending queue respects ``self._max_lines`` when paused."""

        limit = self._max_lines
        if limit <= 0:
            return
        while len(self._queue) > limit:
            self._queue.popleft()


def read_recent_log_lines(path: Path, limit: int) -> str:
    """Return the trailing ``limit`` lines from *path* as a single string."""

    if limit <= 0:
        limit = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if limit == 0:
                return handle.read()
            window: Deque[str] = deque(maxlen=limit)
            for line in handle:
                window.append(line.rstrip("\n"))
    except OSError:
        return ""

    if not window:
        return ""
    return "\n".join(window) + "\n"


__all__ = ["GTKUILogHandler", "read_recent_log_lines"]
