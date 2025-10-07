"""Tests for :mod:`GTKUI.Utils.logging`."""

from __future__ import annotations

import logging
import sys
import types


class _DummyTextIter:
    """Minimal stand-in for ``Gtk.TextIter`` used in tests."""


class _DummyTextBuffer:
    """Stub ``Gtk.TextBuffer`` implementation."""

    def insert(self, *_args, **_kwargs):
        return None

    def get_end_iter(self):
        return _DummyTextIter()

    def get_line_count(self):
        return 0

    def get_start_iter(self):
        return _DummyTextIter()

    def get_iter_at_line(self, *_args, **_kwargs):
        raise ValueError

    def delete(self, *_args, **_kwargs):
        return None

    def set_text(self, *_args, **_kwargs):
        return None


class _DummyTextView:
    """Stub ``Gtk.TextView`` implementation."""

    def scroll_to_iter(self, *_args, **_kwargs):
        return None


def _ensure_gi_stubs() -> None:
    """Install lightweight ``gi.repository`` stubs if the real ones are missing."""

    if "gi" not in sys.modules:
        gi_module = types.ModuleType("gi")
        repository_module = types.ModuleType("repository")
        gi_module.repository = repository_module
        sys.modules["gi"] = gi_module
        sys.modules["gi.repository"] = repository_module
    else:
        gi_module = sys.modules["gi"]
        repository_module = getattr(gi_module, "repository", None)
        if repository_module is None:
            repository_module = types.ModuleType("repository")
            gi_module.repository = repository_module
            sys.modules["gi.repository"] = repository_module

    repository_module = sys.modules["gi.repository"]

    if not hasattr(repository_module, "GLib"):
        def _idle_add(func, *args, **kwargs):
            func(*args, **kwargs)
            return 0

        glib_module = types.SimpleNamespace(
            PRIORITY_DEFAULT=0,
            PRIORITY_DEFAULT_IDLE=0,
            idle_add=_idle_add,
        )
        repository_module.GLib = glib_module
        sys.modules.setdefault("gi.repository.GLib", glib_module)

    if not hasattr(repository_module, "Gtk"):
        gtk_module = types.SimpleNamespace(TextBuffer=_DummyTextBuffer, TextView=_DummyTextView)
        repository_module.Gtk = gtk_module
        sys.modules.setdefault("gi.repository.Gtk", gtk_module)


def test_paused_queue_respects_max_lines() -> None:
    """Ensure the queued messages stay within the configured limit while paused."""

    _ensure_gi_stubs()

    from GTKUI.Utils.logging import GTKUILogHandler

    buffer = _DummyTextBuffer()
    handler = GTKUILogHandler(buffer, max_lines=5)
    handler.set_paused(True)

    for index in range(10):
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=f"message {index}",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

    assert len(handler._queue) == 5  # noqa: SLF001 - accessing protected member for testing.
    assert list(handler._queue) == [f"message {index}" for index in range(5, 10)]
