"""Unit tests for the AtlasWindow sizing helpers."""

import pytest

try:  # pragma: no cover - skip when GTK bindings are unavailable
    from GTKUI.Utils.styled_window import AtlasWindow
except ModuleNotFoundError:  # pragma: no cover - environments without gi
    AtlasWindow = None
    pytest.skip("PyGObject is not installed", allow_module_level=True)


class _FakeWindow:
    """Test double that reuses the AtlasWindow sizing helpers."""

    _clamp_dimension = AtlasWindow._clamp_dimension
    _calculate_safe_size = AtlasWindow._calculate_safe_size

    def __init__(self, monitor_size):
        self._monitor_size = monitor_size

    def _get_primary_monitor_size(self):
        return self._monitor_size


def test_calculate_safe_size_clamps_to_monitor_bounds():
    window = _FakeWindow((1400, 900))
    width, height = window._calculate_safe_size(1600, 1200)

    assert width == 1336  # 1400 - 64 margin
    assert height == 836  # 900 - 64 margin


def test_calculate_safe_size_returns_desired_when_monitor_missing():
    window = _FakeWindow(None)
    width, height = window._calculate_safe_size(800, 600)

    assert width == 800
    assert height == 600


def test_primary_monitor_size_prefers_workarea(monkeypatch):
    window = _FakeWindow((0, 0))

    class _StubRect:
        def __init__(self, width, height):
            self.width = width
            self.height = height

    class _StubMonitor:
        def get_workarea(self):
            return _StubRect(1024, 700)

        def get_geometry(self):  # pragma: no cover - should not be called
            raise AssertionError("geometry should not be queried when workarea is available")

    class _StubDisplay:
        def get_primary_monitor(self):
            return _StubMonitor()

    monkeypatch.setattr(
        "GTKUI.Utils.styled_window.Gdk.Display.get_default",
        lambda: _StubDisplay(),
    )

    # The fake window delegates to AtlasWindow._get_primary_monitor_size.
    size = AtlasWindow._get_primary_monitor_size(window)
    assert size == (1024, 700)
