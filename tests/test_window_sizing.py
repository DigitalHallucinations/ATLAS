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
