"""Tests for Calendar Manager GTK category widgets.

Tests the ColorChooser, CategoryDialog, and CategoryPanel GTK 4 components.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestColorChooserUnit:
    """Unit tests for ColorChooser widget (without GTK)."""

    def test_default_palette_has_12_colors(self):
        """Verify the default palette contains expected number of colors."""
        from GTKUI.Calendar_manager.color_chooser import DEFAULT_PALETTE
        
        assert len(DEFAULT_PALETTE) == 12

    def test_palette_colors_are_valid_hex(self):
        """Verify all palette colors are valid hex strings."""
        from GTKUI.Calendar_manager.color_chooser import DEFAULT_PALETTE
        
        for color in DEFAULT_PALETTE:
            assert color.startswith("#")
            assert len(color) == 7
            # Should be valid hex
            int(color[1:], 16)

    def test_palette_includes_basic_colors(self):
        """Verify palette includes essential calendar colors."""
        from GTKUI.Calendar_manager.color_chooser import DEFAULT_PALETTE
        
        # Should include blue (common default)
        assert "#4285F4" in DEFAULT_PALETTE or any(
            c.lower().startswith("#4") for c in DEFAULT_PALETTE
        )


class TestCategoryDialogUnit:
    """Unit tests for CategoryDialog constants."""

    def test_category_icons_defined(self):
        """Verify CATEGORY_ICONS constant is defined with icons."""
        from GTKUI.Calendar_manager.category_dialog import CATEGORY_ICONS
        
        assert len(CATEGORY_ICONS) >= 10
        # Verify icons are non-empty strings
        for icon in CATEGORY_ICONS:
            assert isinstance(icon, str)
            assert len(icon) > 0

    def test_icon_list_unique(self):
        """Verify all category icons are unique."""
        from GTKUI.Calendar_manager.category_dialog import CATEGORY_ICONS
        
        assert len(CATEGORY_ICONS) == len(set(CATEGORY_ICONS))


class TestCategoryPanelUnit:
    """Unit tests for CategoryPanel without GTK runtime."""

    def test_module_imports(self):
        """Verify module can be imported."""
        from GTKUI.Calendar_manager import category_panel
        
        assert hasattr(category_panel, "CategoryPanel")


class TestWidgetExports:
    """Test that all widgets are properly exported."""

    def test_color_chooser_export(self):
        """Verify ColorChooser is exported from package."""
        from GTKUI.Calendar_manager import ColorChooser, DEFAULT_PALETTE
        
        assert ColorChooser is not None
        assert DEFAULT_PALETTE is not None

    def test_category_dialog_export(self):
        """Verify CategoryDialog is exported from package."""
        from GTKUI.Calendar_manager import CategoryDialog
        
        assert CategoryDialog is not None

    def test_category_panel_export(self):
        """Verify CategoryPanel is exported from package."""
        from GTKUI.Calendar_manager import CategoryPanel
        
        assert CategoryPanel is not None


@pytest.mark.skipif(
    not pytest.importorskip("gi", reason="GTK not available"),
    reason="GTK 4 not available"
)
class TestColorChooserGTK:
    """GTK integration tests for ColorChooser (requires display)."""

    @pytest.fixture
    def mock_gtk_env(self):
        """Set up mock GTK environment."""
        # These tests require a display, skip in CI
        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            pytest.skip("No display available")

    def test_color_chooser_instantiation(self, mock_gtk_env):
        """Test ColorChooser can be instantiated."""
        from GTKUI.Calendar_manager.color_chooser import ColorChooser
        
        chooser = ColorChooser()
        assert chooser is not None

    def test_color_chooser_set_get_color(self, mock_gtk_env):
        """Test setting and getting color."""
        from GTKUI.Calendar_manager.color_chooser import ColorChooser
        
        chooser = ColorChooser()
        chooser.set_color("#FF5733")
        assert chooser.get_color() == "#FF5733"

    def test_color_chooser_default_color(self, mock_gtk_env):
        """Test default color is from palette."""
        from GTKUI.Calendar_manager.color_chooser import ColorChooser, DEFAULT_PALETTE
        
        chooser = ColorChooser()
        assert chooser.get_color() in DEFAULT_PALETTE


@pytest.mark.skipif(
    not pytest.importorskip("gi", reason="GTK not available"),
    reason="GTK 4 not available"
)
class TestCategoryDialogGTK:
    """GTK integration tests for CategoryDialog."""

    @pytest.fixture
    def mock_gtk_env(self):
        """Set up mock GTK environment."""
        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            pytest.skip("No display available")

    def test_dialog_modes(self, mock_gtk_env):
        """Test dialog can be created in add and edit modes."""
        from GTKUI.Calendar_manager.category_dialog import CategoryDialog
        from unittest.mock import MagicMock
        
        mock_atlas = MagicMock()
        add_dialog = CategoryDialog(parent=None, atlas=mock_atlas, mode="add")
        assert add_dialog is not None
        add_dialog.destroy()

        edit_dialog = CategoryDialog(parent=None, atlas=mock_atlas, mode="edit")
        assert edit_dialog is not None
        edit_dialog.destroy()


@pytest.mark.skipif(
    not pytest.importorskip("gi", reason="GTK not available"),
    reason="GTK 4 not available"
)
class TestCategoryPanelGTK:
    """GTK integration tests for CategoryPanel."""

    @pytest.fixture
    def mock_gtk_env(self):
        """Set up mock GTK environment."""
        import os
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            pytest.skip("No display available")

    @pytest.fixture
    def mock_atlas(self):
        """Create mock ATLAS instance."""
        atlas = MagicMock()
        atlas.services = {}
        return atlas

    def test_panel_instantiation(self, mock_gtk_env, mock_atlas):
        """Test CategoryPanel can be instantiated."""
        from GTKUI.Calendar_manager.category_panel import CategoryPanel
        
        panel = CategoryPanel(mock_atlas, None)
        assert panel is not None
