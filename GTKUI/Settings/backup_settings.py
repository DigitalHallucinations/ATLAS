from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Utils.utils import apply_css

logger = logging.getLogger(__name__)


class BackupSettings(Gtk.Box):
    """Backup/export panel for conversations and personas."""

    def __init__(self, atlas: Any) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.ATLAS = atlas

        for setter_name in ("set_margin_top", "set_margin_bottom", "set_margin_start", "set_margin_end"):
            setter = getattr(self, setter_name, None)
            if callable(setter):
                try:
                    setter(12)
                except Exception:  # pragma: no cover - GTK fallback
                    continue

        apply_css()

        self._export_entry = Gtk.Entry()
        self._export_entry.set_placeholder_text("Choose a folder for backup exports…")
        self._export_entry.set_hexpand(True)

        choose_export_button = Gtk.Button(label="Browse…")
        choose_export_button.connect("clicked", self._on_choose_export_folder)

        export_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        export_row.append(self._export_entry)
        export_row.append(choose_export_button)

        export_button = Gtk.Button(label="Export conversations and personas")
        export_button.add_css_class("suggested-action")
        export_button.connect("clicked", self._on_export_clicked)

        self._import_entry = Gtk.Entry()
        self._import_entry.set_placeholder_text("Select a backup file to import…")
        self._import_entry.set_hexpand(True)

        choose_import_button = Gtk.Button(label="Browse…")
        choose_import_button.connect("clicked", self._on_choose_import_file)

        import_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        import_row.append(self._import_entry)
        import_row.append(choose_import_button)

        import_button = Gtk.Button(label="Import backup")
        import_button.connect("clicked", self._on_import_clicked)

        self._status_label = Gtk.Label(label="")
        self._status_label.set_wrap(True)
        self._status_label.set_halign(Gtk.Align.START)

        self.append(Gtk.Label(label="Export"))
        self.append(export_row)
        self.append(export_button)
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self.append(Gtk.Label(label="Import"))
        self.append(import_row)
        self.append(import_button)
        self.append(self._status_label)

    def _update_status(self, message: str) -> None:
        self._status_label.set_text(message)

    def _select_directory(self) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, "SELECT_FOLDER", None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            self._update_status("File chooser not available in this environment.")
            return None

        dialog = chooser_cls(
            title="Select export folder",
            transient_for=self.get_root(),
            action=action_enum,
        )
        dialog.set_modal(True)
        response = dialog.run()
        try:
            if response == Gtk.ResponseType.ACCEPT:
                file_obj = dialog.get_file()
                if file_obj is not None:
                    return file_obj.get_path()
            return None
        finally:
            dialog.destroy()

    def _select_file(self) -> Optional[str]:
        chooser_cls = getattr(Gtk, "FileChooserNative", None)
        action_enum = getattr(Gtk.FileChooserAction, "OPEN", None) if hasattr(Gtk, "FileChooserAction") else None
        if chooser_cls is None or action_enum is None:
            self._update_status("File chooser not available in this environment.")
            return None

        dialog = chooser_cls(
            title="Select backup file",
            transient_for=self.get_root(),
            action=action_enum,
        )
        dialog.set_modal(True)
        response = dialog.run()
        try:
            if response == Gtk.ResponseType.ACCEPT:
                file_obj = dialog.get_file()
                if file_obj is not None:
                    return file_obj.get_path()
            return None
        finally:
            dialog.destroy()

    def _on_choose_export_folder(self, _button: Gtk.Button) -> None:
        chosen = self._select_directory()
        if chosen:
            self._export_entry.set_text(chosen)
            self._update_status("")

    def _on_choose_import_file(self, _button: Gtk.Button) -> None:
        chosen = self._select_file()
        if chosen:
            self._import_entry.set_text(chosen)
            self._update_status("")

    def _on_export_clicked(self, _button: Gtk.Button) -> None:
        target_text = self._export_entry.get_text().strip()
        if not target_text:
            self._update_status("Please choose an export folder first.")
            return

        result = self.ATLAS.export_user_backup(target_text)
        if result.get("success"):
            path = result.get("path") or target_text
            self._update_status(f"Export completed: {path}")
        else:
            self._update_status(result.get("error") or "Backup export failed.")

    def _on_import_clicked(self, _button: Gtk.Button) -> None:
        path_text = self._import_entry.get_text().strip()
        if not path_text:
            self._update_status("Please choose a backup file to import.")
            return

        if not Path(path_text).expanduser().exists():
            self._update_status("Selected backup file does not exist.")
            return

        result = self.ATLAS.import_user_backup(path_text)
        if result.get("success"):
            conversations = result.get("conversations", {}).get("conversations")
            message = result.get("message") or "Import completed."
            if isinstance(conversations, int):
                message = f"Imported {conversations} conversations successfully."
            self._update_status(message)
        else:
            self._update_status(result.get("error") or "Backup import failed.")
