from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Sequence

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from GTKUI.Utils.utils import apply_css
from modules.Tools.Base_Tools.vector_store import available_vector_store_adapters

logger = logging.getLogger(__name__)


class VectorStoreOverview(Gtk.ScrolledWindow):
    """Read-only overview of registered vector store adapters and defaults."""

    _DEFAULT_DESCRIPTIONS: Mapping[str, str] = {
        "in_memory": "Fast, ephemeral index suited for local testing or single-user desktops.",
        "mongodb": "MongoDB/Atlas vector search for teams already running MongoDB clusters.",
        "faiss": "FAISS-backed similarity search tuned for high-performance local or colocated hosts.",
        "chroma": "Chroma collections for lightweight managed or local vector storage with simple ops.",
        "pinecone": "Pinecone for a fully managed vector database when you want scale without self-hosting.",
    }

    def __init__(self, atlas: Any) -> None:
        super().__init__()
        self.ATLAS = atlas
        self._default_adapter = self._resolve_default_adapter()
        self._configured_adapters = self._get_configured_adapters()
        self._available_adapters = self._load_available_adapters()

        self.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.set_hexpand(True)
        self.set_vexpand(True)

        apply_css()

        container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        for setter_name in ("set_margin_top", "set_margin_bottom", "set_margin_start", "set_margin_end"):
            setter = getattr(container, setter_name, None)
            if callable(setter):
                try:
                    setter(12)
                except Exception:  # pragma: no cover - GTK fallback
                    continue
        container.set_hexpand(True)
        container.set_vexpand(True)

        heading = Gtk.Label(label="Vector store overview")
        heading.set_xalign(0.0)
        heading.set_wrap(True)
        if hasattr(heading, "add_css_class"):
            heading.add_css_class("heading")
        container.append(heading)

        summary = Gtk.Label(
            label=(
                "Review the available vector store adapters, see which one is active, and spot providers that "
                "need optional dependencies before they can be used."
            )
        )
        summary.set_xalign(0.0)
        summary.set_wrap(True)
        container.append(summary)

        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        status_box.set_hexpand(True)

        active_adapter_label = Gtk.Label(label=self._build_active_adapter_label())
        active_adapter_label.set_xalign(0.0)
        active_adapter_label.set_wrap(True)
        status_box.append(active_adapter_label)

        notes = self._collect_notes()
        if notes:
            for note in notes:
                note_label = Gtk.Label(label=f"• {note}")
                note_label.set_xalign(0.0)
                note_label.set_wrap(True)
                note_label.add_css_class("dim-label")
                status_box.append(note_label)

        status_frame = Gtk.Frame()
        status_frame.set_hexpand(True)
        status_frame.set_child(status_box)
        container.append(status_frame)

        adapter_frame = Gtk.Frame()
        adapter_frame.set_hexpand(True)
        adapter_frame.set_vexpand(True)
        adapter_frame.set_child(self._build_adapter_list())
        container.append(adapter_frame)

        self.set_child(container)

    def _collect_notes(self) -> Sequence[str]:
        notes = []
        config_manager = getattr(self.ATLAS, "config_manager", None)
        getter = getattr(config_manager, "get_vector_store_settings", None)
        if not callable(getter):
            notes.append("Configuration manager unavailable; showing defaults.")
        available_set = {name.strip().lower() for name in self._available_adapters if isinstance(name, str)}
        if self._default_adapter and self._default_adapter not in available_set:
            notes.append(
                f"Default adapter '{self._default_adapter}' is not registered; install optional dependencies "
                "or enable the provider."
            )
        return notes

    def _build_active_adapter_label(self) -> str:
        adapter_title = self._default_adapter.replace("_", " ").title()
        return f"Active adapter: {adapter_title}"

    def _build_adapter_list(self) -> Gtk.Widget:
        available_set = {name.strip().lower() for name in self._available_adapters if isinstance(name, str)}

        configured_names = self._configured_adapters
        known_names = [self._default_adapter] + list(self._DEFAULT_DESCRIPTIONS)

        all_names = self._merge_names(known_names, configured_names, available_set)

        listbox = Gtk.ListBox()
        listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        listbox.set_hexpand(True)
        listbox.add_css_class("sidebar-nav")

        for name in all_names:
            row = Gtk.ListBoxRow()
            row.set_activatable(False)
            row.add_css_class("sidebar-nav-item")

            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            box.set_margin_top(6)
            box.set_margin_bottom(6)
            box.set_margin_start(12)
            box.set_margin_end(12)

            name_label = Gtk.Label(label=name.replace("_", " ").title())
            name_label.set_xalign(0.0)
            name_label.set_wrap(True)
            box.append(name_label)

            description = self._DEFAULT_DESCRIPTIONS.get(
                name, "Registered by plugins or external providers."
            )
            description_label = Gtk.Label(label=description)
            description_label.set_xalign(0.0)
            description_label.set_wrap(True)
            description_label.add_css_class("dim-label")
            box.append(description_label)

            status_text = "Available"
            if name not in available_set:
                status_text = "Unavailable (missing optional dependencies or not registered)"
            if name == self._default_adapter and name not in available_set:
                status_text = "Configured as default, but unavailable (install optional dependencies or enable provider)"

            status_label = Gtk.Label(label=status_text)
            status_label.set_xalign(0.0)
            status_label.set_wrap(True)
            if name not in available_set:
                status_label.add_css_class("error")
            box.append(status_label)

            row.set_child(box)
            listbox.append(row)

        scroller = Gtk.ScrolledWindow()
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(listbox)
        return scroller

    def _load_available_adapters(self) -> Sequence[str]:
        available_adapters: Sequence[str] = ()
        try:
            available_adapters = available_vector_store_adapters()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unable to load registered vector adapters: %s", exc, exc_info=True)
        return available_adapters

    def _get_configured_adapters(self) -> Sequence[str]:
        config_manager = getattr(self.ATLAS, "config_manager", None)
        getter = getattr(config_manager, "get_vector_store_settings", None)
        configured: Dict[str, Any] = {}
        if callable(getter):
            try:
                settings = getter() or {}
                adapters = settings.get("adapters")
                if isinstance(adapters, Mapping):
                    configured = {k: v for k, v in adapters.items() if isinstance(k, str)}
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("Failed to read configured vector adapters: %s", exc, exc_info=True)
        return tuple(sorted(adapter.lower() for adapter in configured))

    def _resolve_default_adapter(self) -> str:
        default_adapter = "in_memory"
        config_manager = getattr(self.ATLAS, "config_manager", None)
        getter = getattr(config_manager, "get_vector_store_settings", None)
        if callable(getter):
            try:
                settings = getter() or {}
                if isinstance(settings, Mapping):
                    configured = settings.get("default_adapter")
                    if isinstance(configured, str) and configured.strip():
                        default_adapter = configured.strip().lower()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("Failed to read vector store settings: %s", exc, exc_info=True)
        return default_adapter

    def _merge_names(
        self,
        known: Sequence[str],
        configured: Iterable[str],
        available: Iterable[str],
    ) -> Sequence[str]:
        merged = []
        seen = set()
        for name in (*known, *configured, *available):
            key = name.strip().lower()
            if not key or key in seen:
                continue
            merged.append(key)
            seen.add(key)
        return merged
